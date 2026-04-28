use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};

use agent_client_protocol::{
    AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate, Client, ClientCapabilities,
    ConfigOptionUpdate, Content, ContentBlock, ContentChunk, Diff, EmbeddedResource,
    EmbeddedResourceResource, Error, ExtRequest, LoadSessionResponse, Meta, ModelId, ModelInfo,
    PermissionOption, PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    ResourceLink, SelectedPermissionOutcome, SessionConfigId, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigOptionValue, SessionConfigSelectOption,
    SessionConfigValueId, SessionId, SessionInfoUpdate, SessionMode, SessionModeId,
    SessionModeState, SessionModelState, SessionNotification, SessionUpdate, StopReason, Terminal,
    TextContent, TextResourceContents, ToolCall, ToolCallContent, ToolCallId, ToolCallLocation,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind, UnstructuredCommandInput,
    UsageUpdate,
};
use codex_apply_patch::parse_patch;
use codex_core::{
    CodexThread,
    config::{Config, set_project_trust_level},
    review_format::format_review_findings_block,
    review_prompts::user_facing_hint,
};
use codex_login::auth::AuthManager;
use codex_models_manager::manager::{ModelsManager, RefreshStrategy};
use codex_protocol::{
    approvals::{
        ElicitationRequest, ElicitationRequestEvent, GuardianAssessmentAction,
        GuardianCommandSource,
    },
    config_types::{
        CollaborationMode, CollaborationModeMask, ModeKind, ServiceTier, Settings, TrustLevel,
    },
    dynamic_tools::{DynamicToolCallOutputContentItem, DynamicToolCallRequest},
    error::CodexErr,
    items::TurnItem,
    mcp::CallToolResult,
    models::{AdditionalPermissionProfile, ResponseItem, WebSearchAction},
    openai_models::{ModelPreset, ReasoningEffort},
    parse_command::ParsedCommand,
    plan_tool::{PlanItemArg, StepStatus, UpdatePlanArgs},
    protocol::{
        AgentMessageContentDeltaEvent, AgentMessageEvent, AgentReasoningEvent,
        AgentReasoningRawContentEvent, AgentReasoningSectionBreakEvent,
        ApplyPatchApprovalRequestEvent, DynamicToolCallResponseEvent, ElicitationAction,
        ErrorEvent, Event, EventMsg, ExecApprovalRequestEvent, ExecCommandBeginEvent,
        ExecCommandEndEvent, ExecCommandOutputDeltaEvent, ExecCommandStatus, ExitedReviewModeEvent,
        FileChange, GuardianAssessmentEvent, GuardianAssessmentStatus, HookCompletedEvent,
        HookEventName, HookOutputEntryKind, HookRunStatus, HookRunSummary, HookStartedEvent,
        ItemCompletedEvent, ItemStartedEvent, McpInvocation, McpStartupCompleteEvent,
        McpStartupUpdateEvent, McpToolCallBeginEvent, McpToolCallEndEvent, ModelRerouteEvent,
        NetworkApprovalContext, NetworkPolicyRuleAction, Op, PatchApplyBeginEvent,
        PatchApplyEndEvent, PatchApplyStatus, ReasoningContentDeltaEvent,
        ReasoningRawContentDeltaEvent, ReviewDecision, ReviewOutputEvent, ReviewRequest,
        ReviewTarget, RolloutItem, SandboxPolicy, StreamErrorEvent, TerminalInteractionEvent,
        TokenCountEvent, TurnAbortedEvent, TurnCompleteEvent, TurnStartedEvent, UserMessageEvent,
        ViewImageToolCallEvent, WarningEvent, WebSearchBeginEvent, WebSearchEndEvent,
    },
    request_permissions::{
        PermissionGrantScope, RequestPermissionProfile, RequestPermissionsEvent,
        RequestPermissionsResponse,
    },
    request_user_input::{
        RequestUserInputAnswer, RequestUserInputEvent, RequestUserInputQuestion,
        RequestUserInputQuestionOption, RequestUserInputResponse,
    },
    user_input::UserInput,
};
use codex_shell_command::parse_command::parse_command;
use codex_utils_approval_presets::{ApprovalPreset, builtin_approval_presets};
use heck::ToTitleCase;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, value::RawValue};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::prompt_args::{
    CustomPrompt, discover_prompts_in, expand_custom_prompt, parse_slash_name,
};

use crate::ACP_CLIENT;

static APPROVAL_PRESETS: LazyLock<Vec<ApprovalPreset>> = LazyLock::new(builtin_approval_presets);
const INIT_COMMAND_PROMPT: &str = include_str!("./prompt_for_init_command.md");
const ANYHARNESS_META_KEY: &str = "anyharness";
const ANYHARNESS_ASSISTANT_MESSAGE_COMPLETED_EVENT: &str = "assistant_message_completed";
const ANYHARNESS_PROPOSED_PLAN_DELTA_EVENT: &str = "proposed_plan_delta";
const ANYHARNESS_PROPOSED_PLAN_COMPLETED_EVENT: &str = "proposed_plan_completed";
const ANYHARNESS_TRANSIENT_STATUS_EVENT: &str = "transient_status";
const CODEX_REQUEST_USER_INPUT_EXT_METHOD: &str = "experimental/codex/requestUserInput";
const CODEX_MCP_ELICITATION_EXT_METHOD: &str = "experimental/codex/mcpElicitation";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputExtParams {
    call_id: String,
    turn_id: String,
    questions: Vec<CodexRequestUserInputExtQuestion>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputExtQuestion {
    question_id: String,
    header: String,
    question: String,
    is_other: bool,
    is_secret: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    options: Vec<CodexRequestUserInputExtOption>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputExtOption {
    label: String,
    description: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputExtResponse {
    outcome: CodexRequestUserInputExtOutcome,
    #[serde(default)]
    answers: Vec<CodexRequestUserInputExtAnswer>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CodexRequestUserInputExtOutcome {
    Submitted,
    Cancelled,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputExtAnswer {
    question_id: String,
    selected_option_label: Option<String>,
    text: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexMcpElicitationExtParams {
    server_name: String,
    request: ElicitationRequest,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexMcpElicitationExtResponse {
    outcome: CodexMcpElicitationExtOutcome,
    #[serde(default)]
    content: Option<Value>,
    #[serde(rename = "_meta", default)]
    meta: Option<Value>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CodexMcpElicitationExtOutcome {
    Accepted,
    Declined,
    Cancelled,
}

/// Trait for abstracting over the `CodexThread` to make testing easier.
#[async_trait::async_trait]
pub trait CodexThreadImpl {
    async fn submit(&self, op: Op) -> Result<String, CodexErr>;
    async fn next_event(&self) -> Result<Event, CodexErr>;
}

#[async_trait::async_trait]
impl CodexThreadImpl for CodexThread {
    async fn submit(&self, op: Op) -> Result<String, CodexErr> {
        self.submit(op).await
    }

    async fn next_event(&self) -> Result<Event, CodexErr> {
        self.next_event().await
    }
}

#[async_trait::async_trait]
pub trait ModelsManagerImpl {
    async fn get_model(&self, model_id: &Option<String>) -> String;
    async fn list_models(&self) -> Vec<ModelPreset>;
    fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask>;
}

pub fn models_manager_adapter(
    models_manager: Arc<dyn ModelsManager>,
) -> Arc<dyn ModelsManagerImpl> {
    Arc::new(ModelsManagerAdapter(models_manager))
}

struct ModelsManagerAdapter(Arc<dyn ModelsManager>);

#[async_trait::async_trait]
impl ModelsManagerImpl for ModelsManagerAdapter {
    async fn get_model(&self, model_id: &Option<String>) -> String {
        self.0
            .get_default_model(model_id, RefreshStrategy::OnlineIfUncached)
            .await
    }

    async fn list_models(&self) -> Vec<ModelPreset> {
        self.0.list_models(RefreshStrategy::OnlineIfUncached).await
    }

    fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask> {
        self.0.list_collaboration_modes()
    }
}

pub trait Auth {
    fn logout(&self) -> Result<bool, Error>;
}

impl Auth for Arc<AuthManager> {
    fn logout(&self) -> Result<bool, Error> {
        self.as_ref()
            .logout()
            .map_err(|e| Error::internal_error().data(e.to_string()))
    }
}

enum ThreadMessage {
    Load {
        response_tx: oneshot::Sender<Result<LoadSessionResponse, Error>>,
    },
    GetConfigOptions {
        response_tx: oneshot::Sender<Result<Vec<SessionConfigOption>, Error>>,
    },
    Prompt {
        request: PromptRequest,
        response_tx: oneshot::Sender<Result<oneshot::Receiver<Result<StopReason, Error>>, Error>>,
    },
    SetMode {
        mode: SessionModeId,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    SetModel {
        model: ModelId,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    SetConfigOption {
        config_id: SessionConfigId,
        value: SessionConfigOptionValue,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    Cancel {
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    Shutdown {
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    ReplayHistory {
        history: Vec<RolloutItem>,
        response_tx: oneshot::Sender<Result<(), Error>>,
    },
    PermissionRequestResolved {
        submission_id: String,
        request_key: String,
        response: Result<RequestPermissionResponse, Error>,
    },
}

pub struct Thread {
    /// Direct handle to the underlying Codex thread for out-of-band shutdown.
    thread: Arc<dyn CodexThreadImpl>,
    /// A sender for interacting with the thread.
    message_tx: mpsc::UnboundedSender<ThreadMessage>,
    /// Keep the actor task alive for the lifetime of the thread wrapper.
    _handle: tokio::task::JoinHandle<()>,
}

impl Thread {
    pub fn new(
        session_id: SessionId,
        thread: Arc<dyn CodexThreadImpl>,
        auth: Arc<AuthManager>,
        models_manager: Arc<dyn ModelsManagerImpl>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        config: Config,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (resolution_tx, resolution_rx) = mpsc::unbounded_channel();

        let actor = ThreadActor::new(
            auth,
            SessionClient::new(session_id, client_capabilities),
            thread.clone(),
            models_manager,
            config,
            message_rx,
            resolution_tx,
            resolution_rx,
        );
        let handle = tokio::task::spawn_local(actor.spawn());

        Self {
            thread,
            message_tx,
            _handle: handle,
        }
    }

    pub async fn load(&self) -> Result<LoadSessionResponse, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Load { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn config_options(&self) -> Result<Vec<SessionConfigOption>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::GetConfigOptions { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn prompt(&self, request: PromptRequest) -> Result<StopReason, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Prompt {
            request,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))??
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_mode(&self, mode: SessionModeId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetMode { mode, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_model(&self, model: ModelId) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetModel { model, response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn set_config_option(
        &self,
        config_id: SessionConfigId,
        value: SessionConfigOptionValue,
    ) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::SetConfigOption {
            config_id,
            value,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn cancel(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::Cancel { response_tx };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn replay_history(&self, history: Vec<RolloutItem>) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let message = ThreadMessage::ReplayHistory {
            history,
            response_tx,
        };
        drop(self.message_tx.send(message));

        response_rx
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?
    }

    pub async fn shutdown(&self) -> Result<(), Error> {
        let (response_tx, response_rx) = oneshot::channel();
        let message = ThreadMessage::Shutdown { response_tx };

        if self.message_tx.send(message).is_err() {
            self.thread
                .submit(Op::Shutdown)
                .await
                .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        } else {
            response_rx
                .await
                .map_err(|e| Error::internal_error().data(e.to_string()))??;
        }
        // Let the actor drain the resulting turn-aborted/shutdown events so any in-flight
        // prompt callers observe a clean cancellation instead of a dropped response channel.
        Ok(())
    }
}

enum PendingPermissionRequest {
    Exec {
        approval_id: String,
        turn_id: String,
        option_map: HashMap<String, ReviewDecision>,
    },
    Patch {
        call_id: String,
        option_map: HashMap<String, ReviewDecision>,
    },
    RequestPermissions {
        call_id: String,
        option_map: HashMap<String, RequestPermissionsResponse>,
    },
    McpElicitation {
        server_name: String,
        request_id: codex_protocol::mcp::RequestId,
        option_map: HashMap<String, ResolvedMcpElicitation>,
    },
}

struct PendingPermissionInteraction {
    request: PendingPermissionRequest,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct ResolvedMcpElicitation {
    action: ElicitationAction,
    content: Option<serde_json::Value>,
    meta: Option<serde_json::Value>,
}

impl ResolvedMcpElicitation {
    fn accept() -> Self {
        Self {
            action: ElicitationAction::Accept,
            content: None,
            meta: None,
        }
    }

    fn accept_with_persist(persist: &'static str) -> Self {
        Self {
            action: ElicitationAction::Accept,
            content: None,
            meta: Some(serde_json::json!({ "persist": persist })),
        }
    }

    fn decline() -> Self {
        Self {
            action: ElicitationAction::Decline,
            content: None,
            meta: None,
        }
    }

    fn cancel() -> Self {
        Self {
            action: ElicitationAction::Cancel,
            content: None,
            meta: None,
        }
    }
}

fn exec_request_key(call_id: &str) -> String {
    format!("exec:{call_id}")
}

fn patch_request_key(call_id: &str) -> String {
    format!("patch:{call_id}")
}

fn permissions_request_key(call_id: &str) -> String {
    format!("permissions:{call_id}")
}

fn mcp_elicitation_request_key(
    server_name: &str,
    request_id: &codex_protocol::mcp::RequestId,
) -> String {
    format!("mcp-elicitation:{server_name}:{request_id}")
}

const MCP_TOOL_APPROVAL_KIND_KEY: &str = "codex_approval_kind";
const MCP_TOOL_APPROVAL_KIND_MCP_TOOL_CALL: &str = "mcp_tool_call";
const MCP_TOOL_APPROVAL_PERSIST_KEY: &str = "persist";
const MCP_TOOL_APPROVAL_PERSIST_SESSION: &str = "session";
const MCP_TOOL_APPROVAL_PERSIST_ALWAYS: &str = "always";
const MCP_TOOL_APPROVAL_TOOL_TITLE_KEY: &str = "tool_title";
const MCP_TOOL_APPROVAL_TOOL_DESCRIPTION_KEY: &str = "tool_description";
const MCP_TOOL_APPROVAL_CONNECTOR_NAME_KEY: &str = "connector_name";
const MCP_TOOL_APPROVAL_CONNECTOR_DESCRIPTION_KEY: &str = "connector_description";
const MCP_TOOL_APPROVAL_TOOL_PARAMS_KEY: &str = "tool_params";
const MCP_TOOL_APPROVAL_TOOL_PARAMS_DISPLAY_KEY: &str = "tool_params_display";
const MCP_TOOL_APPROVAL_REQUEST_ID_PREFIX: &str = "mcp_tool_call_approval_";
const MCP_TOOL_APPROVAL_ALLOW_OPTION_ID: &str = "approved";
const MCP_TOOL_APPROVAL_ALLOW_SESSION_OPTION_ID: &str = "approved-for-session";
const MCP_TOOL_APPROVAL_ALLOW_ALWAYS_OPTION_ID: &str = "approved-always";
const MCP_TOOL_APPROVAL_CANCEL_OPTION_ID: &str = "cancel";
const MCP_ELICITATION_ACCEPT_OPTION_ID: &str = "accept";
const MCP_ELICITATION_DECLINE_OPTION_ID: &str = "decline";
const MCP_ELICITATION_CANCEL_OPTION_ID: &str = "cancel";
const REQUEST_PERMISSIONS_ALLOW_SESSION_OPTION_ID: &str = "approved-for-session";
const REQUEST_PERMISSIONS_ALLOW_TURN_OPTION_ID: &str = "approved";
const REQUEST_PERMISSIONS_ALLOW_TURN_STRICT_OPTION_ID: &str = "approved-with-strict-auto-review";
const REQUEST_PERMISSIONS_DENY_OPTION_ID: &str = "abort";

struct SupportedMcpElicitationPermissionRequest {
    request_key: String,
    tool_call: ToolCallUpdate,
    options: Vec<PermissionOption>,
    option_map: HashMap<String, ResolvedMcpElicitation>,
}

fn build_supported_mcp_elicitation_permission_request(
    server_name: &str,
    request_id: &codex_protocol::mcp::RequestId,
    request: &ElicitationRequest,
    raw_input: serde_json::Value,
) -> Option<SupportedMcpElicitationPermissionRequest> {
    let ElicitationRequest::Form {
        meta,
        message,
        requested_schema,
    } = request
    else {
        return None;
    };

    if !is_message_only_elicitation_schema(requested_schema) {
        return None;
    }

    let meta = meta
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let approval_kind = meta
        .get(MCP_TOOL_APPROVAL_KIND_KEY)
        .and_then(serde_json::Value::as_str);
    if approval_kind.is_some() && approval_kind != Some(MCP_TOOL_APPROVAL_KIND_MCP_TOOL_CALL) {
        return None;
    }

    let (tool_call_id, title, content, options, option_map) =
        if approval_kind == Some(MCP_TOOL_APPROVAL_KIND_MCP_TOOL_CALL) {
            build_mcp_tool_approval_permission(server_name, request_id, message, &meta)
        } else {
            build_message_only_mcp_permission(request_id, message)
        };

    Some(SupportedMcpElicitationPermissionRequest {
        request_key: mcp_elicitation_request_key(server_name, request_id),
        tool_call: ToolCallUpdate::new(
            ToolCallId::new(tool_call_id),
            ToolCallUpdateFields::new()
                .status(ToolCallStatus::Pending)
                .title(title)
                .content(vec![ToolCallContent::Content(Content::new(
                    ContentBlock::Text(TextContent::new(content)),
                ))])
                .raw_input(raw_input),
        ),
        options,
        option_map,
    })
}

fn build_mcp_tool_approval_permission(
    server_name: &str,
    request_id: &codex_protocol::mcp::RequestId,
    message: &str,
    meta: &serde_json::Map<String, serde_json::Value>,
) -> (
    String,
    String,
    String,
    Vec<PermissionOption>,
    HashMap<String, ResolvedMcpElicitation>,
) {
    let (allow_session_remember, allow_persistent_approval) = mcp_tool_approval_persist_modes(meta);
    let mut options = vec![PermissionOption::new(
        MCP_TOOL_APPROVAL_ALLOW_OPTION_ID,
        "Allow",
        PermissionOptionKind::AllowOnce,
    )];
    let mut option_map = HashMap::from([(
        MCP_TOOL_APPROVAL_ALLOW_OPTION_ID.to_string(),
        ResolvedMcpElicitation::accept(),
    )]);

    if allow_session_remember {
        options.push(PermissionOption::new(
            MCP_TOOL_APPROVAL_ALLOW_SESSION_OPTION_ID,
            "Allow for this session",
            PermissionOptionKind::AllowAlways,
        ));
        option_map.insert(
            MCP_TOOL_APPROVAL_ALLOW_SESSION_OPTION_ID.to_string(),
            ResolvedMcpElicitation::accept_with_persist(MCP_TOOL_APPROVAL_PERSIST_SESSION),
        );
    }

    if allow_persistent_approval {
        options.push(PermissionOption::new(
            MCP_TOOL_APPROVAL_ALLOW_ALWAYS_OPTION_ID,
            "Allow and don't ask again",
            PermissionOptionKind::AllowAlways,
        ));
        option_map.insert(
            MCP_TOOL_APPROVAL_ALLOW_ALWAYS_OPTION_ID.to_string(),
            ResolvedMcpElicitation::accept_with_persist(MCP_TOOL_APPROVAL_PERSIST_ALWAYS),
        );
    }

    options.push(PermissionOption::new(
        MCP_TOOL_APPROVAL_CANCEL_OPTION_ID,
        "Cancel",
        PermissionOptionKind::RejectOnce,
    ));
    option_map.insert(
        MCP_TOOL_APPROVAL_CANCEL_OPTION_ID.to_string(),
        ResolvedMcpElicitation::cancel(),
    );

    let tool_call_id = mcp_tool_approval_call_id(request_id)
        .unwrap_or_else(|| format!("mcp-elicitation:{request_id}"));
    let title = meta
        .get(MCP_TOOL_APPROVAL_TOOL_TITLE_KEY)
        .and_then(serde_json::Value::as_str)
        .filter(|title| !title.trim().is_empty())
        .map(|title| format!("Approve {title}"))
        .unwrap_or_else(|| "Approve MCP tool call".to_string());
    let content = format_mcp_tool_approval_content(server_name, message, meta);

    (tool_call_id, title, content, options, option_map)
}

fn build_message_only_mcp_permission(
    request_id: &codex_protocol::mcp::RequestId,
    message: &str,
) -> (
    String,
    String,
    String,
    Vec<PermissionOption>,
    HashMap<String, ResolvedMcpElicitation>,
) {
    let options = vec![
        PermissionOption::new(
            MCP_ELICITATION_ACCEPT_OPTION_ID,
            "Allow",
            PermissionOptionKind::AllowOnce,
        ),
        PermissionOption::new(
            MCP_ELICITATION_DECLINE_OPTION_ID,
            "Deny",
            PermissionOptionKind::RejectOnce,
        ),
        PermissionOption::new(
            MCP_ELICITATION_CANCEL_OPTION_ID,
            "Cancel",
            PermissionOptionKind::RejectOnce,
        ),
    ];
    let option_map = HashMap::from([
        (
            MCP_ELICITATION_ACCEPT_OPTION_ID.to_string(),
            ResolvedMcpElicitation::accept(),
        ),
        (
            MCP_ELICITATION_DECLINE_OPTION_ID.to_string(),
            ResolvedMcpElicitation::decline(),
        ),
        (
            MCP_ELICITATION_CANCEL_OPTION_ID.to_string(),
            ResolvedMcpElicitation::cancel(),
        ),
    ]);

    (
        format!("mcp-elicitation:{request_id}"),
        "Approve MCP request".to_string(),
        message.trim().to_string(),
        options,
        option_map,
    )
}

fn is_message_only_elicitation_schema(schema: &serde_json::Value) -> bool {
    let Some(schema) = schema.as_object() else {
        return false;
    };

    if schema.get("type").and_then(serde_json::Value::as_str) != Some("object") {
        return false;
    }
    if !schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .is_some_and(serde_json::Map::is_empty)
    {
        return false;
    }
    if schema.get("additionalProperties") != Some(&serde_json::Value::Bool(false)) {
        return false;
    }
    if schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|required| !required.is_empty())
    {
        return false;
    }

    [
        "allOf",
        "anyOf",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "oneOf",
        "patternProperties",
        "propertyNames",
        "then",
        "unevaluatedProperties",
    ]
    .iter()
    .all(|key| !schema.contains_key(*key))
}

fn denied_request_permissions_response() -> RequestPermissionsResponse {
    RequestPermissionsResponse {
        permissions: RequestPermissionProfile::default(),
        scope: PermissionGrantScope::Turn,
        strict_auto_review: false,
    }
}

fn request_permissions_option_map(
    permissions: &RequestPermissionProfile,
) -> HashMap<String, RequestPermissionsResponse> {
    HashMap::from([
        (
            REQUEST_PERMISSIONS_ALLOW_SESSION_OPTION_ID.to_string(),
            RequestPermissionsResponse {
                permissions: permissions.clone(),
                scope: PermissionGrantScope::Session,
                strict_auto_review: false,
            },
        ),
        (
            REQUEST_PERMISSIONS_ALLOW_TURN_OPTION_ID.to_string(),
            RequestPermissionsResponse {
                permissions: permissions.clone(),
                scope: PermissionGrantScope::Turn,
                strict_auto_review: false,
            },
        ),
        (
            REQUEST_PERMISSIONS_ALLOW_TURN_STRICT_OPTION_ID.to_string(),
            RequestPermissionsResponse {
                permissions: permissions.clone(),
                scope: PermissionGrantScope::Turn,
                strict_auto_review: true,
            },
        ),
        (
            REQUEST_PERMISSIONS_DENY_OPTION_ID.to_string(),
            denied_request_permissions_response(),
        ),
    ])
}

fn mcp_tool_approval_persist_modes(
    meta: &serde_json::Map<String, serde_json::Value>,
) -> (bool, bool) {
    match meta.get(MCP_TOOL_APPROVAL_PERSIST_KEY) {
        Some(serde_json::Value::String(persist)) => (
            persist == MCP_TOOL_APPROVAL_PERSIST_SESSION,
            persist == MCP_TOOL_APPROVAL_PERSIST_ALWAYS,
        ),
        Some(serde_json::Value::Array(values)) => (
            values
                .iter()
                .any(|value| value.as_str() == Some(MCP_TOOL_APPROVAL_PERSIST_SESSION)),
            values
                .iter()
                .any(|value| value.as_str() == Some(MCP_TOOL_APPROVAL_PERSIST_ALWAYS)),
        ),
        _ => (false, false),
    }
}

fn mcp_tool_approval_call_id(request_id: &codex_protocol::mcp::RequestId) -> Option<String> {
    match request_id {
        codex_protocol::mcp::RequestId::String(value) => value
            .strip_prefix(MCP_TOOL_APPROVAL_REQUEST_ID_PREFIX)
            .map(ToString::to_string),
        codex_protocol::mcp::RequestId::Integer(_) => None,
    }
}

fn format_mcp_tool_approval_content(
    server_name: &str,
    message: &str,
    meta: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let mut sections = vec![message.trim().to_string()];

    let source = meta
        .get(MCP_TOOL_APPROVAL_CONNECTOR_NAME_KEY)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(|value| format!("Source: {value}"))
        .unwrap_or_else(|| format!("Server: {server_name}"));
    sections.push(source);

    if let Some(description) = meta
        .get(MCP_TOOL_APPROVAL_CONNECTOR_DESCRIPTION_KEY)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(description.to_string());
    }

    if let Some(description) = meta
        .get(MCP_TOOL_APPROVAL_TOOL_DESCRIPTION_KEY)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(description.to_string());
    }

    if let Some(params) = format_mcp_tool_approval_params(meta) {
        sections.push(format!("Arguments:\n{params}"));
    }

    sections.join("\n\n")
}

fn format_mcp_tool_approval_params(
    meta: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    if let Some(serde_json::Value::Array(params)) =
        meta.get(MCP_TOOL_APPROVAL_TOOL_PARAMS_DISPLAY_KEY)
    {
        let params = params
            .iter()
            .filter_map(|param| {
                let object = param.as_object()?;
                let name = object
                    .get("display_name")
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| object.get("name").and_then(serde_json::Value::as_str))?;
                let value = object.get("value")?;
                Some(format!(
                    "- {name}: {}",
                    format_mcp_tool_approval_value(value)
                ))
            })
            .collect::<Vec<_>>();
        if !params.is_empty() {
            return Some(params.join("\n"));
        }
    }

    meta.get(MCP_TOOL_APPROVAL_TOOL_PARAMS_KEY).map(|params| {
        serde_json::to_string_pretty(params)
            .unwrap_or_else(|_| format_mcp_tool_approval_value(params))
    })
}

fn format_mcp_tool_approval_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(value) => value.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    }
}

enum SubmissionState {
    /// User prompts, including slash commands like /init, /review, /compact, /undo.
    Prompt(PromptState),
}

impl SubmissionState {
    fn is_active(&self) -> bool {
        match self {
            Self::Prompt(state) => state.is_active(),
        }
    }

    fn current_turn_id(&self) -> Option<&str> {
        match self {
            Self::Prompt(state) => state.diagnostics.current_turn_id.as_deref(),
        }
    }

    async fn handle_event(&mut self, client: &SessionClient, event: EventMsg) {
        match self {
            Self::Prompt(state) => state.handle_event(client, event).await,
        }
    }

    async fn handle_permission_request_resolved(
        &mut self,
        client: &SessionClient,
        request_key: String,
        response: Result<RequestPermissionResponse, Error>,
    ) -> Result<(), Error> {
        match self {
            Self::Prompt(state) => {
                state
                    .handle_permission_request_resolved(client, request_key, response)
                    .await
            }
        }
    }

    fn abort_pending_interactions(&mut self) {
        let Self::Prompt(state) = self;
        state.abort_pending_interactions();
    }

    fn log_pending_diagnostics(&self) {
        let Self::Prompt(state) = self;
        state.log_pending_diagnostics();
    }

    fn fail(&mut self, err: Error) {
        match self {
            Self::Prompt(state) => {
                if let Some(response_tx) = state.response_tx.take() {
                    drop(response_tx.send(Err(err)));
                }
            }
        }
    }
}

struct ActiveCommand {
    tool_call_id: ToolCallId,
    terminal_output: bool,
    output: String,
    file_extension: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct PromptPlanStatusSummary {
    pending: usize,
    in_progress: usize,
    completed: usize,
}

#[derive(Debug, Clone)]
struct PromptDiagnostics {
    submitted_at: Instant,
    current_turn_id: Option<String>,
    last_event_kind: Option<&'static str>,
    last_event_at: Option<Instant>,
    last_agent_chunk_at: Option<Instant>,
    last_agent_preview: Option<String>,
    last_plan_at: Option<Instant>,
    plan_statuses: PromptPlanStatusSummary,
    turn_complete_seen_at: Option<Instant>,
    response_sent_at: Option<Instant>,
    stream_error_seen_at: Option<Instant>,
}

impl PromptDiagnostics {
    fn new() -> Self {
        Self {
            submitted_at: Instant::now(),
            current_turn_id: None,
            last_event_kind: None,
            last_event_at: None,
            last_agent_chunk_at: None,
            last_agent_preview: None,
            last_plan_at: None,
            plan_statuses: PromptPlanStatusSummary::default(),
            turn_complete_seen_at: None,
            response_sent_at: None,
            stream_error_seen_at: None,
        }
    }

    fn note_event(&mut self, kind: &'static str) {
        self.last_event_kind = Some(kind);
        self.last_event_at = Some(Instant::now());
    }

    fn note_turn_started(&mut self, turn_id: &str) {
        self.current_turn_id = Some(turn_id.to_string());
        self.note_event("turn_started");
    }

    fn note_agent_output(&mut self, kind: &'static str, text: &str) {
        let now = Instant::now();
        self.last_event_kind = Some(kind);
        self.last_event_at = Some(now);
        self.last_agent_chunk_at = Some(now);
        self.last_agent_preview = Some(truncate_preview(text, 120));
    }

    fn note_plan_update(&mut self, plan: &[PlanItemArg]) {
        self.note_event("plan_update");
        self.last_plan_at = self.last_event_at;
        self.plan_statuses = summarize_plan_statuses(plan);
    }

    fn note_stream_error(&mut self) {
        self.note_event("stream_error");
        self.stream_error_seen_at = self.last_event_at;
    }

    fn note_turn_complete(&mut self, turn_id: &str) {
        self.current_turn_id = Some(turn_id.to_string());
        self.note_event("turn_complete");
        self.turn_complete_seen_at = self.last_event_at;
    }

    fn note_response_sent(&mut self) {
        self.response_sent_at = Some(Instant::now());
    }

    fn log_pending(
        &self,
        submission_id: &str,
        active_command_count: usize,
        active_web_search: bool,
        active_guardian_assessment_count: usize,
        pending_permission_count: usize,
        event_count: usize,
    ) {
        let now = Instant::now();
        info!(
            submission_id = submission_id,
            turn_id = ?self.current_turn_id,
            pending_for_ms = self.submitted_at.elapsed().as_millis() as u64,
            event_count = event_count,
            last_event_kind = ?self.last_event_kind,
            last_event_age_ms = age_ms(self.last_event_at, now),
            last_agent_chunk_age_ms = age_ms(self.last_agent_chunk_at, now),
            last_agent_preview = self.last_agent_preview.as_deref().unwrap_or(""),
            last_plan_age_ms = age_ms(self.last_plan_at, now),
            turn_complete_seen_age_ms = age_ms(self.turn_complete_seen_at, now),
            response_sent_age_ms = age_ms(self.response_sent_at, now),
            stream_error_seen_age_ms = age_ms(self.stream_error_seen_at, now),
            plan_pending = self.plan_statuses.pending,
            plan_in_progress = self.plan_statuses.in_progress,
            plan_completed = self.plan_statuses.completed,
            active_command_count = active_command_count,
            active_web_search = active_web_search,
            active_guardian_assessment_count = active_guardian_assessment_count,
            pending_permission_count = pending_permission_count,
            "codex_acp.prompt.pending"
        );
    }
}

fn summarize_plan_statuses(plan: &[PlanItemArg]) -> PromptPlanStatusSummary {
    let mut summary = PromptPlanStatusSummary::default();
    for item in plan {
        match item.status {
            StepStatus::Pending => summary.pending += 1,
            StepStatus::InProgress => summary.in_progress += 1,
            StepStatus::Completed => summary.completed += 1,
        }
    }
    summary
}

fn truncate_preview(text: &str, max_chars: usize) -> String {
    let mut preview = String::new();
    for ch in text.chars().take(max_chars) {
        preview.push(ch);
    }
    if text.chars().count() > max_chars {
        preview.push_str("...");
    }
    preview
}

fn age_ms(since: Option<Instant>, now: Instant) -> u64 {
    since
        .map(|instant| now.saturating_duration_since(instant).as_millis() as u64)
        .unwrap_or(0)
}

struct PromptState {
    submission_id: String,
    active_commands: HashMap<String, ActiveCommand>,
    active_web_search: Option<String>,
    active_guardian_assessments: HashSet<String>,
    thread: Arc<dyn CodexThreadImpl>,
    resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
    pending_permission_interactions: HashMap<String, PendingPermissionInteraction>,
    event_count: usize,
    response_tx: Option<oneshot::Sender<Result<StopReason, Error>>>,
    seen_message_deltas: bool,
    seen_reasoning_deltas: bool,
    agent_message_ids_by_item_id: HashMap<String, String>,
    transient_status_message_id: Option<String>,
    diagnostics: PromptDiagnostics,
}

impl PromptState {
    fn new(
        submission_id: String,
        thread: Arc<dyn CodexThreadImpl>,
        resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
        response_tx: oneshot::Sender<Result<StopReason, Error>>,
    ) -> Self {
        Self {
            submission_id,
            active_commands: HashMap::new(),
            active_web_search: None,
            active_guardian_assessments: HashSet::new(),
            thread,
            resolution_tx,
            pending_permission_interactions: HashMap::new(),
            event_count: 0,
            response_tx: Some(response_tx),
            seen_message_deltas: false,
            seen_reasoning_deltas: false,
            agent_message_ids_by_item_id: HashMap::new(),
            transient_status_message_id: None,
            diagnostics: PromptDiagnostics::new(),
        }
    }

    fn is_active(&self) -> bool {
        let Some(response_tx) = &self.response_tx else {
            return false;
        };
        !response_tx.is_closed()
    }

    fn agent_message_id_for_item(&mut self, item_id: &str) -> String {
        self.agent_message_ids_by_item_id
            .entry(item_id.to_string())
            .or_insert_with(|| Uuid::new_v4().to_string())
            .clone()
    }

    fn transient_status_message_id(&mut self) -> String {
        self.transient_status_message_id
            .get_or_insert_with(|| Uuid::new_v4().to_string())
            .clone()
    }

    fn abort_pending_interactions(&mut self) {
        for (_, interaction) in self.pending_permission_interactions.drain() {
            interaction.task.abort();
        }
    }

    fn log_pending_diagnostics(&self) {
        self.diagnostics.log_pending(
            &self.submission_id,
            self.active_commands.len(),
            self.active_web_search.is_some(),
            self.active_guardian_assessments.len(),
            self.pending_permission_interactions.len(),
            self.event_count,
        );
    }

    fn spawn_permission_request(
        &mut self,
        client: &SessionClient,
        request_key: String,
        pending_request: PendingPermissionRequest,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) {
        let client = client.clone();
        let resolution_tx = self.resolution_tx.clone();
        let submission_id = self.submission_id.clone();
        let resolved_request_key = request_key.clone();
        let handle = tokio::task::spawn_local(async move {
            let response = client.request_permission(tool_call, options).await;
            drop(
                resolution_tx.send(ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key: resolved_request_key,
                    response,
                }),
            );
        });

        if let Some(interaction) = self.pending_permission_interactions.insert(
            request_key,
            PendingPermissionInteraction {
                request: pending_request,
                task: handle,
            },
        ) {
            interaction.task.abort();
        }
    }

    async fn handle_permission_request_resolved(
        &mut self,
        _client: &SessionClient,
        request_key: String,
        response: Result<RequestPermissionResponse, Error>,
    ) -> Result<(), Error> {
        let Some(interaction) = self.pending_permission_interactions.remove(&request_key) else {
            warn!("Ignoring permission response for unknown request key: {request_key}");
            return Ok(());
        };
        let pending_request = interaction.request;
        let response = response?;

        match pending_request {
            PendingPermissionRequest::Exec {
                approval_id,
                turn_id,
                option_map,
            } => {
                let decision = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => option_map
                        .get(option_id.0.as_ref())
                        .cloned()
                        .unwrap_or(ReviewDecision::Abort),
                    RequestPermissionOutcome::Cancelled | _ => ReviewDecision::Abort,
                };

                self.thread
                    .submit(Op::ExecApproval {
                        id: approval_id,
                        turn_id: Some(turn_id),
                        decision,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
            PendingPermissionRequest::Patch {
                call_id,
                option_map,
            } => {
                let decision = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => option_map
                        .get(option_id.0.as_ref())
                        .cloned()
                        .unwrap_or(ReviewDecision::Abort),
                    RequestPermissionOutcome::Cancelled | _ => ReviewDecision::Abort,
                };

                self.thread
                    .submit(Op::PatchApproval {
                        id: call_id,
                        decision,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
            PendingPermissionRequest::RequestPermissions {
                call_id,
                option_map,
            } => {
                let response = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => option_map
                        .get(option_id.0.as_ref())
                        .cloned()
                        .unwrap_or_else(denied_request_permissions_response),
                    RequestPermissionOutcome::Cancelled | _ => {
                        denied_request_permissions_response()
                    }
                };

                self.thread
                    .submit(Op::RequestPermissionsResponse {
                        id: call_id,
                        response,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
            PendingPermissionRequest::McpElicitation {
                server_name,
                request_id,
                option_map,
            } => {
                let response = match response.outcome {
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                        option_id,
                        ..
                    }) => option_map
                        .get(option_id.0.as_ref())
                        .cloned()
                        .unwrap_or_else(ResolvedMcpElicitation::cancel),
                    RequestPermissionOutcome::Cancelled | _ => ResolvedMcpElicitation::cancel(),
                };

                self.thread
                    .submit(Op::ResolveElicitation {
                        server_name,
                        request_id,
                        decision: response.action,
                        content: response.content,
                        meta: response.meta,
                    })
                    .await
                    .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
            }
        }

        Ok(())
    }

    #[expect(clippy::too_many_lines)]
    async fn handle_event(&mut self, client: &SessionClient, event: EventMsg) {
        self.event_count += 1;

        // Complete any previous web search before starting a new one
        match &event {
            EventMsg::Error(..)
            | EventMsg::StreamError(..)
            | EventMsg::WebSearchBegin(..)
            | EventMsg::UserMessage(..)
            | EventMsg::ExecApprovalRequest(..)
            | EventMsg::ExecCommandBegin(..)
            | EventMsg::ExecCommandOutputDelta(..)
            | EventMsg::ExecCommandEnd(..)
            | EventMsg::McpToolCallBegin(..)
            | EventMsg::McpToolCallEnd(..)
            | EventMsg::ApplyPatchApprovalRequest(..)
            | EventMsg::PatchApplyBegin(..)
            | EventMsg::PatchApplyEnd(..)
            | EventMsg::TurnStarted(..)
            | EventMsg::TurnComplete(..)
            | EventMsg::TurnDiff(..)
            | EventMsg::TurnAborted(..)
            | EventMsg::EnteredReviewMode(..)
            | EventMsg::ExitedReviewMode(..)
            | EventMsg::ShutdownComplete => {
                self.complete_web_search(client).await;
            }
            _ => {}
        }

        match event {
            EventMsg::TurnStarted(TurnStartedEvent {
                model_context_window,
                collaboration_mode_kind,
                turn_id,
                ..
            }) => {
                self.transient_status_message_id = None;
                self.diagnostics.note_turn_started(&turn_id);
                info!("Task started with context window of {turn_id} {model_context_window:?} {collaboration_mode_kind:?}");
            }
            EventMsg::TokenCount(TokenCountEvent { info, .. }) => {
                if let Some(info) = info
                    && let Some(size) = info.model_context_window {
                        let used = info.last_token_usage.tokens_in_context_window().max(0) as u64;
                        client
                            .send_notification(SessionUpdate::UsageUpdate(UsageUpdate::new(
                                used,
                                size as u64,
                            )))
                            .await;
                    }
            }
            EventMsg::ItemStarted(ItemStartedEvent { thread_id, turn_id, item }) => {
                self.diagnostics.note_event("item_started");
                info!("Item started with thread_id: {thread_id}, turn_id: {turn_id}, item: {item:?}");
            }
            EventMsg::UserMessage(UserMessageEvent {
                message,
                images: _,
                text_elements: _,
                local_images: _,
            }) => {
                self.diagnostics.note_event("user_message");
                info!("User message: {message:?}");
            }
            EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
            }) => {
                self.diagnostics
                    .note_agent_output("agent_message_delta", &delta);
                info!("Agent message content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, delta: {delta:?}");
                self.seen_message_deltas = true;
                let message_id = self.agent_message_id_for_item(&item_id);
                client.send_agent_text_with_message_id(delta, message_id).await;
            }
            EventMsg::PlanDelta(event) => {
                self.diagnostics.note_agent_output("plan_delta", &event.delta);
                info!(
                    "Plan delta received: thread_id: {}, turn_id: {}, item_id: {}, delta: {:?}",
                    event.thread_id, event.turn_id, event.item_id, event.delta
                );
                let message_id = self.agent_message_id_for_item(&event.item_id);
                client
                    .send_proposed_plan_delta(event.delta, message_id, event.item_id)
                    .await;
            }
            EventMsg::ReasoningContentDelta(ReasoningContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
                summary_index: index,
            })
            | EventMsg::ReasoningRawContentDelta(ReasoningRawContentDeltaEvent {
                thread_id,
                turn_id,
                item_id,
                delta,
                content_index: index,
            }) => {
                self.diagnostics.note_event("reasoning_delta");
                info!("Agent reasoning content delta received: thread_id: {thread_id}, turn_id: {turn_id}, item_id: {item_id}, index: {index}, delta: {delta:?}");
                self.seen_reasoning_deltas = true;
                client.send_agent_thought(delta).await;
            }
            EventMsg::AgentReasoningSectionBreak(AgentReasoningSectionBreakEvent {
                item_id,
                summary_index,
            }) => {
                self.diagnostics.note_event("reasoning_section_break");
                info!("Agent reasoning section break received:  item_id: {item_id}, index: {summary_index}");
                // Make sure the section heading actually get spacing
                self.seen_reasoning_deltas = true;
                client.send_agent_thought("\n\n").await;
            }
            EventMsg::AgentMessage(AgentMessageEvent { message , phase: _, memory_citation: _ }) => {
                self.diagnostics
                    .note_agent_output("agent_message", &message);
                info!("Agent message (non-delta) received: {message:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_message_deltas) {
                    client.send_agent_text(message).await;
                }
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                self.diagnostics.note_event("agent_reasoning");
                info!("Agent reasoning (non-delta) received: {text:?}");
                // We didn't receive this message via streaming
                if !std::mem::take(&mut self.seen_reasoning_deltas) {
                    client.send_agent_thought(text).await;
                }
            }
            EventMsg::ThreadNameUpdated(event) => {
                self.diagnostics.note_event("thread_name_updated");
                info!("Thread name updated: {:?}", event.thread_name);
                if let Some(title) = event.thread_name {
                    client
                        .send_notification(SessionUpdate::SessionInfoUpdate(
                            SessionInfoUpdate::new().title(title),
                        ))
                        .await;
                }
            }
            EventMsg::PlanUpdate(UpdatePlanArgs { explanation, plan }) => {
                self.diagnostics.note_plan_update(&plan);
                // Send this to the client via session/update notification
                info!("Agent plan updated. Explanation: {:?}", explanation);
                client.update_plan(plan).await;
            }
            EventMsg::WebSearchBegin(WebSearchBeginEvent { call_id }) => {
                self.diagnostics.note_event("web_search_begin");
                info!("Web search started: call_id={}", call_id);
                // Create a ToolCall notification for the search beginning
                self.start_web_search(client, call_id).await;
            }
            EventMsg::WebSearchEnd(WebSearchEndEvent {
                call_id,
                query,
                action,
            }) => {
                self.diagnostics.note_event("web_search_query");
                info!("Web search query received: call_id={call_id}, query={query}");
                // Send update that the search is in progress with the query
                // (WebSearchEnd just means we have the query, not that results are ready)
                self.update_web_search_query(client, call_id, query, action)
                    .await;
                // The actual search results will come through AgentMessage events
                // We mark as completed when a new tool call begins
            }
            EventMsg::ExecApprovalRequest(event) => {
                self.diagnostics.note_event("exec_approval_request");
                info!(
                    "Command execution started: call_id={}, command={:?}",
                    event.call_id, event.command
                );
                if let Err(err) = self.exec_approval(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::ExecCommandBegin(event) => {
                self.diagnostics.note_event("exec_command_begin");
                info!(
                    "Command execution started: call_id={}, command={:?}",
                    event.call_id, event.command
                );
                self.exec_command_begin(client, event).await;
            }
            EventMsg::ExecCommandOutputDelta(delta_event) => {
                self.diagnostics.note_event("exec_command_output_delta");
                self.exec_command_output_delta(client, delta_event).await;
            }
            EventMsg::ExecCommandEnd(end_event) => {
                self.diagnostics.note_event("exec_command_end");
                info!(
                    "Command execution ended: call_id={}, exit_code={}",
                    end_event.call_id, end_event.exit_code
                );
                self.exec_command_end(client, end_event).await;
            }
            EventMsg::TerminalInteraction(event) => {
                self.diagnostics.note_event("terminal_interaction");
                info!(
                    "Terminal interaction: call_id={}, process_id={}, stdin={}",
                    event.call_id, event.process_id, event.stdin
                );
                self.terminal_interaction(client, event).await;
            }
            EventMsg::DynamicToolCallRequest(DynamicToolCallRequest { call_id, turn_id, tool, arguments, .. }) => {
                self.diagnostics.note_event("dynamic_tool_call_begin");
                info!("Dynamic tool call request: call_id={call_id}, turn_id={turn_id}, tool={tool}");
                self.start_dynamic_tool_call(client, call_id, tool, arguments).await;
            }
            EventMsg::DynamicToolCallResponse(event) => {
                self.diagnostics.note_event("dynamic_tool_call_end");
                info!(
                    "Dynamic tool call response: call_id={}, turn_id={}, tool={}",
                    event.call_id, event.turn_id, event.tool
                );
                self.end_dynamic_tool_call(client, event).await;
            }
            EventMsg::McpToolCallBegin(McpToolCallBeginEvent {
                call_id,
                invocation,
                ..
            }) => {
                self.diagnostics.note_event("mcp_tool_call_begin");
                info!(
                    "MCP tool call begin: call_id={call_id}, invocation={} {}",
                    invocation.server, invocation.tool
                );
                self.start_mcp_tool_call(client, call_id, invocation).await;
            }
            EventMsg::McpToolCallEnd(McpToolCallEndEvent {
                call_id,
                invocation,
                duration,
                result,
                ..
            }) => {
                self.diagnostics.note_event("mcp_tool_call_end");
                info!(
                    "MCP tool call ended: call_id={call_id}, invocation={} {}, duration={duration:?}",
                    invocation.server, invocation.tool
                );
                self.end_mcp_tool_call(client, call_id, result).await;
            }
            EventMsg::ApplyPatchApprovalRequest(event) => {
                self.diagnostics.note_event("patch_approval_request");
                info!(
                    "Apply patch approval request: call_id={}, reason={:?}",
                    event.call_id, event.reason
                );
                if let Err(err) = self.patch_approval(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::PatchApplyBegin(event) => {
                self.diagnostics.note_event("patch_apply_begin");
                info!(
                    "Patch apply begin: call_id={}, auto_approved={}",
                    event.call_id, event.auto_approved
                );
                self.start_patch_apply(client, event).await;
            }
            EventMsg::PatchApplyEnd(event) => {
                self.diagnostics.note_event("patch_apply_end");
                info!(
                    "Patch apply end: call_id={}, success={}",
                    event.call_id, event.success
                );
                self.end_patch_apply(client, event).await;
            }
            EventMsg::ItemCompleted(ItemCompletedEvent {
                thread_id,
                turn_id,
                item,
            }) => {
                self.diagnostics.note_event("item_completed");
                info!("Item completed: thread_id={}, turn_id={}, item={:?}", thread_id, turn_id, item);
                match item {
                    TurnItem::AgentMessage(agent_message) => {
                        if let Some(message_id) = self
                            .agent_message_ids_by_item_id
                            .remove(agent_message.id.as_str())
                        {
                            client
                                .send_agent_message_completed(message_id, agent_message.id)
                                .await;
                        }
                    }
                    TurnItem::Plan(plan_item) => {
                        self.agent_message_ids_by_item_id
                            .remove(plan_item.id.as_str());
                        client
                            .send_proposed_plan_completed(plan_item.text, plan_item.id)
                            .await;
                    }
                    _ => {}
                }
            }
            EventMsg::TurnComplete(TurnCompleteEvent { last_agent_message, turn_id, .. }) => {
                self.transient_status_message_id = None;
                self.diagnostics.note_turn_complete(&turn_id);
                info!(
                    "Task {turn_id} completed successfully after {} events. Last agent message: {last_agent_message:?}",
                    self.event_count
                );
                info!(
                    submission_id = %self.submission_id,
                    turn_id = turn_id,
                    event_count = self.event_count,
                    last_event_kind = ?self.diagnostics.last_event_kind,
                    last_agent_preview = self.diagnostics.last_agent_preview.as_deref().unwrap_or(""),
                    pending_permission_count = self.pending_permission_interactions.len(),
                    active_command_count = self.active_commands.len(),
                    active_web_search = self.active_web_search.is_some(),
                    active_guardian_assessment_count = self.active_guardian_assessments.len(),
                    "codex_acp.prompt.turn_complete_seen"
                );
                self.abort_pending_interactions();
                if let Some(response_tx) = self.response_tx.take() {
                    let sent = response_tx.send(Ok(StopReason::EndTurn)).is_ok();
                    self.diagnostics.note_response_sent();
                    info!(
                        submission_id = %self.submission_id,
                        turn_id = turn_id,
                        sent = sent,
                        "codex_acp.prompt.response_sent"
                    );
                }
            }
            EventMsg::UndoStarted(event) => {
                client
                    .send_agent_text(
                        event
                            .message
                            .unwrap_or_else(|| "Undo in progress...".to_string()),
                    )
                    .await;
            }
            EventMsg::UndoCompleted(event) => {
                let fallback = if event.success {
                    "Undo completed.".to_string()
                } else {
                    "Undo failed.".to_string()
                };
                client.send_agent_text(event.message.unwrap_or(fallback)).await;
            }
            EventMsg::StreamError(StreamErrorEvent {
                message,
                codex_error_info,
                additional_details,
            }) => {
                self.diagnostics.note_stream_error();
                error!(
                    "Handled error during turn: {message} {codex_error_info:?} {additional_details:?}"
                );
            }
            EventMsg::Error(ErrorEvent {
                message,
                codex_error_info,
            }) => {
                self.diagnostics.note_event("error");
                error!("Unhandled error during turn: {message} {codex_error_info:?}");
                self.abort_pending_interactions();
                if let Some(response_tx) = self.response_tx.take() {
                    let sent = response_tx
                        .send(Err(Error::internal_error().data(
                            json!({ "message": message, "codex_error_info": codex_error_info }),
                        )))
                        .is_ok();
                    self.diagnostics.note_response_sent();
                    info!(
                        submission_id = %self.submission_id,
                        turn_id = ?self.diagnostics.current_turn_id,
                        sent = sent,
                        "codex_acp.prompt.error_response_sent"
                    );
                }
            }
            EventMsg::TurnAborted(TurnAbortedEvent { reason, turn_id, .. }) => {
                self.transient_status_message_id = None;
                self.diagnostics.note_event("turn_aborted");
                info!("Turn {turn_id:?} aborted: {reason:?}");
                self.abort_pending_interactions();
                if let Some(response_tx) = self.response_tx.take() {
                    let sent = response_tx.send(Ok(StopReason::Cancelled)).is_ok();
                    self.diagnostics.note_response_sent();
                    info!(
                        submission_id = %self.submission_id,
                        turn_id = ?turn_id,
                        sent = sent,
                        "codex_acp.prompt.cancel_response_sent"
                    );
                }
            }
            EventMsg::ShutdownComplete => {
                self.diagnostics.note_event("shutdown_complete");
                info!("Agent shutting down");
                self.abort_pending_interactions();
                if let Some(response_tx) = self.response_tx.take() {
                    let sent = response_tx.send(Ok(StopReason::Cancelled)).is_ok();
                    self.diagnostics.note_response_sent();
                    info!(
                        submission_id = %self.submission_id,
                        turn_id = ?self.diagnostics.current_turn_id,
                        sent = sent,
                        "codex_acp.prompt.shutdown_response_sent"
                    );
                }
            }
            EventMsg::ViewImageToolCall(ViewImageToolCallEvent { call_id, path }) => {
                self.diagnostics.note_event("view_image_tool_call");
                info!("ViewImageToolCallEvent received");
                let display_path = path.display().to_string();
                client
                    .send_notification(
                        SessionUpdate::ToolCall(
                            ToolCall::new(call_id, format!("View Image {display_path}"))
                                .kind(ToolKind::Read).status(ToolCallStatus::Completed)
                                .content(vec![ToolCallContent::Content(Content::new(ContentBlock::ResourceLink(ResourceLink::new(display_path.clone(), display_path.clone())
                            )
                        )
                    )]).locations(vec![ToolCallLocation::new(path)])))
                    .await;
            }
            EventMsg::EnteredReviewMode(review_request) => {
                self.diagnostics.note_event("review_mode_entered");
                info!("Review begin: request={review_request:?}");
            }
            EventMsg::ExitedReviewMode(event) => {
                self.diagnostics.note_event("review_mode_exited");
                info!("Review end: output={event:?}");
                if let Err(err) = self.review_mode_exit(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::Warning(WarningEvent { message }) => {
                self.diagnostics.note_event("warning");
                warn!("Warning: {message}");
                // Forward warnings to the client as agent messages so users see
                // informational notices (e.g., the post-compact advisory message).
                client.send_agent_text(message).await;
            }
            EventMsg::McpStartupUpdate(McpStartupUpdateEvent { server, status }) => {
                self.diagnostics.note_event("mcp_startup_update");
                info!("MCP startup update: server={server}, status={status:?}");
            }
            EventMsg::McpStartupComplete(McpStartupCompleteEvent {
                ready,
                failed,
                cancelled,
            }) => {
                self.diagnostics.note_event("mcp_startup_complete");
                info!(
                    "MCP startup complete: ready={ready:?}, failed={failed:?}, cancelled={cancelled:?}"
                );
            }
            EventMsg::ElicitationRequest(event) => {
                self.diagnostics.note_event("elicitation_request");
                info!("Elicitation request: server={}, id={:?}", event.server_name, event.id);
                if let Err(err) = self.mcp_elicitation(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::ModelReroute(ModelRerouteEvent { from_model, to_model, reason }) => {
                self.diagnostics.note_event("model_reroute");
                info!("Model reroute: from={from_model}, to={to_model}, reason={reason:?}");
            }

            EventMsg::ContextCompacted(..) => {
                self.diagnostics.note_event("context_compacted");
                info!("Context compacted");
                client.send_agent_text("Context compacted\n".to_string()).await;
            }
            EventMsg::RequestPermissions(event) => {
                self.diagnostics.note_event("request_permissions");
                info!("Request permissions: {} {}", event.call_id, event.turn_id);
                if let Err(err) = self.request_permissions(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::RequestUserInput(event) => {
                self.diagnostics.note_event("request_user_input");
                if let Err(err) = self.request_user_input(client, event).await
                    && let Some(response_tx) = self.response_tx.take()
                {
                    drop(response_tx.send(Err(err)));
                }
            }
            EventMsg::GuardianAssessment(event) => {
                info!(
                    "Guardian assessment: id={}, status={:?}, turn_id={}",
                    event.id, event.status, event.turn_id
                );
                self.guardian_assessment(client, event).await;
            }
            EventMsg::BackgroundEvent(event) => {
                self.diagnostics.note_event("background_event");
                let message_id = self.transient_status_message_id();
                client.send_transient_status(event.message, message_id).await;
            }
            EventMsg::HookStarted(event) => {
                self.diagnostics.note_event("hook_started");
                self.hook_started(client, event).await;
            }
            EventMsg::HookCompleted(event) => {
                self.diagnostics.note_event("hook_completed");
                self.hook_completed(client, event).await;
            }
            EventMsg::DeprecationNotice(event) => {
                self.diagnostics.note_event("deprecation_notice");
                let mut message = format!("Warning: {}", event.summary.trim());
                if let Some(details) = event.details.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
                    message.push_str("\n\n");
                    message.push_str(details);
                }
                client.send_agent_text(message).await;
            }

            // Ignore these events
            EventMsg::ImageGenerationBegin(..)
            | EventMsg::ImageGenerationEnd(..)
            | EventMsg::AgentReasoningRawContent(..)
            | EventMsg::GuardianWarning(..)
            | EventMsg::ThreadRolledBack(..)
            | EventMsg::PatchApplyUpdated(..)
            // we already have a way to diff the turn, so ignore
            | EventMsg::TurnDiff(..)
            | EventMsg::SkillsUpdateAvailable
            // Old events
            | EventMsg::AgentMessageDelta(..)
            | EventMsg::AgentReasoningDelta(..)
            | EventMsg::AgentReasoningRawContentDelta(..)
            | EventMsg::RawResponseItem(..)
            | EventMsg::SessionConfigured(..)
            // TODO: Subagent UI?
            | EventMsg::CollabAgentSpawnBegin(..)
            | EventMsg::CollabAgentSpawnEnd(..)
            | EventMsg::CollabAgentInteractionBegin(..)
            | EventMsg::CollabAgentInteractionEnd(..)
            | EventMsg::RealtimeConversationStarted(..)
            | EventMsg::RealtimeConversationRealtime(..)
            | EventMsg::RealtimeConversationSdp(..)
            | EventMsg::RealtimeConversationClosed(..)
            | EventMsg::ModelVerification(..)
            | EventMsg::CollabWaitingBegin(..)
            | EventMsg::CollabWaitingEnd(..)
            | EventMsg::CollabResumeBegin(..)
            | EventMsg::CollabResumeEnd(..)
            | EventMsg::CollabCloseBegin(..)
            | EventMsg::CollabCloseEnd(..)=> {}
            e @ (EventMsg::McpListToolsResponse(..)
            | EventMsg::ListSkillsResponse(..)
            | EventMsg::RealtimeConversationListVoicesResponse(..)
            // Used for returning a single history entry
            | EventMsg::GetHistoryEntryResponse(..)
            ) => {
                warn!("Unexpected event: {:?}", e);
            }
        }
    }

    async fn hook_started(&self, client: &SessionClient, event: HookStartedEvent) {
        let call_id = hook_tool_call_id(&event.run.id);
        let content = hook_tool_content(&event.run);
        let mut tool_call = ToolCall::new(call_id, hook_title(&event.run))
            .kind(ToolKind::Other)
            .status(ToolCallStatus::InProgress)
            .meta(hook_meta(&event.run));
        if !content.is_empty() {
            tool_call = tool_call.content(content);
        }
        client.send_tool_call(tool_call).await;
    }

    async fn hook_completed(&self, client: &SessionClient, event: HookCompletedEvent) {
        let status = match event.run.status {
            HookRunStatus::Completed => ToolCallStatus::Completed,
            HookRunStatus::Running => ToolCallStatus::InProgress,
            HookRunStatus::Blocked | HookRunStatus::Failed | HookRunStatus::Stopped => {
                ToolCallStatus::Failed
            }
        };
        let content = hook_tool_content(&event.run);
        let mut fields = ToolCallUpdateFields::new()
            .title(hook_title(&event.run))
            .kind(ToolKind::Other)
            .status(status);
        if !content.is_empty() {
            fields = fields.content(content);
        }
        client
            .send_tool_call_update(
                ToolCallUpdate::new(hook_tool_call_id(&event.run.id), fields)
                    .meta(hook_meta(&event.run)),
            )
            .await;
    }

    async fn request_user_input(
        &mut self,
        client: &SessionClient,
        event: RequestUserInputEvent,
    ) -> Result<(), Error> {
        let turn_id = if event.turn_id.is_empty() {
            self.submission_id.clone()
        } else {
            event.turn_id.clone()
        };
        let question_count = event.questions.len();

        if !client.supports_request_user_input() {
            warn!(
                call_id = %event.call_id,
                turn_id = %turn_id,
                question_count,
                "request_user_input extension is not supported by client; submitting empty answer"
            );
            return self.submit_empty_user_input_answer(turn_id).await;
        }

        let params = Self::codex_request_user_input_params(&event, &turn_id);
        let response = match client.request_user_input(params).await {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    call_id = %event.call_id,
                    turn_id = %turn_id,
                    question_count,
                    error = ?error,
                    "request_user_input extension failed before response delivery; submitting empty answer"
                );
                return self.submit_empty_user_input_answer(turn_id).await;
            }
        };

        let response = Self::codex_request_user_input_response(response);
        self.thread
            .submit(Op::UserInputAnswer {
                id: turn_id,
                response,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        Ok(())
    }

    async fn submit_empty_user_input_answer(&mut self, turn_id: String) -> Result<(), Error> {
        self.thread
            .submit(Op::UserInputAnswer {
                id: turn_id,
                response: RequestUserInputResponse {
                    answers: HashMap::new(),
                },
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        Ok(())
    }

    fn codex_request_user_input_params(
        event: &RequestUserInputEvent,
        turn_id: &str,
    ) -> CodexRequestUserInputExtParams {
        CodexRequestUserInputExtParams {
            call_id: event.call_id.clone(),
            turn_id: turn_id.to_string(),
            questions: event
                .questions
                .iter()
                .map(Self::codex_request_user_input_question)
                .collect(),
        }
    }

    fn codex_request_user_input_question(
        question: &RequestUserInputQuestion,
    ) -> CodexRequestUserInputExtQuestion {
        CodexRequestUserInputExtQuestion {
            question_id: question.id.clone(),
            header: question.header.clone(),
            question: question.question.clone(),
            is_other: question.is_other,
            is_secret: question.is_secret,
            options: question
                .options
                .as_ref()
                .map(|options| {
                    options
                        .iter()
                        .map(Self::codex_request_user_input_option)
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    fn codex_request_user_input_option(
        option: &RequestUserInputQuestionOption,
    ) -> CodexRequestUserInputExtOption {
        CodexRequestUserInputExtOption {
            label: option.label.clone(),
            description: option.description.clone(),
        }
    }

    fn codex_request_user_input_response(
        response: CodexRequestUserInputExtResponse,
    ) -> RequestUserInputResponse {
        if response.outcome != CodexRequestUserInputExtOutcome::Submitted {
            return RequestUserInputResponse {
                answers: HashMap::new(),
            };
        }

        let answers = response
            .answers
            .into_iter()
            .map(|answer| {
                let mut entries = Vec::new();
                if let Some(label) = answer.selected_option_label {
                    entries.push(label);
                }
                if let Some(text) = answer.text {
                    let text = text.trim();
                    if !text.is_empty() {
                        entries.push(format!("user_note: {text}"));
                    }
                }
                (
                    answer.question_id,
                    RequestUserInputAnswer { answers: entries },
                )
            })
            .collect();

        RequestUserInputResponse { answers }
    }

    async fn mcp_elicitation(
        &mut self,
        client: &SessionClient,
        event: ElicitationRequestEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let ElicitationRequestEvent {
            server_name,
            id,
            request,
            turn_id: _,
        } = event;
        if let Some(supported_request) = build_supported_mcp_elicitation_permission_request(
            &server_name,
            &id,
            &request,
            raw_input,
        ) {
            info!(
                "Routing MCP tool approval elicitation through ACP permission request: server={}, id={:?}",
                server_name, id
            );
            self.spawn_permission_request(
                client,
                supported_request.request_key,
                PendingPermissionRequest::McpElicitation {
                    server_name,
                    request_id: id,
                    option_map: supported_request.option_map,
                },
                supported_request.tool_call,
                supported_request.options,
            );
            return Ok(());
        }

        let request_kind = match &request {
            ElicitationRequest::Form { .. } => "form",
            ElicitationRequest::Url { .. } => "url",
        };

        if client.supports_mcp_elicitation() {
            let response = match client
                .mcp_elicitation(CodexMcpElicitationExtParams {
                    server_name: server_name.clone(),
                    request: request.clone(),
                })
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    warn!(
                        "mcp_elicitation extension failed before response delivery; declining: {err:?}"
                    );
                    CodexMcpElicitationExtResponse {
                        outcome: CodexMcpElicitationExtOutcome::Declined,
                        content: None,
                        meta: None,
                    }
                }
            };
            let (decision, content, meta) = Self::codex_mcp_elicitation_response(response);
            self.thread
                .submit(Op::ResolveElicitation {
                    server_name,
                    request_id: id,
                    decision,
                    content,
                    meta,
                })
                .await
                .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

            return Ok(());
        }

        info!(
            "Auto-declining unsupported MCP elicitation: server={}, id={:?}, kind={request_kind}",
            server_name, id
        );

        self.thread
            .submit(Op::ResolveElicitation {
                server_name,
                request_id: id,
                decision: ElicitationAction::Decline,
                content: None,
                meta: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        Ok(())
    }

    fn codex_mcp_elicitation_response(
        response: CodexMcpElicitationExtResponse,
    ) -> (ElicitationAction, Option<Value>, Option<Value>) {
        match response.outcome {
            CodexMcpElicitationExtOutcome::Accepted => {
                (ElicitationAction::Accept, response.content, response.meta)
            }
            CodexMcpElicitationExtOutcome::Declined => (ElicitationAction::Decline, None, None),
            CodexMcpElicitationExtOutcome::Cancelled => (ElicitationAction::Cancel, None, None),
        }
    }

    async fn review_mode_exit(
        &self,
        client: &SessionClient,
        event: ExitedReviewModeEvent,
    ) -> Result<(), Error> {
        let ExitedReviewModeEvent { review_output } = event;
        let Some(ReviewOutputEvent {
            findings,
            overall_correctness: _,
            overall_explanation,
            overall_confidence_score: _,
        }) = review_output
        else {
            return Ok(());
        };

        let text = if findings.is_empty() {
            let explanation = overall_explanation.trim();
            if explanation.is_empty() {
                "Reviewer failed to output a response"
            } else {
                explanation
            }
            .to_string()
        } else {
            format_review_findings_block(&findings, None)
        };

        client.send_agent_text(&text).await;
        Ok(())
    }

    async fn patch_approval(
        &mut self,
        client: &SessionClient,
        event: ApplyPatchApprovalRequestEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let ApplyPatchApprovalRequestEvent {
            call_id,
            changes,
            reason,
            // grant_root doesn't seem to be set anywhere on the codex side
            grant_root: _,
            turn_id: _,
        } = event;
        let (title, locations, content) = extract_tool_call_content_from_changes(changes);
        let request_key = patch_request_key(&call_id);
        let options = vec![
            PermissionOption::new("approved", "Yes", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                "abort",
                "No, provide feedback",
                PermissionOptionKind::RejectOnce,
            ),
        ];
        self.spawn_permission_request(
            client,
            request_key,
            PendingPermissionRequest::Patch {
                call_id: call_id.clone(),
                option_map: HashMap::from([
                    ("approved".to_string(), ReviewDecision::Approved),
                    ("abort".to_string(), ReviewDecision::Abort),
                ]),
            },
            ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::Pending)
                    .title(title)
                    .locations(locations)
                    .content(content.chain(reason.map(|r| r.into())).collect::<Vec<_>>())
                    .raw_input(raw_input),
            ),
            options,
        );
        Ok(())
    }

    async fn start_patch_apply(&self, client: &SessionClient, event: PatchApplyBeginEvent) {
        let raw_input = serde_json::json!(&event);
        let PatchApplyBeginEvent {
            call_id,
            auto_approved: _,
            changes,
            turn_id: _,
        } = event;

        let (title, locations, content) = extract_tool_call_content_from_changes(changes);

        client
            .send_tool_call(
                ToolCall::new(call_id, title)
                    .kind(ToolKind::Edit)
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .content(content.collect())
                    .raw_input(raw_input),
            )
            .await;
    }

    async fn end_patch_apply(&self, client: &SessionClient, event: PatchApplyEndEvent) {
        let raw_output = serde_json::json!(&event);
        let PatchApplyEndEvent {
            call_id,
            stdout: _,
            stderr: _,
            success,
            changes,
            turn_id: _,
            status,
        } = event;

        let (title, locations, content) = if !changes.is_empty() {
            let (title, locations, content) = extract_tool_call_content_from_changes(changes);
            (Some(title), Some(locations), Some(content.collect()))
        } else {
            (None, None, None)
        };

        let status = match status {
            PatchApplyStatus::Completed => ToolCallStatus::Completed,
            _ if success => ToolCallStatus::Completed,
            PatchApplyStatus::Failed | PatchApplyStatus::Declined => ToolCallStatus::Failed,
        };

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(status)
                    .raw_output(raw_output)
                    .title(title)
                    .locations(locations)
                    .content(content),
            ))
            .await;
    }

    async fn start_dynamic_tool_call(
        &self,
        client: &SessionClient,
        call_id: String,
        tool: String,
        arguments: serde_json::Value,
    ) {
        client
            .send_tool_call(
                ToolCall::new(call_id, format!("Tool: {tool}"))
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!(&arguments)),
            )
            .await;
    }

    async fn start_mcp_tool_call(
        &self,
        client: &SessionClient,
        call_id: String,
        invocation: McpInvocation,
    ) {
        let title = format!("Tool: {}/{}", invocation.server, invocation.tool);
        client
            .send_tool_call(
                ToolCall::new(call_id, title)
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!(&invocation)),
            )
            .await;
    }

    async fn end_dynamic_tool_call(
        &self,
        client: &SessionClient,
        event: DynamicToolCallResponseEvent,
    ) {
        let raw_output = serde_json::json!(event);
        let DynamicToolCallResponseEvent {
            call_id,
            turn_id: _,
            tool: _,
            arguments: _,
            content_items,
            success,
            error,
            duration: _,
            ..
        } = event;

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(if success {
                        ToolCallStatus::Completed
                    } else {
                        ToolCallStatus::Failed
                    })
                    .raw_output(raw_output)
                    .content(
                        content_items
                            .into_iter()
                            .map(|item| match item {
                                DynamicToolCallOutputContentItem::InputText { text } => {
                                    ToolCallContent::Content(Content::new(text))
                                }
                                DynamicToolCallOutputContentItem::InputImage { image_url } => {
                                    ToolCallContent::Content(Content::new(
                                        ContentBlock::ResourceLink(ResourceLink::new(
                                            image_url.clone(),
                                            image_url,
                                        )),
                                    ))
                                }
                            })
                            .chain(error.map(|e| ToolCallContent::Content(Content::new(e))))
                            .collect::<Vec<_>>(),
                    ),
            ))
            .await;
    }

    async fn end_mcp_tool_call(
        &self,
        client: &SessionClient,
        call_id: String,
        result: Result<CallToolResult, String>,
    ) {
        let is_error = match result.as_ref() {
            Ok(result) => result.is_error.unwrap_or_default(),
            Err(_) => true,
        };
        let raw_output = match result.as_ref() {
            Ok(result) => serde_json::json!(result),
            Err(err) => serde_json::json!(err),
        };

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(if is_error {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    })
                    .raw_output(raw_output)
                    .content(result.ok().filter(|result| !result.content.is_empty()).map(
                        |result| {
                            result
                                .content
                                .into_iter()
                                .filter_map(|content| {
                                    serde_json::from_value::<ContentBlock>(content).ok()
                                })
                                .map(|content| ToolCallContent::Content(Content::new(content)))
                                .collect()
                        },
                    )),
            ))
            .await;
    }

    async fn exec_approval(
        &mut self,
        client: &SessionClient,
        event: ExecApprovalRequestEvent,
    ) -> Result<(), Error> {
        let available_decisions = event.effective_available_decisions();
        let raw_input = serde_json::json!(&event);
        let ExecApprovalRequestEvent {
            call_id,
            command: _,
            turn_id,
            cwd,
            reason,
            parsed_cmd,
            proposed_execpolicy_amendment,
            approval_id,
            network_approval_context,
            additional_permissions,
            available_decisions: _,
            proposed_network_policy_amendments,
        } = event;

        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId::new(call_id.clone());
        let ParseCommandToolCall {
            title,
            terminal_output,
            file_extension,
            locations,
            kind,
        } = parse_command_tool_call(parsed_cmd, &cwd);
        self.active_commands.insert(
            call_id.clone(),
            ActiveCommand {
                terminal_output,
                tool_call_id: tool_call_id.clone(),
                output: String::new(),
                file_extension,
            },
        );

        let mut content = vec![];

        if let Some(reason) = reason {
            content.push(reason);
        }
        if let Some(amendment) = proposed_execpolicy_amendment.as_ref() {
            content.push(format!(
                "Proposed Amendment: {}",
                amendment.command().join("\n")
            ));
        }
        if let Some(policy) = network_approval_context.as_ref() {
            let NetworkApprovalContext { host, protocol } = policy;
            content.push(format!("Network Approval Context: {:?} {}", protocol, host));
        }
        if let Some(permissions) = additional_permissions.as_ref() {
            content.push(format!(
                "Additional Permissions: {}",
                serde_json::to_string_pretty(&permissions)?
            ));
        }
        content.push(format!(
            "Available Decisions: {}",
            available_decisions.iter().map(|d| d.to_string()).join("\n")
        ));
        if let Some(amendments) = proposed_network_policy_amendments.as_ref() {
            content.push(format!(
                "Proposed Network Policy Amendments: {}",
                amendments
                    .iter()
                    .map(|amendment| format!("{:?} {:?}", amendment.action, amendment.host))
                    .join("\n")
            ));
        }

        let content = if content.is_empty() {
            None
        } else {
            Some(vec![content.join("\n").into()])
        };
        let permission_options = build_exec_permission_options(
            &available_decisions,
            network_approval_context.as_ref(),
            additional_permissions.as_ref(),
        );

        self.spawn_permission_request(
            client,
            exec_request_key(&call_id),
            PendingPermissionRequest::Exec {
                approval_id: approval_id.unwrap_or(call_id.clone()),
                turn_id,
                option_map: permission_options
                    .iter()
                    .map(|option| (option.option_id.to_string(), option.decision.clone()))
                    .collect(),
            },
            ToolCallUpdate::new(
                tool_call_id,
                ToolCallUpdateFields::new()
                    .kind(kind)
                    .status(ToolCallStatus::Pending)
                    .title(title)
                    .raw_input(raw_input)
                    .content(content)
                    .locations(if locations.is_empty() {
                        None
                    } else {
                        Some(locations)
                    }),
            ),
            permission_options
                .into_iter()
                .map(|option| option.permission_option)
                .collect(),
        );

        Ok(())
    }

    async fn exec_command_begin(&mut self, client: &SessionClient, event: ExecCommandBeginEvent) {
        let raw_input = serde_json::json!(&event);
        let ExecCommandBeginEvent {
            turn_id: _,
            source: _,
            interaction_input: _,
            call_id,
            command: _,
            cwd,
            parsed_cmd,
            process_id: _,
        } = event;
        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId::new(call_id.clone());
        let ParseCommandToolCall {
            title,
            file_extension,
            locations,
            terminal_output,
            kind,
        } = parse_command_tool_call(parsed_cmd, &cwd);

        let active_command = ActiveCommand {
            tool_call_id: tool_call_id.clone(),
            output: String::new(),
            file_extension,
            terminal_output,
        };
        let (content, meta) = if client.supports_terminal_output(&active_command) {
            let content = vec![ToolCallContent::Terminal(Terminal::new(call_id.clone()))];
            let meta = Some(Meta::from_iter([(
                "terminal_info".to_owned(),
                serde_json::json!({
                    "terminal_id": call_id,
                    "cwd": cwd
                }),
            )]));
            (content, meta)
        } else {
            (vec![], None)
        };

        self.active_commands.insert(call_id.clone(), active_command);

        client
            .send_tool_call(
                ToolCall::new(tool_call_id, title)
                    .kind(kind)
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(raw_input)
                    .content(content)
                    .meta(meta),
            )
            .await;
    }

    async fn exec_command_output_delta(
        &mut self,
        client: &SessionClient,
        event: ExecCommandOutputDeltaEvent,
    ) {
        let ExecCommandOutputDeltaEvent {
            call_id,
            chunk,
            stream: _,
        } = event;
        // Stream output bytes to the display-only terminal via ToolCallUpdate meta.
        if let Some(active_command) = self.active_commands.get_mut(&call_id) {
            let data_str = String::from_utf8_lossy(&chunk).to_string();

            let update = if client.supports_terminal_output(active_command) {
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new(),
                )
                .meta(Meta::from_iter([(
                    "terminal_output".to_owned(),
                    serde_json::json!({
                        "terminal_id": call_id,
                        "data": data_str
                    }),
                )]))
            } else {
                active_command.output.push_str(&data_str);
                let content = match active_command.file_extension.as_deref() {
                    Some("md") => active_command.output.clone(),
                    Some(ext) => format!(
                        "```{ext}\n{}\n```\n",
                        active_command.output.trim_end_matches('\n')
                    ),
                    None => format!(
                        "```sh\n{}\n```\n",
                        active_command.output.trim_end_matches('\n')
                    ),
                };
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new().content(vec![content.into()]),
                )
            };

            client.send_tool_call_update(update).await;
        }
    }

    async fn exec_command_end(&mut self, client: &SessionClient, event: ExecCommandEndEvent) {
        let raw_output = serde_json::json!(&event);
        let ExecCommandEndEvent {
            turn_id: _,
            command: _,
            cwd: _,
            parsed_cmd: _,
            source: _,
            interaction_input: _,
            call_id,
            exit_code,
            stdout: _,
            stderr: _,
            aggregated_output: _,
            duration: _,
            formatted_output: _,
            process_id: _,
            status,
        } = event;
        if let Some(active_command) = self.active_commands.remove(&call_id) {
            let is_success = exit_code == 0;

            let status = match status {
                ExecCommandStatus::Completed => ToolCallStatus::Completed,
                _ if is_success => ToolCallStatus::Completed,
                ExecCommandStatus::Failed | ExecCommandStatus::Declined => ToolCallStatus::Failed,
            };

            client
                .send_tool_call_update(
                    ToolCallUpdate::new(
                        active_command.tool_call_id.clone(),
                        ToolCallUpdateFields::new()
                            .status(status)
                            .raw_output(raw_output),
                    )
                    .meta(
                        client.supports_terminal_output(&active_command).then(|| {
                            Meta::from_iter([(
                                "terminal_exit".into(),
                                serde_json::json!({
                                    "terminal_id": call_id,
                                    "exit_code": exit_code,
                                    "signal": null
                                }),
                            )])
                        }),
                    ),
                )
                .await;
        }
    }

    async fn terminal_interaction(
        &mut self,
        client: &SessionClient,
        event: TerminalInteractionEvent,
    ) {
        let TerminalInteractionEvent {
            call_id,
            process_id: _,
            stdin,
        } = event;

        let stdin = format!("\n{stdin}\n");
        // Stream output bytes to the display-only terminal via ToolCallUpdate meta.
        if let Some(active_command) = self.active_commands.get_mut(&call_id) {
            let update = if client.supports_terminal_output(active_command) {
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new(),
                )
                .meta(Meta::from_iter([(
                    "terminal_output".to_owned(),
                    serde_json::json!({
                        "terminal_id": call_id,
                        "data": stdin
                    }),
                )]))
            } else {
                active_command.output.push_str(&stdin);
                let content = match active_command.file_extension.as_deref() {
                    Some("md") => active_command.output.clone(),
                    Some(ext) => format!(
                        "```{ext}\n{}\n```\n",
                        active_command.output.trim_end_matches('\n')
                    ),
                    None => format!(
                        "```sh\n{}\n```\n",
                        active_command.output.trim_end_matches('\n')
                    ),
                };
                ToolCallUpdate::new(
                    active_command.tool_call_id.clone(),
                    ToolCallUpdateFields::new().content(vec![content.into()]),
                )
            };

            client.send_tool_call_update(update).await;
        }
    }

    async fn start_web_search(&mut self, client: &SessionClient, call_id: String) {
        self.active_web_search = Some(call_id.clone());
        client
            .send_tool_call(ToolCall::new(call_id, "Searching the Web").kind(ToolKind::Fetch))
            .await;
    }

    async fn update_web_search_query(
        &self,
        client: &SessionClient,
        call_id: String,
        query: String,
        action: WebSearchAction,
    ) {
        let title = match &action {
            WebSearchAction::Search { query, queries } => queries
                .as_ref()
                .map(|q| format!("Searching for: {}", q.join(", ")))
                .or_else(|| query.as_ref().map(|q| format!("Searching for: {q}")))
                .unwrap_or_else(|| "Web search".to_string()),
            WebSearchAction::OpenPage { url } => url
                .as_ref()
                .map(|u| format!("Opening: {u}"))
                .unwrap_or_else(|| "Open page".to_string()),
            WebSearchAction::FindInPage { pattern, url } => match (pattern, url) {
                (Some(p), Some(u)) => format!("Finding: {p} in {u}"),
                (Some(p), None) => format!("Finding: {p}"),
                (None, Some(u)) => format!("Find in page: {u}"),
                (None, None) => "Find in page".to_string(),
            },
            WebSearchAction::Other => "Web search".to_string(),
        };

        client
            .send_tool_call_update(ToolCallUpdate::new(
                call_id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::InProgress)
                    .title(title)
                    .raw_input(serde_json::json!({
                        "query": query,
                        "action": action
                    })),
            ))
            .await;
    }

    async fn complete_web_search(&mut self, client: &SessionClient) {
        if let Some(call_id) = self.active_web_search.take() {
            client
                .send_tool_call_update(ToolCallUpdate::new(
                    call_id,
                    ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
                ))
                .await;
        }
    }

    async fn request_permissions(
        &mut self,
        client: &SessionClient,
        event: RequestPermissionsEvent,
    ) -> Result<(), Error> {
        let raw_input = serde_json::json!(&event);
        let RequestPermissionsEvent {
            call_id,
            turn_id: _,
            reason,
            permissions,
            ..
        } = event;

        // Create a new tool call for the command execution
        let tool_call_id = ToolCallId::new(call_id.clone());

        let mut content = vec![];

        if let Some(reason) = reason.as_ref() {
            content.push(reason.clone());
        }
        if let Some(file_system) = permissions.file_system.as_ref() {
            if let Some((read, write)) = file_system.legacy_read_write_roots() {
                if let Some(read) = read.as_ref() {
                    content.push(format!(
                        "File System Read Access: {}",
                        read.iter().map(|p| p.display()).join(", ")
                    ));
                }
                if let Some(write) = write.as_ref() {
                    content.push(format!(
                        "File System Write Access: {}",
                        write.iter().map(|p| p.display()).join(", ")
                    ));
                }
            } else if !file_system.entries.is_empty() {
                content.push(format!(
                    "File System Access: {}",
                    file_system
                        .entries
                        .iter()
                        .map(|entry| format!("{:?} {:?}", entry.access, entry.path))
                        .join(", ")
                ));
            }
        }
        if let Some(network) = permissions.network.as_ref()
            && let Some(enabled) = network.enabled
        {
            content.push(format!("Network Access: {enabled}"));
        }

        let content = if content.is_empty() {
            None
        } else {
            Some(vec![content.join("\n").into()])
        };

        let permissions: RequestPermissionProfile = permissions;
        let option_map = request_permissions_option_map(&permissions);

        self.spawn_permission_request(
            client,
            permissions_request_key(&call_id),
            PendingPermissionRequest::RequestPermissions {
                call_id,
                option_map,
            },
            ToolCallUpdate::new(
                tool_call_id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Pending)
                    .title(reason.unwrap_or_else(|| "Permissions Request".to_string()))
                    .raw_input(raw_input)
                    .content(content),
            ),
            vec![
                PermissionOption::new(
                    REQUEST_PERMISSIONS_ALLOW_SESSION_OPTION_ID,
                    "Yes, for session",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new(
                    REQUEST_PERMISSIONS_ALLOW_TURN_OPTION_ID,
                    "Yes",
                    PermissionOptionKind::AllowOnce,
                ),
                PermissionOption::new(
                    REQUEST_PERMISSIONS_ALLOW_TURN_STRICT_OPTION_ID,
                    "Yes, with strict review for this turn",
                    PermissionOptionKind::AllowOnce,
                ),
                PermissionOption::new(
                    REQUEST_PERMISSIONS_DENY_OPTION_ID,
                    "No",
                    PermissionOptionKind::RejectOnce,
                ),
            ],
        );

        Ok(())
    }

    async fn guardian_assessment(
        &mut self,
        client: &SessionClient,
        event: GuardianAssessmentEvent,
    ) {
        let call_id = guardian_assessment_tool_call_id(&event.id);
        let status = guardian_assessment_tool_call_status(&event.status);
        let content = guardian_assessment_content(&event);
        let raw_event = serde_json::json!(&event);

        match event.status {
            GuardianAssessmentStatus::InProgress => {
                if self.active_guardian_assessments.insert(event.id.clone()) {
                    client
                        .send_tool_call(
                            ToolCall::new(call_id, "Guardian Review")
                                .kind(ToolKind::Think)
                                .status(status)
                                .content(content)
                                .raw_input(raw_event),
                        )
                        .await;
                } else {
                    client
                        .send_tool_call_update(ToolCallUpdate::new(
                            call_id,
                            ToolCallUpdateFields::new()
                                .status(status)
                                .content(content)
                                .raw_output(raw_event),
                        ))
                        .await;
                }
            }
            GuardianAssessmentStatus::Approved
            | GuardianAssessmentStatus::Denied
            | GuardianAssessmentStatus::TimedOut
            | GuardianAssessmentStatus::Aborted => {
                if self.active_guardian_assessments.remove(&event.id) {
                    client
                        .send_tool_call_update(ToolCallUpdate::new(
                            call_id,
                            ToolCallUpdateFields::new()
                                .status(status)
                                .content(content)
                                .raw_output(raw_event),
                        ))
                        .await;
                } else {
                    client
                        .send_tool_call(
                            ToolCall::new(call_id, "Guardian Review")
                                .kind(ToolKind::Think)
                                .status(status)
                                .content(content)
                                .raw_input(raw_event),
                        )
                        .await;
                }
            }
        }
    }
}

#[derive(Clone)]
struct ExecPermissionOption {
    option_id: &'static str,
    permission_option: PermissionOption,
    decision: ReviewDecision,
}

fn build_exec_permission_options(
    available_decisions: &[ReviewDecision],
    network_approval_context: Option<&NetworkApprovalContext>,
    additional_permissions: Option<&AdditionalPermissionProfile>,
) -> Vec<ExecPermissionOption> {
    available_decisions
        .iter()
        .map(|decision| match decision {
            ReviewDecision::Approved => ExecPermissionOption {
                option_id: "approved",
                permission_option: PermissionOption::new(
                    "approved",
                    if network_approval_context.is_some() {
                        "Yes, just this once"
                    } else {
                        "Yes, proceed"
                    },
                    PermissionOptionKind::AllowOnce,
                ),
                decision: ReviewDecision::Approved,
            },
            ReviewDecision::ApprovedExecpolicyAmendment {
                proposed_execpolicy_amendment,
            } => {
                let command_prefix = proposed_execpolicy_amendment.command().join(" ");
                let label = if command_prefix.contains('\n')
                    || command_prefix.contains('\r')
                    || command_prefix.is_empty()
                {
                    "Yes, and remember this command pattern".to_string()
                } else {
                    format!(
                        "Yes, and don't ask again for commands that start with `{command_prefix}`"
                    )
                };
                ExecPermissionOption {
                    option_id: "approved-execpolicy-amendment",
                    permission_option: PermissionOption::new(
                        "approved-execpolicy-amendment",
                        label,
                        PermissionOptionKind::AllowAlways,
                    ),
                    decision: ReviewDecision::ApprovedExecpolicyAmendment {
                        proposed_execpolicy_amendment: proposed_execpolicy_amendment.clone(),
                    },
                }
            }
            ReviewDecision::ApprovedForSession => ExecPermissionOption {
                option_id: "approved-for-session",
                permission_option: PermissionOption::new(
                    "approved-for-session",
                    if network_approval_context.is_some() {
                        "Yes, and allow this host for this session"
                    } else if additional_permissions.is_some() {
                        "Yes, and allow these permissions for this session"
                    } else {
                        "Yes, and don't ask again for this command in this session"
                    },
                    PermissionOptionKind::AllowAlways,
                ),
                decision: ReviewDecision::ApprovedForSession,
            },
            ReviewDecision::NetworkPolicyAmendment {
                network_policy_amendment,
            } => {
                let (option_id, label, kind) = match network_policy_amendment.action {
                    NetworkPolicyRuleAction::Allow => (
                        "network-policy-amendment-allow",
                        "Yes, and allow this host in the future",
                        PermissionOptionKind::AllowAlways,
                    ),
                    NetworkPolicyRuleAction::Deny => (
                        "network-policy-amendment-deny",
                        "No, and block this host in the future",
                        PermissionOptionKind::RejectAlways,
                    ),
                };
                ExecPermissionOption {
                    option_id,
                    permission_option: PermissionOption::new(option_id, label, kind),
                    decision: ReviewDecision::NetworkPolicyAmendment {
                        network_policy_amendment: network_policy_amendment.clone(),
                    },
                }
            }
            ReviewDecision::Denied => ExecPermissionOption {
                option_id: "denied",
                permission_option: PermissionOption::new(
                    "denied",
                    "No, continue without running it",
                    PermissionOptionKind::RejectOnce,
                ),
                decision: ReviewDecision::Denied,
            },
            ReviewDecision::Abort => ExecPermissionOption {
                option_id: "abort",
                permission_option: PermissionOption::new(
                    "abort",
                    "No, and tell Codex what to do differently",
                    PermissionOptionKind::RejectOnce,
                ),
                decision: ReviewDecision::Abort,
            },
            ReviewDecision::TimedOut => ExecPermissionOption {
                option_id: "timed-out",
                permission_option: PermissionOption::new(
                    "timed-out",
                    "Timed out",
                    PermissionOptionKind::RejectOnce,
                ),
                decision: ReviewDecision::TimedOut,
            },
        })
        .collect()
}

struct ParseCommandToolCall {
    title: String,
    file_extension: Option<String>,
    terminal_output: bool,
    locations: Vec<ToolCallLocation>,
    kind: ToolKind,
}

fn parse_command_tool_call(parsed_cmd: Vec<ParsedCommand>, cwd: &Path) -> ParseCommandToolCall {
    let mut titles = Vec::new();
    let mut locations = Vec::new();
    let mut file_extension = None;
    let mut terminal_output = false;
    let mut kind = ToolKind::Execute;

    for cmd in parsed_cmd {
        let mut cmd_path = None;
        match cmd {
            ParsedCommand::Read { cmd: _, name, path } => {
                titles.push(format!("Read {name}"));
                file_extension = path
                    .extension()
                    .map(|ext| ext.to_string_lossy().to_string());
                cmd_path = Some(path);
                kind = ToolKind::Read;
            }
            ParsedCommand::ListFiles { cmd: _, path } => {
                let dir = if let Some(path) = path.as_ref() {
                    &cwd.join(path)
                } else {
                    cwd
                };
                titles.push(format!("List {}", dir.display()));
                cmd_path = path.map(PathBuf::from);
                kind = ToolKind::Search;
            }
            ParsedCommand::Search { cmd, query, path } => {
                titles.push(match (query, path.as_ref()) {
                    (Some(query), Some(path)) => format!("Search {query} in {path}"),
                    (Some(query), None) => format!("Search {query}"),
                    _ => format!("Search {cmd}"),
                });
                kind = ToolKind::Search;
            }
            ParsedCommand::Unknown { cmd } => {
                titles.push(format!("Run {cmd}"));
                terminal_output = true;
            }
        }

        if let Some(path) = cmd_path {
            locations.push(ToolCallLocation::new(if path.is_relative() {
                cwd.join(&path)
            } else {
                path
            }));
        }
    }

    ParseCommandToolCall {
        title: titles.join(", "),
        file_extension,
        terminal_output,
        locations,
        kind,
    }
}

#[derive(Clone)]
struct SessionClient {
    session_id: SessionId,
    client: Arc<dyn Client>,
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
}

impl SessionClient {
    fn new(session_id: SessionId, client_capabilities: Arc<Mutex<ClientCapabilities>>) -> Self {
        Self {
            session_id,
            client: ACP_CLIENT.get().expect("Client should be set").clone(),
            client_capabilities,
        }
    }

    #[cfg(test)]
    fn with_client(
        session_id: SessionId,
        client: Arc<dyn Client>,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
    ) -> Self {
        Self {
            session_id,
            client,
            client_capabilities,
        }
    }

    fn supports_terminal_output(&self, active_command: &ActiveCommand) -> bool {
        active_command.terminal_output
            && self
                .client_capabilities
                .lock()
                .unwrap()
                .meta
                .as_ref()
                .is_some_and(|v| {
                    v.get("terminal_output")
                        .is_some_and(|v| v.as_bool().unwrap_or_default())
                })
    }

    fn supports_request_user_input(&self) -> bool {
        self.supports_codex_capability("requestUserInput")
    }

    fn supports_mcp_elicitation(&self) -> bool {
        self.supports_codex_capability("mcpElicitation")
    }

    fn supports_codex_capability(&self, capability: &str) -> bool {
        self.client_capabilities
            .lock()
            .unwrap()
            .meta
            .as_ref()
            .is_some_and(|v| {
                v.get("codex").is_some_and(|codex| {
                    codex
                        .get(capability)
                        .is_some_and(|enabled| enabled.as_bool().unwrap_or_default())
                })
            })
    }

    async fn request_user_input(
        &self,
        params: CodexRequestUserInputExtParams,
    ) -> Result<CodexRequestUserInputExtResponse, Error> {
        let params = serde_json::to_string(&params)
            .map_err(|e| Error::internal_error().data(e.to_string()))?;
        let params = RawValue::from_string(params)
            .map_err(|e| Error::internal_error().data(e.to_string()))?;
        let response = self
            .client
            .ext_method(ExtRequest::new(
                CODEX_REQUEST_USER_INPUT_EXT_METHOD,
                params.into(),
            ))
            .await?;

        serde_json::from_str::<CodexRequestUserInputExtResponse>(response.0.get())
            .map_err(|e| Error::invalid_params().data(e.to_string()))
    }

    async fn mcp_elicitation(
        &self,
        params: CodexMcpElicitationExtParams,
    ) -> Result<CodexMcpElicitationExtResponse, Error> {
        let params = serde_json::to_string(&params)
            .map_err(|e| Error::internal_error().data(e.to_string()))?;
        let params = RawValue::from_string(params)
            .map_err(|e| Error::internal_error().data(e.to_string()))?;
        let response = self
            .client
            .ext_method(ExtRequest::new(
                CODEX_MCP_ELICITATION_EXT_METHOD,
                params.into(),
            ))
            .await?;

        serde_json::from_str::<CodexMcpElicitationExtResponse>(response.0.get())
            .map_err(|e| Error::invalid_params().data(e.to_string()))
    }

    async fn send_notification(&self, update: SessionUpdate) {
        if let Err(e) = self
            .client
            .session_notification(SessionNotification::new(self.session_id.clone(), update))
            .await
        {
            error!("Failed to send session notification: {:?}", e);
        }
    }

    async fn send_user_message(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::UserMessageChunk(ContentChunk::new(
            text.into().into(),
        )))
        .await;
    }

    async fn send_agent_text(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            text.into().into(),
        )))
        .await;
    }

    async fn send_agent_text_with_message_id(
        &self,
        text: impl Into<String>,
        message_id: impl Into<String>,
    ) {
        self.send_notification(SessionUpdate::AgentMessageChunk(
            ContentChunk::new(text.into().into()).message_id(message_id.into()),
        ))
        .await;
    }

    async fn send_agent_message_completed(
        &self,
        message_id: impl Into<String>,
        codex_item_id: impl Into<String>,
    ) {
        self.send_notification(SessionUpdate::AgentMessageChunk(
            ContentChunk::new("".to_string().into())
                .message_id(message_id.into())
                .meta(Meta::from_iter([(
                    ANYHARNESS_META_KEY.to_string(),
                    json!({
                        "transcriptEvent": ANYHARNESS_ASSISTANT_MESSAGE_COMPLETED_EVENT,
                        "codexItemId": codex_item_id.into(),
                    }),
                )])),
        ))
        .await;
    }

    async fn send_proposed_plan_delta(
        &self,
        text: impl Into<String>,
        message_id: impl Into<String>,
        codex_item_id: impl Into<String>,
    ) {
        self.send_notification(SessionUpdate::AgentMessageChunk(
            ContentChunk::new(text.into().into())
                .message_id(message_id.into())
                .meta(Meta::from_iter([(
                    ANYHARNESS_META_KEY.to_string(),
                    json!({
                        "transcriptEvent": ANYHARNESS_PROPOSED_PLAN_DELTA_EVENT,
                        "codexItemId": codex_item_id.into(),
                    }),
                )])),
        ))
        .await;
    }

    async fn send_transient_status(&self, text: impl Into<String>, message_id: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentThoughtChunk(
            ContentChunk::new(text.into().into())
                .message_id(message_id.into())
                .meta(Meta::from_iter([(
                    ANYHARNESS_META_KEY.to_string(),
                    json!({
                        "transcriptEvent": ANYHARNESS_TRANSIENT_STATUS_EVENT,
                    }),
                )])),
        ))
        .await;
    }

    async fn send_proposed_plan_completed(
        &self,
        text: impl Into<String>,
        codex_item_id: impl Into<String>,
    ) {
        let codex_item_id = codex_item_id.into();
        self.send_notification(SessionUpdate::AgentMessageChunk(
            ContentChunk::new(text.into().into())
                .message_id(Uuid::new_v4().to_string())
                .meta(Meta::from_iter([(
                    ANYHARNESS_META_KEY.to_string(),
                    json!({
                        "transcriptEvent": ANYHARNESS_PROPOSED_PLAN_COMPLETED_EVENT,
                        "codexItemId": codex_item_id.clone(),
                        "sourceItemId": codex_item_id,
                    }),
                )])),
        ))
        .await;
    }

    async fn send_agent_thought(&self, text: impl Into<String>) {
        self.send_notification(SessionUpdate::AgentThoughtChunk(ContentChunk::new(
            text.into().into(),
        )))
        .await;
    }

    async fn send_tool_call(&self, tool_call: ToolCall) {
        self.send_notification(SessionUpdate::ToolCall(tool_call))
            .await;
    }

    async fn send_tool_call_update(&self, update: ToolCallUpdate) {
        self.send_notification(SessionUpdate::ToolCallUpdate(update))
            .await;
    }

    /// Send a completed tool call (used for replay and simple cases)
    async fn send_completed_tool_call(
        &self,
        call_id: impl Into<ToolCallId>,
        title: impl Into<String>,
        kind: ToolKind,
        raw_input: Option<serde_json::Value>,
    ) {
        let mut tool_call = ToolCall::new(call_id, title)
            .kind(kind)
            .status(ToolCallStatus::Completed);
        if let Some(input) = raw_input {
            tool_call = tool_call.raw_input(input);
        }
        self.send_tool_call(tool_call).await;
    }

    /// Send a tool call completion update (used for replay)
    async fn send_tool_call_completed(
        &self,
        call_id: impl Into<ToolCallId>,
        raw_output: Option<serde_json::Value>,
    ) {
        let mut fields = ToolCallUpdateFields::new().status(ToolCallStatus::Completed);
        if let Some(output) = raw_output {
            fields = fields.raw_output(output);
        }
        self.send_tool_call_update(ToolCallUpdate::new(call_id, fields))
            .await;
    }

    async fn update_plan(&self, plan: Vec<PlanItemArg>) {
        self.send_notification(SessionUpdate::Plan(Plan::new(
            plan.into_iter()
                .map(|entry| {
                    PlanEntry::new(
                        entry.step,
                        PlanEntryPriority::Medium,
                        match entry.status {
                            StepStatus::Pending => PlanEntryStatus::Pending,
                            StepStatus::InProgress => PlanEntryStatus::InProgress,
                            StepStatus::Completed => PlanEntryStatus::Completed,
                        },
                    )
                })
                .collect(),
        )))
        .await;
    }

    async fn request_permission(
        &self,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) -> Result<RequestPermissionResponse, Error> {
        self.client
            .request_permission(RequestPermissionRequest::new(
                self.session_id.clone(),
                tool_call,
                options,
            ))
            .await
    }
}

struct ThreadActor<A> {
    /// Allows for logging out from slash commands
    auth: A,
    /// Used for sending messages back to the client.
    client: SessionClient,
    /// The thread associated with this task.
    thread: Arc<dyn CodexThreadImpl>,
    /// The configuration for the thread.
    config: Config,
    /// The custom prompts loaded for this workspace.
    custom_prompts: Rc<RefCell<Vec<CustomPrompt>>>,
    /// The models available for this thread.
    models_manager: Arc<dyn ModelsManagerImpl>,
    /// Internal message sender used to route spawned interaction results back to the actor.
    resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
    /// A sender for each interested `Op` submission that needs events routed.
    submissions: HashMap<String, SubmissionState>,
    /// A receiver for incoming thread messages.
    message_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    /// A receiver for spawned interaction results.
    resolution_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    /// Last config options state we emitted to the client, used for deduping updates.
    last_sent_config_options: Option<Vec<SessionConfigOption>>,
    /// Active collaboration-mode mask applied to new turns.
    active_collaboration_mask: Option<CollaborationModeMask>,
}

impl<A: Auth> ThreadActor<A> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        auth: A,
        client: SessionClient,
        thread: Arc<dyn CodexThreadImpl>,
        models_manager: Arc<dyn ModelsManagerImpl>,
        config: Config,
        message_rx: mpsc::UnboundedReceiver<ThreadMessage>,
        resolution_tx: mpsc::UnboundedSender<ThreadMessage>,
        resolution_rx: mpsc::UnboundedReceiver<ThreadMessage>,
    ) -> Self {
        let active_collaboration_mask = initial_collaboration_mask(models_manager.as_ref());

        Self {
            auth,
            client,
            thread,
            config,
            custom_prompts: Rc::default(),
            models_manager,
            resolution_tx,
            submissions: HashMap::new(),
            message_rx,
            resolution_rx,
            last_sent_config_options: None,
            active_collaboration_mask,
        }
    }

    async fn spawn(mut self) {
        let mut message_rx_open = true;
        let mut prompt_watchdog = tokio::time::interval(Duration::from_secs(15));
        prompt_watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        prompt_watchdog.tick().await;
        loop {
            tokio::select! {
                biased;
                message = self.message_rx.recv(), if message_rx_open => match message {
                    Some(message) => self.handle_message(message).await,
                    None => message_rx_open = false,
                },
                message = self.resolution_rx.recv() => if let Some(message) = message {
                    self.handle_message(message).await
                },
                _ = prompt_watchdog.tick() => {
                    for submission in self.submissions.values() {
                        submission.log_pending_diagnostics();
                    }
                }
                event = self.thread.next_event() => match event {
                    Ok(event) => self.handle_event(event).await,
                    Err(e) => {
                        error!("Error getting next event: {:?}", e);
                        break;
                    }
                }
            }
            // Litter collection of senders with no receivers
            self.submissions
                .retain(|_, submission| submission.is_active());

            if !message_rx_open && self.submissions.is_empty() {
                break;
            }
        }
    }

    async fn handle_message(&mut self, message: ThreadMessage) {
        match message {
            ThreadMessage::Load { response_tx } => {
                let result = self.handle_load().await;
                drop(response_tx.send(result));
                let client = self.client.clone();
                let mut available_commands = Self::builtin_commands();
                let prompts_dir = self.config.codex_home.to_path_buf().join("prompts");
                let custom_prompts = self.custom_prompts.clone();

                tokio::task::spawn_local(async move {
                    let mut new_custom_prompts = discover_prompts_in(&prompts_dir).await;

                    for prompt in &new_custom_prompts {
                        available_commands.push(
                            AvailableCommand::new(
                                prompt.name.clone(),
                                prompt.description.clone().unwrap_or_default(),
                            )
                            .input(prompt.argument_hint.as_ref().map(
                                |hint| {
                                    AvailableCommandInput::Unstructured(
                                        UnstructuredCommandInput::new(hint.clone()),
                                    )
                                },
                            )),
                        );
                    }
                    *custom_prompts.borrow_mut() = std::mem::take(&mut new_custom_prompts);

                    client
                        .send_notification(SessionUpdate::AvailableCommandsUpdate(
                            AvailableCommandsUpdate::new(available_commands),
                        ))
                        .await;
                });
            }
            ThreadMessage::GetConfigOptions { response_tx } => {
                let result = self.config_options().await;
                drop(response_tx.send(result));
            }
            ThreadMessage::Prompt {
                request,
                response_tx,
            } => {
                let result = self.handle_prompt(request).await;
                drop(response_tx.send(result));
            }
            ThreadMessage::SetMode { mode, response_tx } => {
                let result = self.handle_set_mode(mode).await;
                drop(response_tx.send(result));
                self.maybe_emit_config_options_update().await;
            }
            ThreadMessage::SetModel { model, response_tx } => {
                let result = self.handle_set_model(model).await;
                drop(response_tx.send(result));
                self.maybe_emit_config_options_update().await;
            }
            ThreadMessage::SetConfigOption {
                config_id,
                value,
                response_tx,
            } => {
                let result = self.handle_set_config_option(config_id, value).await;
                let should_emit_update = result.is_ok();
                drop(response_tx.send(result));
                if should_emit_update {
                    self.maybe_emit_config_options_update().await;
                }
            }
            ThreadMessage::Cancel { response_tx } => {
                let result = self.handle_cancel().await;
                drop(response_tx.send(result));
            }
            ThreadMessage::Shutdown { response_tx } => {
                let result = self.handle_shutdown().await;
                drop(response_tx.send(result));
            }
            ThreadMessage::ReplayHistory {
                history,
                response_tx,
            } => {
                let result = self.handle_replay_history(history).await;
                drop(response_tx.send(result));
            }
            ThreadMessage::PermissionRequestResolved {
                submission_id,
                request_key,
                response,
            } => {
                let Some(submission) = self.submissions.get_mut(&submission_id) else {
                    warn!(
                        "Ignoring permission response for unknown submission ID: {submission_id}"
                    );
                    return;
                };

                if let Err(err) = submission
                    .handle_permission_request_resolved(&self.client, request_key, response)
                    .await
                {
                    submission.abort_pending_interactions();
                    submission.fail(err);
                }
            }
        }
    }

    fn builtin_commands() -> Vec<AvailableCommand> {
        vec![
            AvailableCommand::new("review", "Review my current changes and find issues").input(
                AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                    "optional custom review instructions",
                )),
            ),
            AvailableCommand::new(
                "review-branch",
                "Review the code changes against a specific branch",
            )
            .input(AvailableCommandInput::Unstructured(
                UnstructuredCommandInput::new("branch name"),
            )),
            AvailableCommand::new(
                "review-commit",
                "Review the code changes introduced by a commit",
            )
            .input(AvailableCommandInput::Unstructured(
                UnstructuredCommandInput::new("commit sha"),
            )),
            AvailableCommand::new(
                "init",
                "create an AGENTS.md file with instructions for Codex",
            ),
            AvailableCommand::new(
                "compact",
                "summarize conversation to prevent hitting the context limit",
            ),
            AvailableCommand::new("undo", "undo Codex’s most recent turn"),
            AvailableCommand::new("logout", "logout of Codex"),
        ]
    }

    fn modes(&self) -> Option<SessionModeState> {
        let current_mode_id = APPROVAL_PRESETS
            .iter()
            .find(|preset| {
                std::mem::discriminant(&preset.approval)
                    == std::mem::discriminant(self.config.permissions.approval_policy.get())
                    && std::mem::discriminant(&preset.sandbox)
                        == std::mem::discriminant(self.config.permissions.sandbox_policy.get())
            })
            .or_else(|| {
                // When the project is untrusted, the above code won't match
                // since AskForApproval::UnlessTrusted is not part of the
                // default presets. However, in this case we still want to show
                // the mode selector, which allows the user to choose a
                // different mode (which will set the project to be trusted)
                // See https://github.com/zed-industries/zed/issues/48132
                if self.config.active_project.is_untrusted() {
                    APPROVAL_PRESETS
                        .iter()
                        .find(|preset| preset.id == "read-only")
                } else {
                    None
                }
            })
            .map(|preset| SessionModeId::new(preset.id))?;

        Some(SessionModeState::new(
            current_mode_id,
            APPROVAL_PRESETS
                .iter()
                .map(|preset| {
                    SessionMode::new(preset.id, preset.label).description(preset.description)
                })
                .collect(),
        ))
    }

    async fn find_current_model(&self) -> Option<ModelId> {
        let model_presets = self.models_manager.list_models().await;
        let effective_mode = self.effective_collaboration_mode().await;
        let config_model = effective_mode.model().to_string();
        let preset = model_presets
            .iter()
            .find(|preset| preset.model == config_model)?;

        let effort = effective_mode
            .reasoning_effort()
            .and_then(|effort| {
                preset
                    .supported_reasoning_efforts
                    .iter()
                    .find_map(|e| (e.effort == effort).then_some(effort))
            })
            .unwrap_or(preset.default_reasoning_effort);

        Some(Self::model_id(&preset.id, effort))
    }

    fn model_id(id: &str, effort: ReasoningEffort) -> ModelId {
        ModelId::new(format!("{id}/{effort}"))
    }

    fn parse_model_id(id: &ModelId) -> Option<(String, ReasoningEffort)> {
        let (model, reasoning) = id.0.split_once('/')?;
        let reasoning = serde_json::from_value(reasoning.into()).ok()?;
        Some((model.to_owned(), reasoning))
    }

    fn collaboration_mode_presets(&self) -> Vec<CollaborationModeMask> {
        self.models_manager
            .list_collaboration_modes()
            .into_iter()
            .filter(|mask| mask.mode.is_some_and(ModeKind::is_tui_visible))
            .collect()
    }

    fn active_mode_kind(&self) -> ModeKind {
        self.active_collaboration_mask
            .as_ref()
            .and_then(|mask| mask.mode)
            .unwrap_or(ModeKind::Default)
    }

    async fn effective_collaboration_mode(&self) -> CollaborationMode {
        let current = CollaborationMode {
            mode: ModeKind::Default,
            settings: Settings {
                model: self.get_current_model().await,
                reasoning_effort: self.config.model_reasoning_effort,
                developer_instructions: None,
            },
        };

        self.active_collaboration_mask
            .as_ref()
            .map_or(current.clone(), |mask| current.apply_mask(mask))
    }

    async fn config_options(&self) -> Result<Vec<SessionConfigOption>, Error> {
        let mut options = Vec::new();

        if let Some(modes) = self.modes() {
            let select_options = modes
                .available_modes
                .into_iter()
                .map(|m| SessionConfigSelectOption::new(m.id.0, m.name).description(m.description))
                .collect::<Vec<_>>();

            options.push(
                SessionConfigOption::select(
                    "mode",
                    "Approval Preset",
                    modes.current_mode_id.0,
                    select_options,
                )
                .category(SessionConfigOptionCategory::Mode)
                .description("Choose an approval and sandboxing preset for your session"),
            );
        }

        let collaboration_mode_options = self.collaboration_mode_presets();
        if !collaboration_mode_options.is_empty() {
            let select_options = collaboration_mode_options
                .iter()
                .filter_map(|mask| {
                    let kind = mask.mode?;
                    let description = match kind {
                        ModeKind::Plan => "Switch to planning mode with Codex plan instructions",
                        ModeKind::Default => {
                            "Switch to default execution mode with Codex default instructions"
                        }
                        _ => return None,
                    };

                    Some(
                        SessionConfigSelectOption::new(
                            collaboration_mode_value_id(kind),
                            mask.name.clone(),
                        )
                        .description(description),
                    )
                })
                .collect::<Vec<_>>();

            options.push(
                SessionConfigOption::select(
                    "collaboration_mode",
                    "Collaboration Mode",
                    collaboration_mode_value_id(self.active_mode_kind()),
                    select_options,
                )
                .category(SessionConfigOptionCategory::Other(
                    "collaboration_mode".into(),
                ))
                .description("Choose whether Codex should work in default or planning mode"),
            );
        }

        let presets = self.models_manager.list_models().await;

        let effective_mode = self.effective_collaboration_mode().await;
        let current_model = effective_mode.model().to_string();
        let current_preset = presets.iter().find(|p| p.model == current_model).cloned();

        let mut model_select_options = Vec::new();

        if current_preset.is_none() {
            // If no preset found, return the current model string as-is
            model_select_options.push(SessionConfigSelectOption::new(
                current_model.clone(),
                current_model.clone(),
            ));
        };

        model_select_options.extend(
            presets
                .into_iter()
                .filter(|model| model.show_in_picker || model.model == current_model)
                .map(|preset| {
                    SessionConfigSelectOption::new(preset.id, preset.display_name)
                        .description(preset.description)
                }),
        );

        options.push(
            SessionConfigOption::select(
                "model",
                "Model",
                current_model.clone(),
                model_select_options,
            )
            .category(SessionConfigOptionCategory::Model)
            .description("Choose which model Codex should use"),
        );

        // Reasoning effort selector (only if the current preset exists and has >1 supported effort)
        if let Some(preset) = current_preset.as_ref()
            && preset.supported_reasoning_efforts.len() > 1
        {
            let supported = &preset.supported_reasoning_efforts;

            let current_effort = effective_mode
                .reasoning_effort()
                .and_then(|effort| {
                    supported
                        .iter()
                        .find_map(|e| (e.effort == effort).then_some(effort))
                })
                .unwrap_or(preset.default_reasoning_effort);

            let effort_select_options = supported
                .iter()
                .map(|e| {
                    SessionConfigSelectOption::new(
                        e.effort.to_string(),
                        e.effort.to_string().to_title_case(),
                    )
                    .description(e.description.clone())
                })
                .collect::<Vec<_>>();

            options.push(
                SessionConfigOption::select(
                    "reasoning_effort",
                    "Reasoning Effort",
                    current_effort.to_string(),
                    effort_select_options,
                )
                .category(SessionConfigOptionCategory::ThoughtLevel)
                .description("Choose how much reasoning effort the model should use"),
            );
        }

        let fast_mode_select_options = vec![
            SessionConfigSelectOption::new("off", "Off")
                .description("Use Codex's default service tier"),
            SessionConfigSelectOption::new("on", "On")
                .description("Request the fast Codex service tier"),
        ];
        let fast_mode_current_value = if self.config.service_tier == Some(ServiceTier::Fast) {
            "on"
        } else {
            "off"
        };
        options.push(
            SessionConfigOption::select(
                "fast_mode",
                "Fast Mode",
                fast_mode_current_value,
                fast_mode_select_options,
            )
            .category(SessionConfigOptionCategory::Other("fast_mode".to_string()))
            .description("Choose whether Codex should use fast mode"),
        );

        Ok(options)
    }

    async fn maybe_emit_config_options_update(&mut self) {
        let config_options = self.config_options().await.unwrap_or_default();

        if self
            .last_sent_config_options
            .as_ref()
            .is_some_and(|prev| prev == &config_options)
        {
            return;
        }

        self.last_sent_config_options = Some(config_options.clone());

        self.client
            .send_notification(SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(
                config_options,
            )))
            .await;
    }

    async fn handle_set_config_option(
        &mut self,
        config_id: SessionConfigId,
        value: SessionConfigOptionValue,
    ) -> Result<(), Error> {
        let SessionConfigOptionValue::ValueId { value } = value else {
            return Err(Error::invalid_params().data("Unsupported config option value"));
        };
        match config_id.0.as_ref() {
            "mode" => self.handle_set_mode(SessionModeId::new(value.0)).await,
            "collaboration_mode" => self.handle_set_config_collaboration_mode(value).await,
            "model" => self.handle_set_config_model(value).await,
            "reasoning_effort" => self.handle_set_config_reasoning_effort(value).await,
            "fast_mode" => self.handle_set_config_fast_mode(value).await,
            _ => Err(Error::invalid_params().data("Unsupported config option")),
        }
    }

    async fn handle_set_config_model(&mut self, value: SessionConfigValueId) -> Result<(), Error> {
        let model_id = value.0;

        let presets = self.models_manager.list_models().await;
        let preset = presets.iter().find(|p| p.id.as_str() == &*model_id);

        let model_to_use = preset
            .map(|p| p.model.clone())
            .unwrap_or_else(|| model_id.to_string());

        if model_to_use.is_empty() {
            return Err(Error::invalid_params().data("No model selected"));
        }

        let effort_to_use = if let Some(preset) = preset {
            if let Some(effort) = self.config.model_reasoning_effort
                && preset
                    .supported_reasoning_efforts
                    .iter()
                    .any(|e| e.effort == effort)
            {
                Some(effort)
            } else {
                Some(preset.default_reasoning_effort)
            }
        } else {
            // If the user selected a raw model string (not a known preset), don't invent a default.
            // Keep whatever was previously configured (or leave unset) so Codex can decide.
            self.config.model_reasoning_effort
        };

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(model_to_use.clone()),
                effort: Some(effort_to_use),
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
                service_tier: None,
                approvals_reviewer: None,
                permission_profile: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;
        if let Some(mask) = self.active_collaboration_mask.as_mut() {
            mask.model = self.config.model.clone();
            mask.reasoning_effort = Some(effort_to_use);
        }

        Ok(())
    }

    async fn handle_set_config_reasoning_effort(
        &mut self,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        let effort: ReasoningEffort =
            serde_json::from_value(value.0.as_ref().into()).map_err(|_| Error::invalid_params())?;

        let current_model = self
            .effective_collaboration_mode()
            .await
            .model()
            .to_string();
        let presets = self.models_manager.list_models().await;
        let Some(preset) = presets.iter().find(|p| p.model == current_model) else {
            return Err(Error::invalid_params()
                .data("Reasoning effort can only be set for known model presets"));
        };

        if !preset
            .supported_reasoning_efforts
            .iter()
            .any(|e| e.effort == effort)
        {
            return Err(
                Error::invalid_params().data("Unsupported reasoning effort for selected model")
            );
        }

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: None,
                effort: Some(Some(effort)),
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
                service_tier: None,
                approvals_reviewer: None,
                permission_profile: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model_reasoning_effort = Some(effort);
        if let Some(mask) = self.active_collaboration_mask.as_mut() {
            mask.reasoning_effort = Some(Some(effort));
        }

        Ok(())
    }

    async fn handle_set_config_collaboration_mode(
        &mut self,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        let selected_kind = collaboration_mode_kind_from_id(value.0.as_ref())
            .ok_or_else(|| Error::invalid_params().data("Unsupported collaboration mode"))?;
        let selected_mask = self
            .collaboration_mode_presets()
            .into_iter()
            .find(|mask| mask.mode == Some(selected_kind))
            .ok_or_else(|| Error::invalid_params().data("Unsupported collaboration mode"))?;

        let current_model = self.get_current_model().await;
        let collaboration_mode = CollaborationMode {
            mode: selected_kind,
            settings: Settings {
                model: selected_mask
                    .model
                    .clone()
                    .unwrap_or_else(|| current_model.clone()),
                reasoning_effort: selected_mask
                    .reasoning_effort
                    .unwrap_or(self.config.model_reasoning_effort),
                developer_instructions: selected_mask
                    .developer_instructions
                    .clone()
                    .unwrap_or(None),
            },
        };

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: None,
                effort: None,
                summary: None,
                collaboration_mode: Some(collaboration_mode.clone()),
                personality: None,
                windows_sandbox_level: None,
                service_tier: None,
                approvals_reviewer: None,
                permission_profile: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(collaboration_mode.settings.model.clone());
        self.config.model_reasoning_effort = collaboration_mode.settings.reasoning_effort;
        self.active_collaboration_mask = Some(selected_mask);

        Ok(())
    }

    async fn handle_set_config_fast_mode(
        &mut self,
        value: SessionConfigValueId,
    ) -> Result<(), Error> {
        let service_tier = match value.0.as_ref() {
            "on" => Some(ServiceTier::Fast),
            "off" => None,
            _ => return Err(Error::invalid_params().data("Unsupported fast mode value")),
        };

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: None,
                effort: None,
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
                service_tier: Some(service_tier),
                approvals_reviewer: None,
                permission_profile: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.service_tier = service_tier;

        Ok(())
    }

    async fn models(&self) -> Result<SessionModelState, Error> {
        let mut available_models = Vec::new();
        let config_model = self
            .effective_collaboration_mode()
            .await
            .model()
            .to_string();

        let current_model_id = if let Some(model_id) = self.find_current_model().await {
            model_id
        } else {
            // If no preset found, return the current model string as-is
            let model_id = ModelId::new(config_model.clone());
            available_models.push(ModelInfo::new(model_id.clone(), model_id.to_string()));
            model_id
        };

        available_models.extend(
            self.models_manager
                .list_models()
                .await
                .iter()
                .filter(|model| model.show_in_picker || model.model == config_model)
                .flat_map(|preset| {
                    preset.supported_reasoning_efforts.iter().map(|effort| {
                        ModelInfo::new(
                            Self::model_id(&preset.id, effort.effort),
                            format!("{} ({})", preset.display_name, effort.effort),
                        )
                        .description(format!("{} {}", preset.description, effort.description))
                    })
                }),
        );

        Ok(SessionModelState::new(current_model_id, available_models))
    }

    async fn handle_load(&mut self) -> Result<LoadSessionResponse, Error> {
        Ok(LoadSessionResponse::new()
            .models(self.models().await?)
            .modes(self.modes())
            .config_options(self.config_options().await?))
    }

    async fn handle_prompt(
        &mut self,
        request: PromptRequest,
    ) -> Result<oneshot::Receiver<Result<StopReason, Error>>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let items = build_prompt_items(request.prompt);
        let op;
        if let Some((name, rest)) = extract_slash_command(&items) {
            match name {
                "compact" => op = Op::Compact,
                "undo" => op = Op::Undo,
                "init" => {
                    op = Op::UserInput {
                        items: vec![UserInput::Text {
                            text: INIT_COMMAND_PROMPT.into(),
                            text_elements: vec![],
                        }],
                        environments: None,
                        final_output_json_schema: None,
                        responsesapi_client_metadata: None,
                    }
                }
                "review" => {
                    let instructions = rest.trim();
                    let target = if instructions.is_empty() {
                        ReviewTarget::UncommittedChanges
                    } else {
                        ReviewTarget::Custom {
                            instructions: instructions.to_owned(),
                        }
                    };

                    op = Op::Review {
                        review_request: ReviewRequest {
                            user_facing_hint: Some(user_facing_hint(&target)),
                            target,
                        },
                    }
                }
                "review-branch" if !rest.is_empty() => {
                    let target = ReviewTarget::BaseBranch {
                        branch: rest.trim().to_owned(),
                    };
                    op = Op::Review {
                        review_request: ReviewRequest {
                            user_facing_hint: Some(user_facing_hint(&target)),
                            target,
                        },
                    }
                }
                "review-commit" if !rest.is_empty() => {
                    let target = ReviewTarget::Commit {
                        sha: rest.trim().to_owned(),
                        title: None,
                    };
                    op = Op::Review {
                        review_request: ReviewRequest {
                            user_facing_hint: Some(user_facing_hint(&target)),
                            target,
                        },
                    }
                }
                "logout" => {
                    self.auth.logout()?;
                    return Err(Error::auth_required());
                }
                _ => {
                    if let Some(prompt) =
                        expand_custom_prompt(name, rest, self.custom_prompts.borrow().as_ref())
                            .map_err(|e| Error::invalid_params().data(e.user_message()))?
                    {
                        op = Op::UserInput {
                            items: vec![UserInput::Text {
                                text: prompt,
                                text_elements: vec![],
                            }],
                            environments: None,
                            final_output_json_schema: None,
                            responsesapi_client_metadata: None,
                        }
                    } else {
                        op = Op::UserInput {
                            items,
                            environments: None,
                            final_output_json_schema: None,
                            responsesapi_client_metadata: None,
                        }
                    }
                }
            }
        } else {
            op = Op::UserInput {
                items,
                environments: None,
                final_output_json_schema: None,
                responsesapi_client_metadata: None,
            }
        }

        let submission_id = self
            .thread
            .submit(op.clone())
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        info!("Submitted prompt with submission_id: {submission_id}");
        info!("Starting to wait for conversation events for submission_id: {submission_id}");

        let state = SubmissionState::Prompt(PromptState::new(
            submission_id.clone(),
            self.thread.clone(),
            self.resolution_tx.clone(),
            response_tx,
        ));

        self.submissions.insert(submission_id, state);

        Ok(response_rx)
    }

    async fn handle_set_mode(&mut self, mode: SessionModeId) -> Result<(), Error> {
        let preset = APPROVAL_PRESETS
            .iter()
            .find(|preset| mode.0.as_ref() == preset.id)
            .ok_or_else(Error::invalid_params)?;

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: Some(preset.approval),
                sandbox_policy: Some(preset.sandbox.clone()),
                model: None,
                effort: None,
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
                service_tier: None,
                approvals_reviewer: None,
                permission_profile: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config
            .permissions
            .approval_policy
            .set(preset.approval)
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        self.config
            .permissions
            .sandbox_policy
            .set(preset.sandbox.clone())
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        match preset.sandbox {
            // Treat this user action as a trusted dir
            SandboxPolicy::DangerFullAccess
            | SandboxPolicy::WorkspaceWrite { .. }
            | SandboxPolicy::ExternalSandbox { .. } => {
                set_project_trust_level(
                    &self.config.codex_home,
                    &self.config.cwd,
                    TrustLevel::Trusted,
                )?;
            }
            SandboxPolicy::ReadOnly { .. } => {}
        }

        Ok(())
    }

    async fn get_current_model(&self) -> String {
        self.models_manager.get_model(&self.config.model).await
    }

    async fn handle_set_model(&mut self, model: ModelId) -> Result<(), Error> {
        // Try parsing as preset format, otherwise use as-is, fallback to config
        let (model_to_use, effort_to_use) = if let Some((m, e)) = Self::parse_model_id(&model) {
            (m, Some(e))
        } else {
            let model_str = model.0.to_string();
            let fallback = if !model_str.is_empty() {
                model_str
            } else {
                self.get_current_model().await
            };
            (fallback, self.config.model_reasoning_effort)
        };

        if model_to_use.is_empty() {
            return Err(Error::invalid_params().data("No model parsed or configured"));
        }

        self.thread
            .submit(Op::OverrideTurnContext {
                cwd: None,
                approval_policy: None,
                sandbox_policy: None,
                model: Some(model_to_use.clone()),
                effort: Some(effort_to_use),
                summary: None,
                collaboration_mode: None,
                personality: None,
                windows_sandbox_level: None,
                service_tier: None,
                approvals_reviewer: None,
                permission_profile: None,
            })
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;

        self.config.model = Some(model_to_use);
        self.config.model_reasoning_effort = effort_to_use;
        if let Some(mask) = self.active_collaboration_mask.as_mut() {
            mask.model = self.config.model.clone();
            mask.reasoning_effort = Some(effort_to_use);
        }

        Ok(())
    }

    async fn handle_cancel(&mut self) -> Result<(), Error> {
        self.abort_pending_interactions();
        self.thread
            .submit(Op::Interrupt)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    async fn handle_shutdown(&mut self) -> Result<(), Error> {
        self.abort_pending_interactions();
        self.thread
            .submit(Op::Shutdown)
            .await
            .map_err(|e| Error::from(anyhow::anyhow!(e)))?;
        Ok(())
    }

    fn abort_pending_interactions(&mut self) {
        for submission in self.submissions.values_mut() {
            submission.abort_pending_interactions();
        }
    }

    /// Replay conversation history to the client via session/update notifications.
    /// This is called when loading a session to stream all prior messages.
    ///
    /// We process both `EventMsg` and `ResponseItem`:
    /// - `EventMsg` for user/agent messages and reasoning (like the TUI does)
    /// - `ResponseItem` for tool calls only (not persisted as EventMsg)
    async fn handle_replay_history(&mut self, history: Vec<RolloutItem>) -> Result<(), Error> {
        for item in history {
            match item {
                RolloutItem::EventMsg(event_msg) => {
                    self.replay_event_msg(&event_msg).await;
                }
                RolloutItem::ResponseItem(response_item) => {
                    self.replay_response_item(&response_item).await;
                }
                // Skip SessionMeta, TurnContext, Compacted
                _ => {}
            }
        }
        Ok(())
    }

    /// Convert and send an EventMsg as ACP notification(s) during replay.
    /// Handles messages and reasoning - mirrors the live event handling in PromptState.
    async fn replay_event_msg(&mut self, msg: &EventMsg) {
        match msg {
            EventMsg::TurnStarted(TurnStartedEvent {
                collaboration_mode_kind,
                ..
            }) => {
                self.sync_collaboration_mode_kind(*collaboration_mode_kind);
            }
            EventMsg::UserMessage(UserMessageEvent { message, .. }) => {
                self.client.send_user_message(message.clone()).await;
            }
            EventMsg::AgentMessage(AgentMessageEvent {
                message,
                phase: _,
                memory_citation: _,
            }) => {
                self.client.send_agent_text(message.clone()).await;
            }
            EventMsg::AgentReasoning(AgentReasoningEvent { text }) => {
                self.client.send_agent_thought(text.clone()).await;
            }
            EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent { text }) => {
                self.client.send_agent_thought(text.clone()).await;
            }
            // Skip other event types during replay - they either:
            // - Are transient (deltas, turn lifecycle)
            // - Don't have direct ACP equivalents
            // - Are handled via ResponseItem instead
            _ => {}
        }
    }

    /// Parse apply_patch call input to extract patch content for display.
    /// Returns (title, locations, content) if successful.
    /// For CustomToolCall, the input is the patch string directly.
    fn parse_apply_patch_call(
        &self,
        input: &str,
    ) -> Option<(String, Vec<ToolCallLocation>, Vec<ToolCallContent>)> {
        // Try to parse the patch using codex-apply-patch parser
        let parsed = parse_patch(input).ok()?;

        let mut locations = Vec::new();
        let mut file_names = Vec::new();
        let mut content = Vec::new();

        for hunk in &parsed.hunks {
            match hunk {
                codex_apply_patch::Hunk::AddFile { path, contents } => {
                    let full_path = self.config.cwd.as_path().join(path);
                    file_names.push(path.display().to_string());
                    locations.push(ToolCallLocation::new(full_path.clone()));
                    // New file: no old_text, new_text is the contents
                    content.push(ToolCallContent::Diff(Diff::new(
                        full_path,
                        contents.clone(),
                    )));
                }
                codex_apply_patch::Hunk::DeleteFile { path } => {
                    let full_path = self.config.cwd.as_path().join(path);
                    file_names.push(path.display().to_string());
                    locations.push(ToolCallLocation::new(full_path.clone()));
                    // Delete file: old_text would be original content, new_text is empty
                    content.push(ToolCallContent::Diff(
                        Diff::new(full_path, "").old_text("[file deleted]"),
                    ));
                }
                codex_apply_patch::Hunk::UpdateFile {
                    path,
                    move_path,
                    chunks,
                } => {
                    let full_path = self.config.cwd.as_path().join(path);
                    let dest_path = move_path
                        .as_ref()
                        .map(|p| self.config.cwd.as_path().join(p))
                        .unwrap_or_else(|| full_path.clone());
                    file_names.push(path.display().to_string());
                    locations.push(ToolCallLocation::new(dest_path.clone()));

                    // Build old and new text from chunks
                    let old_lines: Vec<String> = chunks
                        .iter()
                        .flat_map(|c| c.old_lines.iter().cloned())
                        .collect();
                    let new_lines: Vec<String> = chunks
                        .iter()
                        .flat_map(|c| c.new_lines.iter().cloned())
                        .collect();

                    content.push(ToolCallContent::Diff(
                        Diff::new(dest_path, new_lines.join("\n")).old_text(old_lines.join("\n")),
                    ));
                }
            }
        }

        let title = if file_names.is_empty() {
            "Apply patch".to_string()
        } else {
            format!("Edit {}", file_names.join(", "))
        };

        Some((title, locations, content))
    }

    /// Parse shell function call arguments to extract command info for rich display.
    /// Returns (title, kind, locations) if successful.
    ///
    /// Handles both:
    /// - `shell` / `container.exec`: `command` is `Vec<String>`
    /// - `shell_command`: `command` is a `String` (shell script)
    fn parse_shell_function_call(
        &self,
        name: &str,
        arguments: &str,
    ) -> Option<(String, ToolKind, Vec<ToolCallLocation>)> {
        // Extract command and workdir based on tool type
        let (command_vec, workdir): (Vec<String>, Option<String>) = if name == "shell_command" {
            // shell_command: command is a string (shell script)
            #[derive(serde::Deserialize)]
            struct ShellCommandArgs {
                command: String,
                #[serde(default)]
                workdir: Option<String>,
            }
            let args: ShellCommandArgs = serde_json::from_str(arguments).ok()?;
            // Wrap in bash -lc for parsing
            (
                vec!["bash".to_string(), "-lc".to_string(), args.command],
                args.workdir,
            )
        } else {
            // shell / container.exec: command is Vec<String>
            #[derive(serde::Deserialize)]
            struct ShellArgs {
                command: Vec<String>,
                #[serde(default)]
                workdir: Option<String>,
            }
            let args: ShellArgs = serde_json::from_str(arguments).ok()?;
            (args.command, args.workdir)
        };

        let cwd = workdir
            .map(PathBuf::from)
            .unwrap_or_else(|| self.config.cwd.clone().into());

        let parsed_cmd = parse_command(&command_vec);
        let ParseCommandToolCall {
            title,
            file_extension: _,
            terminal_output: _,
            locations,
            kind,
        } = parse_command_tool_call(parsed_cmd, &cwd);

        Some((title, kind, locations))
    }

    /// Convert and send a single ResponseItem as ACP notification(s) during replay.
    /// Only handles tool calls - messages/reasoning are handled via EventMsg.
    async fn replay_response_item(&self, item: &ResponseItem) {
        match item {
            // Skip Message and Reasoning - these are handled via EventMsg
            ResponseItem::Message { .. } | ResponseItem::Reasoning { .. } => {}
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                // Check if this is a shell command - parse it like we do for LocalShellCall
                if matches!(name.as_str(), "shell" | "container.exec" | "shell_command")
                    && let Some((title, kind, locations)) =
                        self.parse_shell_function_call(name, arguments)
                {
                    self.client
                        .send_tool_call(
                            ToolCall::new(call_id.clone(), title)
                                .kind(kind)
                                .status(ToolCallStatus::Completed)
                                .locations(locations)
                                .raw_input(
                                    serde_json::from_str::<serde_json::Value>(arguments).ok(),
                                ),
                        )
                        .await;
                    return;
                }

                // Fall through to generic function call handling
                self.client
                    .send_completed_tool_call(
                        call_id.clone(),
                        name.clone(),
                        ToolKind::Other,
                        serde_json::from_str(arguments).ok(),
                    )
                    .await;
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                self.client
                    .send_tool_call_completed(call_id.clone(), serde_json::to_value(output).ok())
                    .await;
            }
            ResponseItem::LocalShellCall {
                call_id: Some(call_id),
                action,
                status,
                ..
            } => {
                let codex_protocol::models::LocalShellAction::Exec(exec) = action;
                let cwd = exec
                    .working_directory
                    .as_ref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.config.cwd.clone().into());

                // Parse the command to get rich info like the live event handler does
                let parsed_cmd = parse_command(&exec.command);
                let ParseCommandToolCall {
                    title,
                    file_extension: _,
                    terminal_output: _,
                    locations,
                    kind,
                } = parse_command_tool_call(parsed_cmd, &cwd);

                let tool_status = match status {
                    codex_protocol::models::LocalShellStatus::Completed => {
                        ToolCallStatus::Completed
                    }
                    codex_protocol::models::LocalShellStatus::InProgress
                    | codex_protocol::models::LocalShellStatus::Incomplete => {
                        ToolCallStatus::Failed
                    }
                };
                self.client
                    .send_tool_call(
                        ToolCall::new(call_id.clone(), title)
                            .kind(kind)
                            .status(tool_status)
                            .locations(locations),
                    )
                    .await;
            }
            ResponseItem::CustomToolCall {
                name,
                input,
                call_id,
                ..
            } => {
                // Check if this is an apply_patch call - show the patch content
                if name == "apply_patch"
                    && let Some((title, locations, content)) = self.parse_apply_patch_call(input)
                {
                    self.client
                        .send_tool_call(
                            ToolCall::new(call_id.clone(), title)
                                .kind(ToolKind::Edit)
                                .status(ToolCallStatus::Completed)
                                .locations(locations)
                                .content(content)
                                .raw_input(serde_json::from_str::<serde_json::Value>(input).ok()),
                        )
                        .await;
                    return;
                }

                // Fall through to generic custom tool call handling
                self.client
                    .send_completed_tool_call(
                        call_id.clone(),
                        name.clone(),
                        ToolKind::Other,
                        serde_json::from_str(input).ok(),
                    )
                    .await;
            }
            ResponseItem::CustomToolCallOutput {
                name: _,
                call_id,
                output,
            } => {
                self.client
                    .send_tool_call_completed(call_id.clone(), Some(serde_json::json!(output)))
                    .await;
            }
            ResponseItem::WebSearchCall { id, action, .. } => {
                let (title, call_id) = if let Some(action) = action {
                    web_search_action_to_title_and_id(id, action)
                } else {
                    ("Web Search".into(), generate_fallback_id("web_search"))
                };
                self.client
                    .send_tool_call(
                        ToolCall::new(call_id, title)
                            .kind(ToolKind::Search)
                            .status(ToolCallStatus::Completed),
                    )
                    .await;
            }
            // Skip GhostSnapshot, Compaction, Other, LocalShellCall without call_id
            _ => {}
        }
    }

    async fn handle_event(&mut self, Event { id, msg }: Event) {
        if let EventMsg::TurnStarted(TurnStartedEvent {
            collaboration_mode_kind,
            ..
        }) = &msg
            && self.sync_collaboration_mode_kind(*collaboration_mode_kind)
        {
            self.maybe_emit_config_options_update().await;
        }

        if let Some(submission) = self.submissions.get_mut(&id) {
            submission.handle_event(&self.client, msg).await;
        } else if let EventMsg::ElicitationRequest(event) = &msg {
            let Some(submission_id) = self.submission_id_for_unscoped_elicitation(event) else {
                self.decline_unroutable_mcp_elicitation(&id, event).await;
                return;
            };

            let Some(submission) = self.submissions.get_mut(&submission_id) else {
                warn!("Resolved MCP elicitation route to missing submission ID: {submission_id}");
                self.decline_unroutable_mcp_elicitation(&id, event).await;
                return;
            };

            info!(
                "Routing MCP elicitation event with id {id} to active submission {submission_id}"
            );
            submission.handle_event(&self.client, msg).await;
        } else {
            warn!("Received event for unknown submission ID: {id} {msg:?}");
        }
    }

    async fn decline_unroutable_mcp_elicitation(
        &self,
        event_id: &str,
        event: &ElicitationRequestEvent,
    ) {
        warn!(
            "Declining MCP elicitation event with no routable active submission: {event_id} {event:?}"
        );
        if let Err(err) = self
            .thread
            .submit(Op::ResolveElicitation {
                server_name: event.server_name.clone(),
                request_id: event.id.clone(),
                decision: ElicitationAction::Decline,
                content: None,
                meta: None,
            })
            .await
        {
            warn!("Failed to decline unroutable MCP elicitation: {err:?}");
        }
    }

    fn submission_id_for_unscoped_elicitation(
        &self,
        event: &ElicitationRequestEvent,
    ) -> Option<String> {
        if let Some(turn_id) = &event.turn_id {
            if self.submissions.contains_key(turn_id) {
                return Some(turn_id.clone());
            }

            let mut matches = self
                .submissions
                .iter()
                .filter(|(_, submission)| submission.current_turn_id() == Some(turn_id.as_str()))
                .map(|(submission_id, _)| submission_id.clone());
            let first = matches.next()?;
            if matches.next().is_none() {
                return Some(first);
            }

            return None;
        }

        let mut active = self
            .submissions
            .iter()
            .filter(|(_, submission)| submission.is_active())
            .map(|(submission_id, _)| submission_id.clone());
        let first = active.next()?;
        if active.next().is_none() {
            Some(first)
        } else {
            None
        }
    }

    fn sync_collaboration_mode_kind(&mut self, kind: ModeKind) -> bool {
        if self.active_mode_kind() == kind {
            return false;
        }

        if let Some(mask) = self
            .collaboration_mode_presets()
            .into_iter()
            .find(|candidate| candidate.mode == Some(kind))
        {
            self.active_collaboration_mask = Some(mask);
            return true;
        }

        false
    }
}

fn build_prompt_items(prompt: Vec<ContentBlock>) -> Vec<UserInput> {
    prompt
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text_block) => Some(UserInput::Text {
                text: text_block.text,
                text_elements: vec![],
            }),
            ContentBlock::Image(image_block) => Some(UserInput::Image {
                image_url: format!("data:{};base64,{}", image_block.mime_type, image_block.data),
            }),
            ContentBlock::ResourceLink(ResourceLink { name, uri, .. }) => Some(UserInput::Text {
                text: format_uri_as_link(Some(name), uri),
                text_elements: vec![],
            }),
            ContentBlock::Resource(EmbeddedResource {
                resource:
                    EmbeddedResourceResource::TextResourceContents(TextResourceContents {
                        text,
                        uri,
                        ..
                    }),
                ..
            }) => Some(UserInput::Text {
                text: format!(
                    "{}\n<context ref=\"{uri}\">\n{text}\n</context>",
                    format_uri_as_link(None, uri.clone())
                ),
                text_elements: vec![],
            }),
            // Skip other content types for now
            ContentBlock::Audio(..) | ContentBlock::Resource(..) | _ => None,
        })
        .collect()
}

fn format_uri_as_link(name: Option<String>, uri: String) -> String {
    if let Some(name) = name
        && !name.is_empty()
    {
        format!("[@{name}]({uri})")
    } else if let Some(path) = uri.strip_prefix("file://") {
        let name = path.split('/').next_back().unwrap_or(path);
        format!("[@{name}]({uri})")
    } else if uri.starts_with("zed://") {
        let name = uri.split('/').next_back().unwrap_or(&uri);
        format!("[@{name}]({uri})")
    } else {
        uri
    }
}

fn extract_tool_call_content_from_changes(
    changes: HashMap<PathBuf, FileChange>,
) -> (
    String,
    Vec<ToolCallLocation>,
    impl Iterator<Item = ToolCallContent>,
) {
    let changes = changes.into_iter().collect_vec();
    let title = if changes.is_empty() {
        "Edit".to_string()
    } else {
        format!(
            "Edit {}",
            changes
                .iter()
                .map(|(path, change)| tool_call_location_for_change(path, change)
                    .display()
                    .to_string())
                .join(", ")
        )
    };
    let locations = changes
        .iter()
        .map(|(path, change)| ToolCallLocation::new(tool_call_location_for_change(path, change)))
        .collect_vec();
    let content = changes
        .into_iter()
        .flat_map(|(path, change)| extract_tool_call_content_from_change(path, change));

    (title, locations, content)
}

fn tool_call_location_for_change(path: &Path, change: &FileChange) -> PathBuf {
    match change {
        FileChange::Update {
            move_path: Some(move_path),
            ..
        } => move_path.clone(),
        _ => path.to_path_buf(),
    }
}

fn extract_tool_call_content_from_change(
    path: PathBuf,
    change: FileChange,
) -> Vec<ToolCallContent> {
    match change {
        FileChange::Add { content } => vec![ToolCallContent::Diff(Diff::new(path, content))],
        FileChange::Delete { content } => {
            vec![ToolCallContent::Diff(
                Diff::new(path, String::new()).old_text(content),
            )]
        }
        FileChange::Update {
            unified_diff,
            move_path,
        } => extract_tool_call_content_from_unified_diff(move_path.unwrap_or(path), unified_diff),
    }
}

fn extract_tool_call_content_from_unified_diff(
    path: PathBuf,
    unified_diff: String,
) -> Vec<ToolCallContent> {
    let Ok(patch) = diffy::Patch::from_str(&unified_diff) else {
        return vec![ToolCallContent::Content(Content::new(ContentBlock::Text(
            TextContent::new(unified_diff),
        )))];
    };

    let diffs = patch
        .hunks()
        .iter()
        .map(|hunk| {
            let mut old_text = String::new();
            let mut new_text = String::new();

            for line in hunk.lines() {
                match line {
                    diffy::Line::Context(text) => {
                        old_text.push_str(text);
                        new_text.push_str(text);
                    }
                    diffy::Line::Delete(text) => old_text.push_str(text),
                    diffy::Line::Insert(text) => new_text.push_str(text),
                }
            }

            ToolCallContent::Diff(Diff::new(path.clone(), new_text).old_text(old_text))
        })
        .collect_vec();

    if diffs.is_empty() {
        vec![ToolCallContent::Content(Content::new(ContentBlock::Text(
            TextContent::new(unified_diff),
        )))]
    } else {
        diffs
    }
}

fn guardian_assessment_tool_call_id(id: &str) -> String {
    format!("guardian_assessment:{id}")
}

fn guardian_assessment_tool_call_status(status: &GuardianAssessmentStatus) -> ToolCallStatus {
    match status {
        GuardianAssessmentStatus::InProgress => ToolCallStatus::InProgress,
        GuardianAssessmentStatus::Approved => ToolCallStatus::Completed,
        GuardianAssessmentStatus::Denied
        | GuardianAssessmentStatus::Aborted
        | GuardianAssessmentStatus::TimedOut => ToolCallStatus::Failed,
    }
}

fn guardian_assessment_content(event: &GuardianAssessmentEvent) -> Vec<ToolCallContent> {
    let mut lines = vec![format!(
        "Status: {}",
        match event.status {
            GuardianAssessmentStatus::InProgress => "In progress",
            GuardianAssessmentStatus::Approved => "Approved",
            GuardianAssessmentStatus::Denied => "Denied",
            GuardianAssessmentStatus::Aborted => "Aborted",
            GuardianAssessmentStatus::TimedOut => "Timed out",
        }
    )];

    if let Some(summary) = guardian_action_summary(&event.action) {
        lines.push(format!("Action: {summary}"));
    }

    if let Some(level) = event.risk_level {
        lines.push(format!("Risk: {}", format!("{level:?}").to_lowercase()));
    }

    if let Some(rationale) = event.rationale.as_ref()
        && !rationale.trim().is_empty()
    {
        lines.push(format!("Rationale: {rationale}"));
    }

    let mut content = vec![ToolCallContent::Content(Content::new(ContentBlock::Text(
        TextContent::new(lines.join("\n")),
    )))];

    if guardian_action_summary(&event.action).is_none()
        && let Ok(action_json) = serde_json::to_string_pretty(&event.action)
    {
        content.push(ToolCallContent::Content(Content::new(ContentBlock::Text(
            TextContent::new(format!("Action payload:\n{action_json}")),
        ))));
    }

    content
}

fn guardian_action_summary(action: &GuardianAssessmentAction) -> Option<String> {
    match action {
        GuardianAssessmentAction::Command {
            source,
            command,
            cwd: _,
        } => {
            let label = guardian_command_source_label(source);
            Some(format!("{label} {command}"))
        }
        GuardianAssessmentAction::Execve {
            source,
            program,
            argv,
            cwd: _,
        } => {
            let label = guardian_command_source_label(source);
            let command: Vec<&str> = if argv.is_empty() {
                vec![program.as_str()]
            } else {
                argv.iter().map(String::as_str).collect()
            };
            let joined = shlex::try_join(command.iter().copied())
                .ok()
                .unwrap_or_else(|| command.join(" "));
            Some(format!("{label} {joined}"))
        }
        GuardianAssessmentAction::ApplyPatch { files, cwd: _ } => Some(if files.len() == 1 {
            format!("apply_patch touching {}", files[0].display())
        } else {
            format!("apply_patch touching {} files", files.len())
        }),
        GuardianAssessmentAction::NetworkAccess { target, host, .. } => {
            let label = if target.is_empty() { host } else { target };
            Some(format!("network access to {label}"))
        }
        GuardianAssessmentAction::McpToolCall {
            server,
            tool_name,
            connector_name,
            ..
        } => {
            let label = connector_name.as_deref().unwrap_or(server.as_str());
            Some(format!("MCP {tool_name} on {label}"))
        }
        GuardianAssessmentAction::RequestPermissions { reason, .. } => Some(
            reason
                .clone()
                .unwrap_or_else(|| "request additional permissions".to_string()),
        ),
    }
}

fn guardian_command_source_label(source: &GuardianCommandSource) -> &'static str {
    match source {
        GuardianCommandSource::Shell => "shell",
        GuardianCommandSource::UnifiedExec => "exec",
    }
}

/// Extract title and call_id from a WebSearchAction (used for replay)
fn web_search_action_to_title_and_id(
    id: &Option<String>,
    action: &codex_protocol::models::WebSearchAction,
) -> (String, String) {
    match action {
        codex_protocol::models::WebSearchAction::Search { query, queries } => {
            let title = queries
                .as_ref()
                .map(|q| q.join(", "))
                .or_else(|| query.clone())
                .unwrap_or_else(|| "Web search".to_string());
            let call_id = id
                .clone()
                .unwrap_or_else(|| generate_fallback_id("web_search"));
            (title, call_id)
        }
        codex_protocol::models::WebSearchAction::OpenPage { url } => {
            let title = url.clone().unwrap_or_else(|| "Open page".to_string());
            let call_id = id
                .clone()
                .unwrap_or_else(|| generate_fallback_id("web_open"));
            (title, call_id)
        }
        codex_protocol::models::WebSearchAction::FindInPage { pattern, .. } => {
            let title = pattern
                .clone()
                .unwrap_or_else(|| "Find in page".to_string());
            let call_id = id
                .clone()
                .unwrap_or_else(|| generate_fallback_id("web_find"));
            (title, call_id)
        }
        codex_protocol::models::WebSearchAction::Other => {
            ("Unknown".to_string(), generate_fallback_id("web_search"))
        }
    }
}

/// Generate a fallback ID using UUID (used when id is missing)
fn generate_fallback_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

fn hook_tool_call_id(hook_id: &str) -> String {
    format!("codex-hook-{hook_id}")
}

fn hook_title(run: &HookRunSummary) -> String {
    format!("Hook: {}", hook_event_name_label(run.event_name))
}

fn hook_meta(run: &HookRunSummary) -> Meta {
    Meta::from_iter([(
        ANYHARNESS_META_KEY.to_string(),
        json!({
            "nativeToolName": "CodexHook",
            "toolKind": "hook",
            "hookId": run.id,
            "hookStatus": hook_status_label(run.status),
        }),
    )])
}

fn hook_event_name_label(event_name: HookEventName) -> &'static str {
    match event_name {
        HookEventName::PreToolUse => "Pre Tool Use",
        HookEventName::PermissionRequest => "Permission Request",
        HookEventName::PostToolUse => "Post Tool Use",
        HookEventName::SessionStart => "Session Start",
        HookEventName::UserPromptSubmit => "User Prompt Submit",
        HookEventName::Stop => "Stop",
    }
}

fn hook_status_label(status: HookRunStatus) -> &'static str {
    match status {
        HookRunStatus::Running => "Running",
        HookRunStatus::Completed => "Completed",
        HookRunStatus::Failed => "Failed",
        HookRunStatus::Blocked => "Blocked",
        HookRunStatus::Stopped => "Stopped",
    }
}

fn hook_output_entry_kind_label(kind: HookOutputEntryKind) -> &'static str {
    match kind {
        HookOutputEntryKind::Warning => "Warning",
        HookOutputEntryKind::Stop => "Stop",
        HookOutputEntryKind::Feedback => "Feedback",
        HookOutputEntryKind::Context => "Context",
        HookOutputEntryKind::Error => "Error",
    }
}

fn hook_tool_content(run: &HookRunSummary) -> Vec<ToolCallContent> {
    let mut lines = Vec::new();
    if let Some(message) = run
        .status_message
        .as_deref()
        .map(str::trim)
        .filter(|message| !message.is_empty())
    {
        lines.push(message.to_string());
    }
    for entry in &run.entries {
        let text = entry.text.trim();
        if text.is_empty() {
            continue;
        }
        lines.push(format!(
            "{}: {text}",
            hook_output_entry_kind_label(entry.kind)
        ));
    }
    if lines.is_empty() {
        Vec::new()
    } else {
        vec![ToolCallContent::Content(Content::new(lines.join("\n")))]
    }
}

fn initial_collaboration_mask(
    models_manager: &dyn ModelsManagerImpl,
) -> Option<CollaborationModeMask> {
    let visible = models_manager
        .list_collaboration_modes()
        .into_iter()
        .filter(|mask| mask.mode.is_some_and(ModeKind::is_tui_visible))
        .collect::<Vec<_>>();

    visible
        .iter()
        .find(|mask| mask.mode == Some(ModeKind::Default))
        .cloned()
        .or_else(|| visible.into_iter().next())
}

fn collaboration_mode_value_id(kind: ModeKind) -> &'static str {
    match kind {
        ModeKind::Plan => "plan",
        ModeKind::Default | ModeKind::PairProgramming | ModeKind::Execute => "default",
    }
}

fn collaboration_mode_kind_from_id(value: &str) -> Option<ModeKind> {
    match value {
        "plan" => Some(ModeKind::Plan),
        "default" => Some(ModeKind::Default),
        _ => None,
    }
}

/// Checks if a prompt is slash command
fn extract_slash_command(content: &[UserInput]) -> Option<(&str, &str)> {
    let line = content.first().and_then(|block| match block {
        UserInput::Text { text, .. } => Some(text),
        _ => None,
    })?;

    parse_slash_name(line)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use agent_client_protocol::{RequestPermissionResponse, TextContent};
    use codex_core::{
        config::ConfigOverrides,
        test_support::{all_model_presets, builtin_collaboration_mode_presets},
    };
    use codex_protocol::config_types::ModeKind;
    use codex_protocol::protocol::{
        BackgroundEventEvent, DeprecationNoticeEvent, HookEventName, HookExecutionMode,
        HookHandlerType, HookOutputEntry, HookOutputEntryKind, HookScope, PlanDeltaEvent,
        RawResponseItemEvent,
    };
    use tokio::{
        sync::{Mutex, Notify, mpsc::UnboundedSender},
        task::LocalSet,
    };

    use super::*;

    fn request_user_input_capabilities() -> Arc<std::sync::Mutex<ClientCapabilities>> {
        codex_capabilities(json!({
            "requestUserInput": true,
        }))
    }

    fn mcp_elicitation_capabilities() -> Arc<std::sync::Mutex<ClientCapabilities>> {
        codex_capabilities(json!({
            "requestUserInput": true,
            "mcpElicitation": true,
        }))
    }

    fn codex_capabilities(codex: serde_json::Value) -> Arc<std::sync::Mutex<ClientCapabilities>> {
        Arc::new(std::sync::Mutex::new(
            ClientCapabilities::new().meta(Meta::from_iter([("codex".to_string(), codex)])),
        ))
    }

    fn ext_response(value: serde_json::Value) -> agent_client_protocol::ExtResponse {
        agent_client_protocol::ExtResponse::new(
            RawValue::from_string(value.to_string()).unwrap().into(),
        )
    }

    #[test]
    fn test_message_only_elicitation_schema_requires_sealed_empty_object() {
        assert!(is_message_only_elicitation_schema(&json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        })));

        for schema in [
            json!({
                "type": "object",
                "properties": {},
            }),
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
                "required": ["name"],
            }),
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
                "patternProperties": {
                    "^x-": { "type": "string" },
                },
            }),
        ] {
            assert!(!is_message_only_elicitation_schema(&schema));
        }
    }

    fn request_user_input_event() -> RequestUserInputEvent {
        RequestUserInputEvent {
            call_id: "call-1".to_string(),
            turn_id: "turn-1".to_string(),
            questions: vec![RequestUserInputQuestion {
                id: "provider".to_string(),
                header: "Provider".to_string(),
                question: "Which provider should be used?".to_string(),
                is_other: true,
                is_secret: false,
                options: Some(vec![RequestUserInputQuestionOption {
                    label: "Recommended".to_string(),
                    description: "Use the recommended provider".to_string(),
                }]),
            }],
        }
    }

    fn prompt_state_with_stub_client(
        client: Arc<StubClient>,
    ) -> (SessionClient, PromptState, Arc<StubCodexThread>) {
        let session_client =
            SessionClient::with_client(SessionId::new("test"), client, Arc::default());
        let thread = Arc::new(StubCodexThread::new());
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
        let prompt_state = PromptState::new(
            "submission-id".to_string(),
            thread.clone(),
            message_tx,
            response_tx,
        );
        (session_client, prompt_state, thread)
    }

    fn hook_run_summary(
        id: &str,
        status: HookRunStatus,
        status_message: Option<&str>,
        entries: Vec<HookOutputEntry>,
    ) -> HookRunSummary {
        HookRunSummary {
            id: id.to_string(),
            event_name: HookEventName::PreToolUse,
            handler_type: HookHandlerType::Command,
            execution_mode: HookExecutionMode::Sync,
            scope: HookScope::Turn,
            source_path: PathBuf::from("/tmp/hook.json").try_into().unwrap(),
            source: Default::default(),
            display_order: 0,
            status,
            status_message: status_message.map(str::to_string),
            started_at: 1,
            completed_at: None,
            duration_ms: None,
            entries,
        }
    }

    #[tokio::test]
    async fn test_prompt() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["Hi".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Hi"
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_streamed_agent_message_completion_marker_reuses_message_id() -> anyhow::Result<()>
    {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::new());
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state =
                    PromptState::new("submission-id".to_string(), thread, message_tx, response_tx);

                for delta in ["Hel", "lo"] {
                    prompt_state
                        .handle_event(
                            &session_client,
                            EventMsg::AgentMessageContentDelta(AgentMessageContentDeltaEvent {
                                thread_id: "thread-id".to_string(),
                                turn_id: "turn-id".to_string(),
                                item_id: "item-1".to_string(),
                                delta: delta.to_string(),
                            }),
                        )
                        .await;
                }

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::ItemCompleted(ItemCompletedEvent {
                            thread_id: codex_protocol::ThreadId::new(),
                            turn_id: "turn-id".to_string(),
                            item: TurnItem::AgentMessage(codex_protocol::items::AgentMessageItem {
                                id: "item-1".to_string(),
                                content: vec![],
                                phase: None,
                                memory_citation: None,
                            }),
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 3);

                let first_parts = agent_message_chunk_parts(&notifications[0].update);
                assert_eq!(first_parts.0, "Hel");
                assert!(first_parts.2.is_none());
                let first_message_id = first_parts.1.to_string();
                Uuid::parse_str(&first_message_id)?;

                let second_parts = agent_message_chunk_parts(&notifications[1].update);
                assert_eq!(second_parts.0, "lo");
                assert_eq!(second_parts.1, first_message_id);
                assert!(second_parts.2.is_none());

                let completion_parts = agent_message_chunk_parts(&notifications[2].update);
                assert_eq!(completion_parts.0, "");
                assert_eq!(completion_parts.1, first_message_id);
                assert_eq!(
                    serde_json::to_value(completion_parts.2.expect("completion meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "assistant_message_completed",
                            "codexItemId": "item-1",
                        },
                    })
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_plan_delta_streams_as_proposed_plan_delta() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                for delta in ["- Step 1\n", "- Step 2\n"] {
                    prompt_state
                        .handle_event(
                            &session_client,
                            EventMsg::PlanDelta(PlanDeltaEvent {
                                thread_id: "thread-id".to_string(),
                                turn_id: "turn-id".to_string(),
                                item_id: "plan-1".to_string(),
                                delta: delta.to_string(),
                            }),
                        )
                        .await;
                }

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::ItemCompleted(ItemCompletedEvent {
                            thread_id: codex_protocol::ThreadId::new(),
                            turn_id: "turn-id".to_string(),
                            item: TurnItem::Plan(codex_protocol::items::PlanItem {
                                id: "plan-1".to_string(),
                                text: "- Step 1\n- Step 2\n".to_string(),
                            }),
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 3);

                let first_parts = agent_message_chunk_parts(&notifications[0].update);
                assert_eq!(first_parts.0, "- Step 1\n");
                let first_message_id = first_parts.1.to_string();
                Uuid::parse_str(&first_message_id)?;
                assert_eq!(
                    serde_json::to_value(first_parts.2.expect("plan delta meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "proposed_plan_delta",
                            "codexItemId": "plan-1",
                        },
                    })
                );

                let second_parts = agent_message_chunk_parts(&notifications[1].update);
                assert_eq!(second_parts.0, "- Step 2\n");
                assert_eq!(second_parts.1, first_message_id);
                assert_eq!(
                    serde_json::to_value(second_parts.2.expect("plan delta meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "proposed_plan_delta",
                            "codexItemId": "plan-1",
                        },
                    })
                );

                let completion_parts = agent_message_chunk_parts(&notifications[2].update);
                assert_eq!(completion_parts.0, "- Step 1\n- Step 2\n");
                Uuid::parse_str(completion_parts.1)?;
                assert_ne!(completion_parts.1, first_message_id);
                assert_eq!(
                    serde_json::to_value(completion_parts.2.expect("completion meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "proposed_plan_completed",
                            "codexItemId": "plan-1",
                            "sourceItemId": "plan-1",
                        },
                    })
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_completed_plan_item_emits_proposed_plan_completed() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::ItemCompleted(ItemCompletedEvent {
                            thread_id: codex_protocol::ThreadId::new(),
                            turn_id: "turn-id".to_string(),
                            item: TurnItem::Plan(codex_protocol::items::PlanItem {
                                id: "plan-1".to_string(),
                                text: "1. Investigate\n2. Fix\n".to_string(),
                            }),
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 1);

                let completion_parts = agent_message_chunk_parts(&notifications[0].update);
                assert_eq!(completion_parts.0, "1. Investigate\n2. Fix\n");
                Uuid::parse_str(completion_parts.1)?;
                assert_eq!(
                    serde_json::to_value(completion_parts.2.expect("completion meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "proposed_plan_completed",
                            "codexItemId": "plan-1",
                            "sourceItemId": "plan-1",
                        },
                    })
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_status_background_event_emits_transient_thought() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                for message in ["Working in the background", "Still checking hooks"] {
                    prompt_state
                        .handle_event(
                            &session_client,
                            EventMsg::BackgroundEvent(BackgroundEventEvent {
                                message: message.to_string(),
                            }),
                        )
                        .await;
                }

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 2);

                let first = agent_thought_chunk_parts(&notifications[0].update);
                assert_eq!(first.0, "Working in the background");
                Uuid::parse_str(first.1)?;
                assert_eq!(
                    serde_json::to_value(first.2.expect("transient meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "transient_status",
                        },
                    })
                );

                let second = agent_thought_chunk_parts(&notifications[1].update);
                assert_eq!(second.0, "Still checking hooks");
                assert_eq!(second.1, first.1);
                assert_eq!(
                    serde_json::to_value(second.2.expect("transient meta"))?,
                    json!({
                        "anyharness": {
                            "transcriptEvent": "transient_status",
                        },
                    })
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_status_hook_started_emits_visible_hook_tool_call() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::HookStarted(HookStartedEvent {
                            turn_id: Some("turn-1".to_string()),
                            run: hook_run_summary(
                                "hook-1",
                                HookRunStatus::Running,
                                Some("Checking command policy"),
                                Vec::new(),
                            ),
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 1);
                let SessionUpdate::ToolCall(tool_call) = &notifications[0].update else {
                    panic!("expected hook tool call, got {:?}", notifications[0].update);
                };
                assert_eq!(tool_call.tool_call_id.0.as_ref(), "codex-hook-hook-1");
                assert_eq!(tool_call.title, "Hook: Pre Tool Use");
                assert_eq!(tool_call.kind, ToolKind::Other);
                assert_eq!(tool_call.status, ToolCallStatus::InProgress);
                assert_eq!(
                    tool_call_content_text(&tool_call.content),
                    "Checking command policy"
                );
                assert_eq!(
                    serde_json::to_value(tool_call.meta.as_ref().expect("hook meta"))?,
                    json!({
                        "anyharness": {
                            "nativeToolName": "CodexHook",
                            "toolKind": "hook",
                            "hookId": "hook-1",
                            "hookStatus": "Running",
                        },
                    })
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_status_hook_completed_emits_terminal_hook_update() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::HookCompleted(HookCompletedEvent {
                            turn_id: Some("turn-1".to_string()),
                            run: hook_run_summary(
                                "hook-1",
                                HookRunStatus::Blocked,
                                Some("Hook blocked the action"),
                                vec![HookOutputEntry {
                                    kind: HookOutputEntryKind::Error,
                                    text: "Missing approval".to_string(),
                                }],
                            ),
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 1);
                let SessionUpdate::ToolCallUpdate(update) = &notifications[0].update else {
                    panic!(
                        "expected hook tool call update, got {:?}",
                        notifications[0].update
                    );
                };
                assert_eq!(update.tool_call_id.0.as_ref(), "codex-hook-hook-1");
                assert_eq!(update.fields.title.as_deref(), Some("Hook: Pre Tool Use"));
                assert_eq!(update.fields.kind, Some(ToolKind::Other));
                assert_eq!(update.fields.status, Some(ToolCallStatus::Failed));
                let content = update
                    .fields
                    .content
                    .as_deref()
                    .expect("hook update content");
                assert_eq!(
                    tool_call_content_text(content),
                    "Hook blocked the action\nError: Missing approval"
                );
                assert_eq!(
                    serde_json::to_value(update.meta.as_ref().expect("hook meta"))?,
                    json!({
                        "anyharness": {
                            "nativeToolName": "CodexHook",
                            "toolKind": "hook",
                            "hookId": "hook-1",
                            "hookStatus": "Blocked",
                        },
                    })
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_status_deprecation_notice_emits_warning_prose() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::DeprecationNotice(DeprecationNoticeEvent {
                            summary: "Legacy configuration is deprecated".to_string(),
                            details: Some("Use the new config key instead.".to_string()),
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert_eq!(notifications.len(), 1);
                let SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) = &notifications[0].update
                else {
                    panic!(
                        "expected deprecation warning prose, got {:?}",
                        notifications[0].update
                    );
                };
                assert!(text.contains("Warning: Legacy configuration is deprecated"));
                assert!(text.contains("Use the new config key instead."));

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_status_raw_response_item_remains_non_visible() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let client = Arc::new(StubClient::new());
                let (session_client, mut prompt_state, _thread) =
                    prompt_state_with_stub_client(client.clone());

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::RawResponseItem(RawResponseItemEvent {
                            item: ResponseItem::CustomToolCall {
                                id: None,
                                status: None,
                                call_id: "call-1".to_string(),
                                name: "raw-tool".to_string(),
                                input: "{}".to_string(),
                            },
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert!(notifications.is_empty());

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_request_user_input_submits_empty_answer() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::new());
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::RequestUserInput(RequestUserInputEvent {
                            call_id: "call-1".to_string(),
                            turn_id: "turn-1".to_string(),
                            questions: vec![
                                codex_protocol::request_user_input::RequestUserInputQuestion {
                                    id: "provider".to_string(),
                                    header: "Provider".to_string(),
                                    question: "Which provider should be used?".to_string(),
                                    is_other: true,
                                    is_secret: false,
                                    options: Some(vec![
                                        codex_protocol::request_user_input::RequestUserInputQuestionOption {
                                            label: "Recommended".to_string(),
                                            description: "Use the recommended provider".to_string(),
                                        },
                                    ]),
                                },
                            ],
                        }),
                    )
                    .await;

                let ops = thread.ops.lock().unwrap();
                assert_eq!(ops.len(), 1);
                let Op::UserInputAnswer { id, response } = &ops[0] else {
                    panic!("expected UserInputAnswer op, got {:?}", ops[0]);
                };
                assert_eq!(id, "turn-1");
                assert!(response.answers.is_empty());

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_request_user_input_uses_ext_method_when_supported() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_ext_responses(vec![ext_response(json!({
                    "outcome": "submitted",
                    "answers": [
                        {
                            "questionId": "provider",
                            "selectedOptionLabel": "Recommended",
                            "text": "custom answer"
                        }
                    ]
                }))]));
                let session_client = SessionClient::with_client(
                    session_id,
                    client.clone(),
                    request_user_input_capabilities(),
                );
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::RequestUserInput(request_user_input_event()),
                    )
                    .await;

                let ext_requests = client.ext_requests.lock().unwrap();
                assert_eq!(ext_requests.len(), 1);
                assert_eq!(
                    ext_requests[0].method.as_ref(),
                    CODEX_REQUEST_USER_INPUT_EXT_METHOD
                );
                let params: serde_json::Value = serde_json::from_str(ext_requests[0].params.get())?;
                assert_eq!(params["callId"], "call-1");
                assert_eq!(params["turnId"], "turn-1");
                assert_eq!(params["questions"][0]["questionId"], "provider");
                drop(ext_requests);

                let ops = thread.ops.lock().unwrap();
                assert_eq!(ops.len(), 1);
                let Op::UserInputAnswer { id, response } = &ops[0] else {
                    panic!("expected UserInputAnswer op, got {:?}", ops[0]);
                };
                assert_eq!(id, "turn-1");
                assert_eq!(
                    response.answers.get("provider").unwrap().answers,
                    vec!["Recommended", "user_note: custom answer"]
                );

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_request_user_input_invalid_ext_response_falls_back_empty() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_ext_responses(vec![ext_response(json!(
                    null
                ))]));
                let session_client = SessionClient::with_client(
                    session_id,
                    client,
                    request_user_input_capabilities(),
                );
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::RequestUserInput(request_user_input_event()),
                    )
                    .await;

                let ops = thread.ops.lock().unwrap();
                assert_eq!(ops.len(), 1);
                let Op::UserInputAnswer { response, .. } = &ops[0] else {
                    panic!("expected UserInputAnswer op, got {:?}", ops[0]);
                };
                assert!(response.answers.is_empty());

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_config_options_include_collaboration_mode_control() -> anyhow::Result<()> {
        let (_session_id, _client, _thread, message_tx, local_set) = setup().await?;

        local_set
            .run_until(async move {
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();
                message_tx
                    .send(ThreadMessage::GetConfigOptions { response_tx })
                    .unwrap();

                let options = response_rx.await??;
                let collaboration_mode = options
                    .iter()
                    .find(|option| option.id.0.as_ref() == "collaboration_mode")
                    .expect("collaboration mode control");

                assert!(matches!(
                    collaboration_mode.category.as_ref(),
                    Some(SessionConfigOptionCategory::Other(category))
                        if category == "collaboration_mode"
                ));

                let agent_client_protocol::SessionConfigKind::Select(select) =
                    &collaboration_mode.kind
                else {
                    panic!("expected select config option");
                };
                assert_eq!(select.current_value.0.as_ref(), "default");

                let agent_client_protocol::SessionConfigSelectOptions::Ungrouped(values) =
                    &select.options
                else {
                    panic!("expected ungrouped collaboration options");
                };
                let values = values
                    .iter()
                    .map(|option| option.value.0.as_ref())
                    .collect::<Vec<_>>();
                assert!(values.contains(&"default"));
                assert!(values.contains(&"plan"));

                let fast_mode = options
                    .iter()
                    .find(|option| option.id.0.as_ref() == "fast_mode")
                    .expect("fast mode control");
                assert_eq!(fast_mode.name, "Fast Mode");

                anyhow::Ok(())
            })
            .await
    }

    #[tokio::test]
    async fn test_compact() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/compact".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Compact task completed"
        ));
        let ops = thread.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::Compact]);

        Ok(())
    }

    #[tokio::test]
    async fn test_undo() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/undo".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(
            notifications.len(),
            2,
            "notifications don't match {notifications:?}"
        );
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Undo in progress..."
        ));
        assert!(matches!(
            &notifications[1].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "Undo completed."
        ));

        let ops = thread.ops.lock().unwrap();
        assert_eq!(ops.as_slice(), &[Op::Undo]);

        Ok(())
    }

    #[tokio::test]
    async fn test_init() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/init".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }), ..
                }) if text == INIT_COMMAND_PROMPT // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );
        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::UserInput {
                items: vec![UserInput::Text {
                    text: INIT_COMMAND_PROMPT.to_string(),
                    text_elements: vec![]
                }],
                environments: None,
                final_output_json_schema: None,
                responsesapi_client_metadata: None,
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_review() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/review".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "current changes" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::UncommittedChanges)),
                    target: ReviewTarget::UncommittedChanges,
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_custom_review() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
        let instructions = "Review what we did in agents.md";

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(
                session_id.clone(),
                vec![format!("/review {instructions}").into()],
            ),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Review what we did in agents.md" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::Custom {
                        instructions: instructions.to_owned()
                    })),
                    target: ReviewTarget::Custom {
                        instructions: instructions.to_owned()
                    },
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_commit_review() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/review-commit 123456".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "commit 123456" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::Commit {
                        sha: "123456".to_owned(),
                        title: None
                    })),
                    target: ReviewTarget::Commit {
                        sha: "123456".to_owned(),
                        title: None
                    },
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_branch_review() -> anyhow::Result<()> {
        let (session_id, client, thread, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/review-branch feature".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "changes against 'feature'" // we echo the prompt
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::Review {
                review_request: ReviewRequest {
                    user_facing_hint: Some(user_facing_hint(&ReviewTarget::BaseBranch {
                        branch: "feature".to_owned()
                    })),
                    target: ReviewTarget::BaseBranch {
                        branch: "feature".to_owned()
                    },
                }
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_custom_prompts() -> anyhow::Result<()> {
        let custom_prompts = vec![CustomPrompt {
            name: "custom".to_string(),
            path: "/tmp/custom.md".into(),
            content: "Custom prompt with $1 arg.".into(),
            description: None,
            argument_hint: None,
        }];
        let (session_id, client, thread, message_tx, local_set) =
            setup_with_custom_prompts(custom_prompts).await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["/custom foo".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(
            matches!(
                &notifications[0].update,
                SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) if text == "Custom prompt with foo arg."
            ),
            "notifications don't match {notifications:?}"
        );

        let ops = thread.ops.lock().unwrap();
        assert_eq!(
            ops.as_slice(),
            &[Op::UserInput {
                items: vec![UserInput::Text {
                    text: "Custom prompt with foo arg.".into(),
                    text_elements: vec![]
                }],
                environments: None,
                final_output_json_schema: None,
                responsesapi_client_metadata: None,
            }],
            "ops don't match {ops:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_load_lists_custom_prompts_as_available_commands() -> anyhow::Result<()> {
        let codex_home = unique_temp_dir("custom-prompts");
        let prompts_dir = codex_home.join("prompts");
        std::fs::create_dir_all(&prompts_dir)?;
        std::fs::write(
            prompts_dir.join("custom.md"),
            "---\ndescription: Run my custom prompt\nargument-hint: argument\n---\nCustom prompt with $1 arg.",
        )?;
        let config =
            Config::load_default_with_cli_overrides_for_codex_home(codex_home.clone(), vec![])
                .await?;
        let (_session_id, client, thread, message_tx, local_set) =
            setup_with_config_and_custom_prompts(config, Vec::new()).await?;
        let (load_response_tx, load_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Load {
            response_tx: load_response_tx,
        })?;

        tokio::try_join!(
            async {
                load_response_rx.await??;
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let ops = thread.ops.lock().unwrap();
        assert!(ops.is_empty());
        let notifications = client.notifications.lock().unwrap();
        assert!(notifications.iter().any(|notification| {
            matches!(
                &notification.update,
                SessionUpdate::AvailableCommandsUpdate(update)
                    if update
                        .available_commands
                        .iter()
                        .any(|command| command.name == "custom")
            )
        }));

        drop(std::fs::remove_dir_all(codex_home));
        Ok(())
    }

    #[tokio::test]
    async fn test_delta_deduplication() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["test delta".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        // We should only get ONE notification, not duplicates from both delta and non-delta
        let notifications = client.notifications.lock().unwrap();
        assert_eq!(
            notifications.len(),
            1,
            "Should only receive delta event, not duplicate non-delta. Got: {notifications:?}"
        );
        assert!(matches!(
            &notifications[0].update,
            SessionUpdate::AgentMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent { text, .. }),
                ..
            }) if text == "test delta"
        ));

        Ok(())
    }

    async fn setup() -> anyhow::Result<(
        SessionId,
        Arc<StubClient>,
        Arc<StubCodexThread>,
        UnboundedSender<ThreadMessage>,
        LocalSet,
    )> {
        setup_with_custom_prompts(Vec::new()).await
    }

    async fn setup_with_custom_prompts(
        custom_prompts: Vec<CustomPrompt>,
    ) -> anyhow::Result<(
        SessionId,
        Arc<StubClient>,
        Arc<StubCodexThread>,
        UnboundedSender<ThreadMessage>,
        LocalSet,
    )> {
        let config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        setup_with_config_and_custom_prompts(config, custom_prompts).await
    }

    async fn setup_with_config_and_custom_prompts(
        config: Config,
        custom_prompts: Vec<CustomPrompt>,
    ) -> anyhow::Result<(
        SessionId,
        Arc<StubClient>,
        Arc<StubCodexThread>,
        UnboundedSender<ThreadMessage>,
        LocalSet,
    )> {
        let session_id = SessionId::new("test");
        let client = Arc::new(StubClient::new());
        let session_client =
            SessionClient::with_client(session_id.clone(), client.clone(), Arc::default());
        let conversation = Arc::new(StubCodexThread::new());
        let models_manager = Arc::new(StubModelsManager);
        let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();
        let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut actor = ThreadActor::new(
            StubAuth,
            session_client,
            conversation.clone(),
            models_manager,
            config,
            message_rx,
            resolution_tx,
            resolution_rx,
        );
        actor.custom_prompts = Rc::new(RefCell::new(custom_prompts));

        let local_set = LocalSet::new();
        local_set.spawn_local(actor.spawn());
        Ok((session_id, client, conversation, message_tx, local_set))
    }

    fn unique_temp_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("codex-acp-{name}-{}", Uuid::new_v4()))
    }

    struct StubAuth;

    impl Auth for StubAuth {
        fn logout(&self) -> Result<bool, Error> {
            Ok(true)
        }
    }

    struct StubModelsManager;

    #[async_trait::async_trait]
    impl ModelsManagerImpl for StubModelsManager {
        async fn get_model(&self, _model_id: &Option<String>) -> String {
            all_model_presets()[0].to_owned().id
        }

        async fn list_models(&self) -> Vec<ModelPreset> {
            all_model_presets().to_owned()
        }

        fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask> {
            builtin_collaboration_mode_presets()
        }
    }

    struct StubCodexThread {
        current_id: AtomicUsize,
        active_prompt_id: std::sync::Mutex<Option<String>>,
        ops: std::sync::Mutex<Vec<Op>>,
        op_tx: mpsc::UnboundedSender<Event>,
        op_rx: Mutex<mpsc::UnboundedReceiver<Event>>,
    }

    impl StubCodexThread {
        fn new() -> Self {
            let (op_tx, op_rx) = mpsc::unbounded_channel();
            StubCodexThread {
                current_id: AtomicUsize::new(0),
                active_prompt_id: std::sync::Mutex::default(),
                ops: std::sync::Mutex::default(),
                op_tx,
                op_rx: Mutex::new(op_rx),
            }
        }
    }

    #[async_trait::async_trait]
    impl CodexThreadImpl for StubCodexThread {
        async fn submit(&self, op: Op) -> Result<String, CodexErr> {
            let id = self
                .current_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            self.ops.lock().unwrap().push(op.clone());

            match op {
                Op::UserInput { items, .. } => {
                    *self.active_prompt_id.lock().unwrap() = Some(id.to_string());
                    let prompt = items
                        .into_iter()
                        .map(|i| match i {
                            UserInput::Text { text, .. } => text,
                            _ => unimplemented!(),
                        })
                        .join("\n");

                    if prompt == "parallel-exec" {
                        // Emit interleaved exec events: Begin A, Begin B, End A, End B
                        let turn_id = id.to_string();
                        let cwd: codex_utils_absolute_path::AbsolutePathBuf =
                            std::env::current_dir().unwrap().try_into().unwrap();
                        let send = |msg| {
                            self.op_tx
                                .send(Event {
                                    id: id.to_string(),
                                    msg,
                                })
                                .unwrap();
                        };
                        send(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                            call_id: "call-a".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "a".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo a".into(),
                            }],
                            source: Default::default(),
                            interaction_input: None,
                        }));
                        send(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                            call_id: "call-b".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "b".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo b".into(),
                            }],
                            source: Default::default(),
                            interaction_input: None,
                        }));
                        send(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                            call_id: "call-a".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "a".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![],
                            source: Default::default(),
                            interaction_input: None,
                            stdout: "a\n".into(),
                            stderr: String::new(),
                            aggregated_output: "a\n".into(),
                            exit_code: 0,
                            duration: std::time::Duration::from_millis(10),
                            formatted_output: "a\n".into(),
                            status: ExecCommandStatus::Completed,
                        }));
                        send(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                            call_id: "call-b".into(),
                            process_id: None,
                            turn_id: turn_id.clone(),
                            command: vec!["echo".into(), "b".into()],
                            cwd: cwd.clone(),
                            parsed_cmd: vec![],
                            source: Default::default(),
                            interaction_input: None,
                            stdout: "b\n".into(),
                            stderr: String::new(),
                            aggregated_output: "b\n".into(),
                            exit_code: 0,
                            duration: std::time::Duration::from_millis(10),
                            formatted_output: "b\n".into(),
                            status: ExecCommandStatus::Completed,
                        }));
                        send(EventMsg::TurnComplete(TurnCompleteEvent {
                            last_agent_message: None,
                            turn_id,
                            completed_at: None,
                            duration_ms: None,
                            time_to_first_token_ms: None,
                        }));
                    } else if prompt == "approval-block" {
                        self.op_tx
                            .send(Event {
                                id: id.to_string(),
                                msg: EventMsg::ExecApprovalRequest(ExecApprovalRequestEvent {
                                    call_id: "call-id".to_string(),
                                    approval_id: Some("approval-id".to_string()),
                                    turn_id: id.to_string(),
                                    command: vec!["echo".to_string(), "hi".to_string()],
                                    cwd: std::env::current_dir().unwrap().try_into().unwrap(),
                                    reason: None,
                                    network_approval_context: None,
                                    proposed_execpolicy_amendment: None,
                                    proposed_network_policy_amendments: None,
                                    additional_permissions: None,
                                    available_decisions: Some(vec![
                                        ReviewDecision::Approved,
                                        ReviewDecision::Abort,
                                    ]),
                                    parsed_cmd: vec![ParsedCommand::Unknown {
                                        cmd: "echo hi".to_string(),
                                    }],
                                }),
                            })
                            .unwrap();
                    } else {
                        self.op_tx
                            .send(Event {
                                id: id.to_string(),
                                msg: EventMsg::AgentMessageContentDelta(
                                    AgentMessageContentDeltaEvent {
                                        thread_id: id.to_string(),
                                        turn_id: id.to_string(),
                                        item_id: id.to_string(),
                                        delta: prompt.clone(),
                                    },
                                ),
                            })
                            .unwrap();
                        // Send non-delta event (should be deduplicated, but handled by deduplication)
                        self.op_tx
                            .send(Event {
                                id: id.to_string(),
                                msg: EventMsg::AgentMessage(AgentMessageEvent {
                                    message: prompt,
                                    phase: None,
                                    memory_citation: None,
                                }),
                            })
                            .unwrap();
                        self.op_tx
                            .send(Event {
                                id: id.to_string(),
                                msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                    last_agent_message: None,
                                    turn_id: id.to_string(),
                                    completed_at: None,
                                    duration_ms: None,
                                    time_to_first_token_ms: None,
                                }),
                            })
                            .unwrap();
                    }
                }
                Op::Compact => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnStarted(TurnStartedEvent {
                                model_context_window: None,
                                collaboration_mode_kind: ModeKind::default(),
                                turn_id: id.to_string(),
                                started_at: None,
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::AgentMessage(AgentMessageEvent {
                                message: "Compact task completed".to_string(),
                                phase: None,
                                memory_citation: None,
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
                                completed_at: None,
                                duration_ms: None,
                                time_to_first_token_ms: None,
                            }),
                        })
                        .unwrap();
                }
                Op::Undo => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::UndoStarted(
                                codex_protocol::protocol::UndoStartedEvent {
                                    message: Some("Undo in progress...".to_string()),
                                },
                            ),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::UndoCompleted(
                                codex_protocol::protocol::UndoCompletedEvent {
                                    success: true,
                                    message: Some("Undo completed.".to_string()),
                                },
                            ),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
                                completed_at: None,
                                duration_ms: None,
                                time_to_first_token_ms: None,
                            }),
                        })
                        .unwrap();
                }
                Op::Review { review_request } => {
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::EnteredReviewMode(review_request.clone()),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::ExitedReviewMode(ExitedReviewModeEvent {
                                review_output: Some(ReviewOutputEvent {
                                    findings: vec![],
                                    overall_correctness: String::new(),
                                    overall_explanation: review_request
                                        .user_facing_hint
                                        .clone()
                                        .unwrap_or_default(),
                                    overall_confidence_score: 1.,
                                }),
                            }),
                        })
                        .unwrap();
                    self.op_tx
                        .send(Event {
                            id: id.to_string(),
                            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                                last_agent_message: None,
                                turn_id: id.to_string(),
                                completed_at: None,
                                duration_ms: None,
                                time_to_first_token_ms: None,
                            }),
                        })
                        .unwrap();
                }
                Op::ExecApproval { .. }
                | Op::ResolveElicitation { .. }
                | Op::RequestPermissionsResponse { .. }
                | Op::UserInputAnswer { .. }
                | Op::PatchApproval { .. }
                | Op::Interrupt => {}
                Op::Shutdown => {
                    if let Some(active_prompt_id) = self.active_prompt_id.lock().unwrap().take() {
                        self.op_tx
                            .send(Event {
                                id: active_prompt_id.clone(),
                                msg: EventMsg::TurnAborted(TurnAbortedEvent {
                                    turn_id: Some(active_prompt_id),
                                    reason: codex_protocol::protocol::TurnAbortReason::Interrupted,
                                    completed_at: None,
                                    duration_ms: None,
                                }),
                            })
                            .unwrap();
                    }
                }
                _ => {
                    unimplemented!()
                }
            }
            Ok(id.to_string())
        }

        async fn next_event(&self) -> Result<Event, CodexErr> {
            let Some(event) = self.op_rx.lock().await.recv().await else {
                return Err(CodexErr::InternalAgentDied);
            };
            Ok(event)
        }
    }

    struct StubClient {
        notifications: std::sync::Mutex<Vec<SessionNotification>>,
        permission_requests: std::sync::Mutex<Vec<RequestPermissionRequest>>,
        permission_responses: std::sync::Mutex<VecDeque<RequestPermissionResponse>>,
        ext_requests: std::sync::Mutex<Vec<ExtRequest>>,
        ext_responses: std::sync::Mutex<VecDeque<agent_client_protocol::ExtResponse>>,
        block_permission_requests: Option<Arc<Notify>>,
    }

    impl StubClient {
        fn new() -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::default(),
                ext_requests: std::sync::Mutex::default(),
                ext_responses: std::sync::Mutex::default(),
                block_permission_requests: None,
            }
        }

        fn with_permission_responses(responses: Vec<RequestPermissionResponse>) -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::new(responses.into()),
                ext_requests: std::sync::Mutex::default(),
                ext_responses: std::sync::Mutex::default(),
                block_permission_requests: None,
            }
        }

        fn with_ext_responses(responses: Vec<agent_client_protocol::ExtResponse>) -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::default(),
                ext_requests: std::sync::Mutex::default(),
                ext_responses: std::sync::Mutex::new(responses.into()),
                block_permission_requests: None,
            }
        }

        fn with_blocked_permission_requests(
            responses: Vec<RequestPermissionResponse>,
            notify: Arc<Notify>,
        ) -> Self {
            StubClient {
                notifications: std::sync::Mutex::default(),
                permission_requests: std::sync::Mutex::default(),
                permission_responses: std::sync::Mutex::new(responses.into()),
                ext_requests: std::sync::Mutex::default(),
                ext_responses: std::sync::Mutex::default(),
                block_permission_requests: Some(notify),
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Client for StubClient {
        async fn request_permission(
            &self,
            args: RequestPermissionRequest,
        ) -> Result<RequestPermissionResponse, Error> {
            self.permission_requests.lock().unwrap().push(args);
            if let Some(notify) = &self.block_permission_requests {
                notify.notified().await;
            }
            Ok(self
                .permission_responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled)
                }))
        }

        async fn session_notification(&self, args: SessionNotification) -> Result<(), Error> {
            self.notifications.lock().unwrap().push(args);
            Ok(())
        }

        async fn ext_method(
            &self,
            args: ExtRequest,
        ) -> Result<agent_client_protocol::ExtResponse, Error> {
            self.ext_requests.lock().unwrap().push(args);
            Ok(self
                .ext_responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    agent_client_protocol::ExtResponse::new(RawValue::NULL.to_owned().into())
                }))
        }
    }

    fn agent_message_chunk_parts(update: &SessionUpdate) -> (&str, &str, Option<&Meta>) {
        let SessionUpdate::AgentMessageChunk(ContentChunk {
            content: ContentBlock::Text(TextContent { text, .. }),
            message_id: Some(message_id),
            meta,
            ..
        }) = update
        else {
            panic!("expected agent message chunk with message id, got {update:?}");
        };
        (text, message_id, meta.as_ref())
    }

    fn agent_thought_chunk_parts(update: &SessionUpdate) -> (&str, &str, Option<&Meta>) {
        let SessionUpdate::AgentThoughtChunk(ContentChunk {
            content: ContentBlock::Text(TextContent { text, .. }),
            message_id: Some(message_id),
            meta,
            ..
        }) = update
        else {
            panic!("expected agent thought chunk with message id, got {update:?}");
        };
        (text, message_id, meta.as_ref())
    }

    fn tool_call_content_text(content: &[ToolCallContent]) -> String {
        content
            .iter()
            .filter_map(|content| match content {
                ToolCallContent::Content(Content {
                    content: ContentBlock::Text(TextContent { text, .. }),
                    ..
                }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn test_parallel_exec_commands() -> anyhow::Result<()> {
        let (session_id, client, _, message_tx, local_set) = setup().await?;
        let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();

        message_tx.send(ThreadMessage::Prompt {
            request: PromptRequest::new(session_id.clone(), vec!["parallel-exec".into()]),
            response_tx: prompt_response_tx,
        })?;

        tokio::try_join!(
            async {
                let stop_reason = prompt_response_rx.await??.await??;
                assert_eq!(stop_reason, StopReason::EndTurn);
                drop(message_tx);
                anyhow::Ok(())
            },
            async {
                local_set.await;
                anyhow::Ok(())
            }
        )?;

        let notifications = client.notifications.lock().unwrap();

        // Collect all ToolCall (begin) notifications keyed by their tool_call_id prefix.
        let tool_calls: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();

        // Collect all ToolCallUpdate notifications that carry a terminal status.
        let completed_updates: Vec<_> = notifications
            .iter()
            .filter_map(|n| match &n.update {
                SessionUpdate::ToolCallUpdate(update) => {
                    if update.fields.status == Some(ToolCallStatus::Completed) {
                        Some(update.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        // Both commands A and B should have produced a ToolCall (begin).
        assert_eq!(
            tool_calls.len(),
            2,
            "expected 2 ToolCall begin notifications, got {tool_calls:?}"
        );

        // Both commands A and B should have produced a completed ToolCallUpdate.
        assert_eq!(
            completed_updates.len(),
            2,
            "expected 2 completed ToolCallUpdate notifications, got {completed_updates:?}"
        );

        // The completed updates should reference the same tool_call_ids as the begins.
        let begin_ids: std::collections::HashSet<_> = tool_calls
            .iter()
            .map(|tc| tc.tool_call_id.clone())
            .collect();
        let end_ids: std::collections::HashSet<_> = completed_updates
            .iter()
            .map(|u| u.tool_call_id.clone())
            .collect();
        assert_eq!(
            begin_ids, end_ids,
            "completed update tool_call_ids should match begin tool_call_ids"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_exec_approval_uses_available_decisions() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new("denied"),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, mut message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .exec_approval(
                        &session_client,
                        ExecApprovalRequestEvent {
                            call_id: "call-id".to_string(),
                            approval_id: Some("approval-id".to_string()),
                            turn_id: "turn-id".to_string(),
                            command: vec!["echo".to_string(), "hi".to_string()],
                            cwd: std::env::current_dir()?.try_into()?,
                            reason: None,
                            network_approval_context: None,
                            proposed_execpolicy_amendment: None,
                            proposed_network_policy_amendments: None,
                            additional_permissions: None,
                            available_decisions: Some(vec![
                                ReviewDecision::Approved,
                                ReviewDecision::Denied,
                            ]),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo hi".to_string(),
                            }],
                        },
                    )
                    .await?;

                let ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key,
                    response,
                } = message_rx.recv().await.unwrap()
                else {
                    panic!("expected permission resolution message");
                };
                assert_eq!(submission_id, "submission-id");
                prompt_state
                    .handle_permission_request_resolved(&session_client, request_key, response)
                    .await?;

                let requests = client.permission_requests.lock().unwrap();
                let request = requests.last().unwrap();
                let option_ids = request
                    .options
                    .iter()
                    .map(|option| option.option_id.0.to_string())
                    .collect::<Vec<_>>();
                assert_eq!(option_ids, vec!["approved", "denied"]);

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ExecApproval {
                        id,
                        turn_id,
                        decision: ReviewDecision::Denied,
                    }) if id == "approval-id" && turn_id.as_deref() == Some("turn-id")
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_request_permissions_strict_auto_review_option() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new(
                            REQUEST_PERMISSIONS_ALLOW_TURN_STRICT_OPTION_ID,
                        ),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, mut message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .request_permissions(
                        &session_client,
                        RequestPermissionsEvent {
                            call_id: "permissions-call-id".to_string(),
                            turn_id: "turn-id".to_string(),
                            reason: Some("Need network access".to_string()),
                            permissions: RequestPermissionProfile {
                                network: Some(codex_protocol::models::NetworkPermissions {
                                    enabled: Some(true),
                                }),
                                file_system: None,
                            },
                            cwd: None,
                        },
                    )
                    .await?;

                let ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key,
                    response,
                } = message_rx.recv().await.unwrap()
                else {
                    panic!("expected permission resolution message");
                };
                assert_eq!(submission_id, "submission-id");

                {
                    let requests = client.permission_requests.lock().unwrap();
                    let request = requests.last().unwrap();
                    assert_eq!(
                        request
                            .options
                            .iter()
                            .map(|option| option.option_id.0.to_string())
                            .collect::<Vec<_>>(),
                        vec![
                            REQUEST_PERMISSIONS_ALLOW_SESSION_OPTION_ID.to_string(),
                            REQUEST_PERMISSIONS_ALLOW_TURN_OPTION_ID.to_string(),
                            REQUEST_PERMISSIONS_ALLOW_TURN_STRICT_OPTION_ID.to_string(),
                            REQUEST_PERMISSIONS_DENY_OPTION_ID.to_string(),
                        ]
                    );
                }

                prompt_state
                    .handle_permission_request_resolved(&session_client, request_key, response)
                    .await?;

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::RequestPermissionsResponse { id, response })
                        if id == "permissions-call-id"
                            && response.scope == PermissionGrantScope::Turn
                            && response.strict_auto_review
                            && response.permissions.network.as_ref().and_then(|network| network.enabled) == Some(true)
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_tool_approval_elicitation_routes_to_permission_request() -> anyhow::Result<()>
    {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new(MCP_TOOL_APPROVAL_ALLOW_SESSION_OPTION_ID),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, mut message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                let request_id = format!("{MCP_TOOL_APPROVAL_REQUEST_ID_PREFIX}call-123");
                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String(request_id.clone()),
                            request: ElicitationRequest::Form {
                                meta: Some(serde_json::json!({
                                    "codex_approval_kind": "mcp_tool_call",
                                    "persist": ["session", "always"],
                                    "connector_name": "Docs",
                                    "tool_title": "search_docs",
                                    "tool_description": "Search project documentation",
                                    "tool_params_display": [
                                        {
                                            "display_name": "Query",
                                            "name": "query",
                                            "value": "approval flow"
                                        }
                                    ]
                                })),
                                message: "Allow Docs to run tool \"search_docs\"?".to_string(),
                                requested_schema: serde_json::json!({
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": false,
                                }),
                            },
                        },
                    )
                    .await?;

                let ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key,
                    response,
                } = message_rx.recv().await.unwrap()
                else {
                    panic!("expected permission resolution message");
                };
                assert_eq!(submission_id, "submission-id");

                {
                    let requests = client.permission_requests.lock().unwrap();
                    let request = requests.last().unwrap();
                    assert_eq!(request.tool_call.tool_call_id.0.as_ref(), "call-123");
                    assert_eq!(
                        request
                            .options
                            .iter()
                            .map(|option| option.option_id.0.to_string())
                            .collect::<Vec<_>>(),
                        vec![
                            MCP_TOOL_APPROVAL_ALLOW_OPTION_ID.to_string(),
                            MCP_TOOL_APPROVAL_ALLOW_SESSION_OPTION_ID.to_string(),
                            MCP_TOOL_APPROVAL_ALLOW_ALWAYS_OPTION_ID.to_string(),
                            MCP_TOOL_APPROVAL_CANCEL_OPTION_ID.to_string(),
                        ]
                    );
                }

                prompt_state
                    .handle_permission_request_resolved(&session_client, request_key, response)
                    .await?;

                let op = thread.ops.lock().unwrap().last().cloned().unwrap();
                match op {
                    Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(id),
                        decision,
                        content,
                        meta,
                    } => {
                        assert_eq!(server_name, "test-server");
                        assert_eq!(id, request_id);
                        assert_eq!(decision, ElicitationAction::Accept);
                        assert!(content.is_none());
                        assert_eq!(
                            meta.as_ref()
                                .and_then(|value| value.get("persist"))
                                .and_then(serde_json::Value::as_str),
                            Some(MCP_TOOL_APPROVAL_PERSIST_SESSION)
                        );
                    }
                    other => panic!("unexpected op: {other:?}"),
                }

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_declines_unsupported_form_requests() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new("decline"),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Need some structured input".to_string(),
                                requested_schema: serde_json::json!({
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" }
                                    }
                                }),
                            },
                        },
                    )
                    .await?;

                let requests = client.permission_requests.lock().unwrap();
                assert!(
                    requests.is_empty(),
                    "unsupported MCP elicitations should be auto-declined"
                );

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                    }) if server_name == "test-server" && request_id == "request-id"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_declines_null_form_schema() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::new());
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Need unspecified input".to_string(),
                                requested_schema: serde_json::Value::Null,
                            },
                        },
                    )
                    .await?;

                let requests = client.permission_requests.lock().unwrap();
                assert!(
                    requests.is_empty(),
                    "null MCP elicitation schema should not be treated as message-only"
                );

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                    }) if server_name == "test-server" && request_id == "request-id"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_declines_unsealed_empty_object_schema() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::new());
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Need maybe-empty input".to_string(),
                                requested_schema: serde_json::json!({
                                    "type": "object",
                                    "properties": {},
                                }),
                            },
                        },
                    )
                    .await?;

                let requests = client.permission_requests.lock().unwrap();
                assert!(
                    requests.is_empty(),
                    "open object schemas should not be treated as message-only"
                );

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                    }) if server_name == "test-server" && request_id == "request-id"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_routes_message_only_form_through_permission() -> anyhow::Result<()>
    {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_permission_responses(vec![
                    RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                        SelectedPermissionOutcome::new(MCP_ELICITATION_DECLINE_OPTION_ID),
                    )),
                ]));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, mut message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Approve the message-only action".to_string(),
                                requested_schema: serde_json::json!({
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": false,
                                }),
                            },
                        },
                    )
                    .await?;

                let ThreadMessage::PermissionRequestResolved {
                    submission_id,
                    request_key,
                    response,
                } = message_rx.recv().await.unwrap()
                else {
                    panic!("expected permission resolution message");
                };
                assert_eq!(submission_id, "submission-id");

                {
                    let requests = client.permission_requests.lock().unwrap();
                    let request = requests.last().unwrap();
                    assert_eq!(
                        request.tool_call.fields.title.as_deref(),
                        Some("Approve MCP request")
                    );
                    assert_eq!(
                        request
                            .options
                            .iter()
                            .map(|option| option.option_id.0.to_string())
                            .collect::<Vec<_>>(),
                        vec![
                            MCP_ELICITATION_ACCEPT_OPTION_ID.to_string(),
                            MCP_ELICITATION_DECLINE_OPTION_ID.to_string(),
                            MCP_ELICITATION_CANCEL_OPTION_ID.to_string(),
                        ]
                    );
                }

                prompt_state
                    .handle_permission_request_resolved(&session_client, request_key, response)
                    .await?;

                let op = thread.ops.lock().unwrap().last().cloned().unwrap();
                assert!(matches!(
                    op,
                    Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                    } if server_name == "test-server" && request_id == "request-id"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_uses_ext_method_when_supported() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_ext_responses(vec![ext_response(json!({
                    "outcome": "accepted",
                    "content": {
                        "name": "docs"
                    }
                }))]));
                let session_client = SessionClient::with_client(
                    session_id,
                    client.clone(),
                    mcp_elicitation_capabilities(),
                );
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: Some(json!({ "debug": "live-only" })),
                                message: "Need some structured input".to_string(),
                                requested_schema: json!({
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" }
                                    }
                                }),
                            },
                        },
                    )
                    .await?;

                {
                    let ext_requests = client.ext_requests.lock().unwrap();
                    assert_eq!(ext_requests.len(), 1);
                    assert_eq!(
                        ext_requests[0].method.as_ref(),
                        CODEX_MCP_ELICITATION_EXT_METHOD
                    );
                    let params: serde_json::Value =
                        serde_json::from_str(ext_requests[0].params.get())?;
                    assert_eq!(params["serverName"], "test-server");
                    assert!(params.get("id").is_none());
                    assert_eq!(params["request"]["mode"], "form");
                    assert_eq!(params["request"]["message"], "Need some structured input");
                }

                let ops = thread.ops.lock().unwrap();
                assert_eq!(ops.len(), 1);
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Accept,
                        content: Some(content),
                        meta: None,
                    }) if server_name == "test-server"
                        && request_id == "request-id"
                        && content["name"] == "docs"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_unscoped_mcp_elicitation_routes_to_single_active_submission() -> anyhow::Result<()>
    {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_ext_responses(vec![ext_response(json!({
                    "outcome": "accepted",
                    "content": {
                        "name": "docs"
                    }
                }))]));
                let session_client = SessionClient::with_client(
                    session_id,
                    client.clone(),
                    mcp_elicitation_capabilities(),
                );
                let thread = Arc::new(StubCodexThread::new());
                let models_manager = Arc::new(StubModelsManager);
                let config = Config::load_with_cli_overrides_and_harness_overrides(
                    vec![],
                    ConfigOverrides::default(),
                )
                .await?;
                let (_message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();
                let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut actor = ThreadActor::new(
                    StubAuth,
                    session_client,
                    thread.clone(),
                    models_manager,
                    config,
                    message_rx,
                    resolution_tx.clone(),
                    resolution_rx,
                );
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                actor.submissions.insert(
                    "submission-id".to_string(),
                    SubmissionState::Prompt(PromptState::new(
                        "submission-id".to_string(),
                        thread.clone(),
                        resolution_tx,
                        response_tx,
                    )),
                );

                actor
                    .handle_event(Event {
                        id: "mcp_elicitation_request".to_string(),
                        msg: EventMsg::ElicitationRequest(ElicitationRequestEvent {
                            turn_id: None,
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Need some structured input".to_string(),
                                requested_schema: json!({
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" }
                                    }
                                }),
                            },
                        }),
                    })
                    .await;

                let ops = thread.ops.lock().unwrap();
                assert_eq!(ops.len(), 1);
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Accept,
                        content: Some(content),
                        meta: None,
                    }) if server_name == "test-server"
                        && request_id == "request-id"
                        && content["name"] == "docs"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_unscoped_mcp_elicitation_declines_when_ambiguous() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::new());
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let models_manager = Arc::new(StubModelsManager);
                let config = Config::load_with_cli_overrides_and_harness_overrides(
                    vec![],
                    ConfigOverrides::default(),
                )
                .await?;
                let (_message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();
                let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut actor = ThreadActor::new(
                    StubAuth,
                    session_client,
                    thread.clone(),
                    models_manager,
                    config,
                    message_rx,
                    resolution_tx.clone(),
                    resolution_rx,
                );
                let (response_tx_1, _response_rx_1) = tokio::sync::oneshot::channel();
                let (response_tx_2, _response_rx_2) = tokio::sync::oneshot::channel();
                actor.submissions.insert(
                    "submission-1".to_string(),
                    SubmissionState::Prompt(PromptState::new(
                        "submission-1".to_string(),
                        thread.clone(),
                        resolution_tx.clone(),
                        response_tx_1,
                    )),
                );
                actor.submissions.insert(
                    "submission-2".to_string(),
                    SubmissionState::Prompt(PromptState::new(
                        "submission-2".to_string(),
                        thread.clone(),
                        resolution_tx,
                        response_tx_2,
                    )),
                );

                actor
                    .handle_event(Event {
                        id: "mcp_elicitation_request".to_string(),
                        msg: EventMsg::ElicitationRequest(ElicitationRequestEvent {
                            turn_id: None,
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Form {
                                meta: None,
                                message: "Need some structured input".to_string(),
                                requested_schema: json!({
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" }
                                    }
                                }),
                            },
                        }),
                    })
                    .await;

                let ops = thread.ops.lock().unwrap();
                assert_eq!(ops.len(), 1);
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        server_name,
                        request_id: codex_protocol::mcp::RequestId::String(request_id),
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                    }) if server_name == "test-server" && request_id == "request-id"
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_invalid_ext_response_declines() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_ext_responses(vec![ext_response(json!(
                    null
                ))]));
                let session_client =
                    SessionClient::with_client(session_id, client, mcp_elicitation_capabilities());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Url {
                                meta: None,
                                message: "Authorize access".to_string(),
                                url: "https://example.com/authorize".to_string(),
                                elicitation_id: "elicitation-id".to_string(),
                            },
                        },
                    )
                    .await?;

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        decision: ElicitationAction::Decline,
                        content: None,
                        meta: None,
                        ..
                    })
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_mcp_elicitation_cancel_response_cancels() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_ext_responses(vec![ext_response(json!({
                    "outcome": "cancelled"
                }))]));
                let session_client =
                    SessionClient::with_client(session_id, client, mcp_elicitation_capabilities());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state = PromptState::new(
                    "submission-id".to_string(),
                    thread.clone(),
                    message_tx,
                    response_tx,
                );

                prompt_state
                    .mcp_elicitation(
                        &session_client,
                        ElicitationRequestEvent {
                            turn_id: Some("turn-id".to_string()),
                            server_name: "test-server".to_string(),
                            id: codex_protocol::mcp::RequestId::String("request-id".to_string()),
                            request: ElicitationRequest::Url {
                                meta: None,
                                message: "Authorize access".to_string(),
                                url: "https://example.com/authorize".to_string(),
                                elicitation_id: "elicitation-id".to_string(),
                            },
                        },
                    )
                    .await?;

                let ops = thread.ops.lock().unwrap();
                assert!(matches!(
                    ops.last(),
                    Some(Op::ResolveElicitation {
                        decision: ElicitationAction::Cancel,
                        content: None,
                        meta: None,
                        ..
                    })
                ));

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_blocked_approval_does_not_block_followup_events() -> anyhow::Result<()> {
        LocalSet::new()
            .run_until(async {
                let session_id = SessionId::new("test");
                let client = Arc::new(StubClient::with_blocked_permission_requests(
                    vec![],
                    Arc::new(Notify::new()),
                ));
                let session_client =
                    SessionClient::with_client(session_id, client.clone(), Arc::default());
                let thread = Arc::new(StubCodexThread::new());
                let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
                let (message_tx, _message_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut prompt_state =
                    PromptState::new("submission-id".to_string(), thread, message_tx, response_tx);

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::ExecApprovalRequest(ExecApprovalRequestEvent {
                            call_id: "call-id".to_string(),
                            approval_id: Some("approval-id".to_string()),
                            turn_id: "turn-id".to_string(),
                            command: vec!["echo".to_string(), "hi".to_string()],
                            cwd: std::env::current_dir()?.try_into()?,
                            reason: None,
                            network_approval_context: None,
                            proposed_execpolicy_amendment: None,
                            proposed_network_policy_amendments: None,
                            additional_permissions: None,
                            available_decisions: Some(vec![
                                ReviewDecision::Approved,
                                ReviewDecision::Abort,
                            ]),
                            parsed_cmd: vec![ParsedCommand::Unknown {
                                cmd: "echo hi".to_string(),
                            }],
                        }),
                    )
                    .await;

                prompt_state
                    .handle_event(
                        &session_client,
                        EventMsg::AgentMessage(AgentMessageEvent {
                            message: "still flowing".to_string(),
                            phase: None,
                            memory_citation: None,
                        }),
                    )
                    .await;

                let notifications = client.notifications.lock().unwrap();
                assert!(notifications.iter().any(|notification| {
                    matches!(
                        &notification.update,
                        SessionUpdate::AgentMessageChunk(ContentChunk {
                            content: ContentBlock::Text(TextContent { text, .. }),
                            ..
                        }) if text == "still flowing"
                    )
                }));

                drop(notifications);
                prompt_state.abort_pending_interactions();

                anyhow::Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_thread_shutdown_bypasses_blocked_permission_request() -> anyhow::Result<()> {
        let session_id = SessionId::new("test");
        let client = Arc::new(StubClient::with_blocked_permission_requests(
            vec![RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled,
            )],
            Arc::new(Notify::new()),
        ));
        let session_client =
            SessionClient::with_client(session_id.clone(), client.clone(), Arc::default());
        let conversation = Arc::new(StubCodexThread::new());
        let models_manager = Arc::new(StubModelsManager);
        let config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();
        let (resolution_tx, resolution_rx) = tokio::sync::mpsc::unbounded_channel();
        let actor = ThreadActor::new(
            StubAuth,
            session_client,
            conversation.clone(),
            models_manager,
            config,
            message_rx,
            resolution_tx,
            resolution_rx,
        );

        let local_set = LocalSet::new();
        let handle = local_set.spawn_local(actor.spawn());
        let thread = Thread {
            thread: conversation.clone(),
            message_tx,
            _handle: handle,
        };

        local_set
            .run_until(async move {
                let (prompt_response_tx, prompt_response_rx) = tokio::sync::oneshot::channel();
                thread.message_tx.send(ThreadMessage::Prompt {
                    request: PromptRequest::new(session_id, vec!["approval-block".into()]),
                    response_tx: prompt_response_tx,
                })?;
                let stop_reason_rx = prompt_response_rx.await??;

                tokio::time::timeout(Duration::from_millis(100), async {
                    loop {
                        if !client.permission_requests.lock().unwrap().is_empty() {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await?;

                tokio::time::timeout(Duration::from_millis(100), thread.shutdown()).await??;
                let stop_reason =
                    tokio::time::timeout(Duration::from_millis(100), stop_reason_rx).await??;
                assert_eq!(stop_reason?, StopReason::Cancelled);

                anyhow::Ok(())
            })
            .await?;

        let ops = conversation.ops.lock().unwrap();
        assert!(matches!(ops.last(), Some(Op::Shutdown)));

        Ok(())
    }
}
