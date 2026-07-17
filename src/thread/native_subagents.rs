use super::{ANYHARNESS_META_KEY, SessionClient};
use agent_client_protocol::schema::{
    Meta, ToolCall, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use codex_protocol::{
    ThreadId,
    items::{
        CollabAgentTool, CollabAgentToolCallItem, CollabAgentToolCallStatus, SubAgentActivityItem,
        TurnItem,
    },
    models::{FunctionCallOutputPayload, ResponseItem},
    protocol::{AgentStatus, EventMsg, ItemCompletedEvent, ItemStartedEvent, SubAgentActivityKind},
};
use itertools::Itertools;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

const MULTI_AGENT_V1_NAMESPACE: &str = "multi_agent_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeSubagentTool {
    Spawn,
    SendInput,
    Wait,
    Resume,
    Close,
}

impl NativeSubagentTool {
    fn from_v1_name(name: &str) -> Option<Self> {
        match name {
            "spawn_agent" => Some(Self::Spawn),
            "send_input" => Some(Self::SendInput),
            "wait_agent" => Some(Self::Wait),
            "resume_agent" => Some(Self::Resume),
            "close_agent" => Some(Self::Close),
            _ => None,
        }
    }

    fn from_collab_tool(tool: CollabAgentTool) -> Self {
        match tool {
            CollabAgentTool::SpawnAgent => Self::Spawn,
            CollabAgentTool::SendInput => Self::SendInput,
            CollabAgentTool::Wait => Self::Wait,
            CollabAgentTool::ResumeAgent => Self::Resume,
            CollabAgentTool::CloseAgent => Self::Close,
        }
    }

    fn native_name(self) -> &'static str {
        match self {
            Self::Spawn => "Agent",
            Self::SendInput => "send_input",
            Self::Wait => "wait",
            Self::Resume => "resume_agent",
            Self::Close => "close_agent",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Spawn => "Spawn subagent",
            Self::SendInput => "Message subagent",
            Self::Wait => "Wait for subagent",
            Self::Resume => "Resume subagent",
            Self::Close => "Close subagent",
        }
    }
}

#[derive(Clone, Debug)]
struct NativeSubagentOperation {
    tool: NativeSubagentTool,
    parent_tool_call_id: Option<String>,
    target_thread_ids: Vec<ThreadId>,
    emitted_status: Option<ToolCallStatus>,
}

/// Normalizes Codex collaboration events at the ACP boundary.
///
/// Codex persists two rollout generations: legacy function call/output pairs
/// and paginated `ItemCompleted` records. Live delivery additionally fans out
/// legacy `Collab*`/`SubAgentActivity` events from the structured turn items.
/// One session-scoped state machine consumes all of those shapes, so replay
/// and live delivery keep the same stable parent identity without duplicates.
#[derive(Default)]
pub(super) struct NativeSubagentState {
    parents_by_thread: HashMap<ThreadId, String>,
    agents_by_path: HashMap<String, (ThreadId, String)>,
    parent_tool_calls: HashSet<String>,
    parent_statuses: HashMap<String, ToolCallStatus>,
    operations: HashMap<String, NativeSubagentOperation>,
    seen_activities: HashSet<String>,
    suppressed_response_calls: HashSet<String>,
}

impl NativeSubagentState {
    pub(super) fn handle_event(&mut self, client: &SessionClient, event: &EventMsg) -> bool {
        match event {
            EventMsg::RawResponseItem(event) => self.handle_response_item(client, &event.item),
            EventMsg::ItemStarted(ItemStartedEvent { item, .. })
            | EventMsg::ItemCompleted(ItemCompletedEvent { item, .. }) => {
                self.handle_turn_item(client, item)
            }
            EventMsg::CollabAgentSpawnBegin(event) => {
                self.start_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Spawn,
                    Vec::new(),
                    None,
                    Some(json!({
                        "prompt": event.prompt,
                        "model": event.model,
                        "reasoningEffort": event.reasoning_effort,
                    })),
                );
                true
            }
            EventMsg::CollabAgentSpawnEnd(event) => {
                self.start_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Spawn,
                    Vec::new(),
                    None,
                    Some(json!({
                        "prompt": event.prompt,
                        "model": event.model,
                        "reasoningEffort": event.reasoning_effort,
                    })),
                );
                if let Some(thread_id) = event.new_thread_id {
                    self.parents_by_thread
                        .insert(thread_id, event.call_id.clone());
                }
                let status = event
                    .new_thread_id
                    .map(|_| native_agent_status(&event.status))
                    .unwrap_or(ToolCallStatus::Failed);
                let title = native_agent_title(
                    event.new_agent_nickname.as_deref(),
                    event.new_agent_role.as_deref(),
                );
                self.finish_operation(
                    client,
                    &event.call_id,
                    status,
                    Some(title),
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::CollabAgentInteractionBegin(event) => {
                self.start_thread_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::SendInput,
                    event.receiver_thread_id,
                    Some(json!({
                        "subagentId": event.receiver_thread_id,
                        "prompt": event.prompt,
                    })),
                );
                true
            }
            EventMsg::CollabAgentInteractionEnd(event) => {
                self.finish_agent_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::SendInput,
                    event.receiver_thread_id,
                    &event.status,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::CollabWaitingBegin(event) => {
                let targets = sorted_thread_ids(event.receiver_thread_ids.clone());
                let parent = self.parent_for_targets(&targets);
                self.start_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Wait,
                    targets,
                    parent,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::CollabWaitingEnd(event) => {
                let statuses = sorted_agent_statuses(&event.statuses);
                let targets = statuses
                    .iter()
                    .map(|(thread_id, _)| *thread_id)
                    .collect::<Vec<_>>();
                let parent = self.parent_for_targets(&targets);
                self.start_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Wait,
                    targets,
                    parent,
                    None,
                );
                let failed = statuses
                    .iter()
                    .any(|(_, status)| native_operation_status(status) == ToolCallStatus::Failed);
                self.finish_operation(
                    client,
                    &event.call_id,
                    if failed {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    },
                    None,
                    serde_json::to_value(event).ok(),
                );
                for (thread_id, status) in statuses {
                    self.update_parent_for_thread(client, thread_id, status);
                }
                true
            }
            EventMsg::CollabResumeBegin(event) => {
                self.start_thread_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Resume,
                    event.receiver_thread_id,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::CollabResumeEnd(event) => {
                self.finish_agent_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Resume,
                    event.receiver_thread_id,
                    &event.status,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::CollabCloseBegin(event) => {
                self.start_thread_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Close,
                    event.receiver_thread_id,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::CollabCloseEnd(event) => {
                self.finish_agent_operation(
                    client,
                    &event.call_id,
                    NativeSubagentTool::Close,
                    event.receiver_thread_id,
                    &event.status,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            EventMsg::SubAgentActivity(event) => {
                self.handle_activity(
                    client,
                    &event.event_id,
                    event.agent_thread_id,
                    event.agent_path.as_ref(),
                    event.kind,
                    serde_json::to_value(event).ok(),
                );
                true
            }
            _ => false,
        }
    }

    pub(super) fn handle_response_item(
        &mut self,
        client: &SessionClient,
        item: &ResponseItem,
    ) -> bool {
        match item {
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } if namespace.as_deref() == Some(MULTI_AGENT_V1_NAMESPACE) => {
                let Some(tool) = NativeSubagentTool::from_v1_name(name) else {
                    return false;
                };
                let raw_input = serde_json::from_str::<Value>(arguments).ok();
                let targets = raw_input
                    .as_ref()
                    .map(|value| native_targets_from_input(tool, value))
                    .unwrap_or_default();
                let parent = self.parent_for_targets(&targets);
                self.start_operation(client, call_id, tool, targets, parent, raw_input);
                true
            }
            ResponseItem::FunctionCall {
                name,
                namespace: None,
                call_id,
                ..
            } if matches!(
                name.as_str(),
                "spawn_agent" | "send_message" | "followup_task" | "interrupt_agent"
            ) =>
            {
                // Multi-agent v2 persists these raw calls as well as the
                // canonical SubAgentActivity item. Defer to that item so the
                // replay shape matches live delivery and does not double-log.
                self.suppressed_response_calls.insert(call_id.clone());
                true
            }
            ResponseItem::FunctionCall {
                name,
                namespace: None,
                arguments,
                call_id,
                ..
            } if name == "wait_agent" => {
                self.start_operation(
                    client,
                    call_id,
                    NativeSubagentTool::Wait,
                    Vec::new(),
                    None,
                    serde_json::from_str(arguments).ok(),
                );
                true
            }
            ResponseItem::FunctionCallOutput { call_id, .. }
                if self.suppressed_response_calls.contains(call_id) =>
            {
                true
            }
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } if self.operations.contains_key(call_id) => {
                self.handle_response_output(client, call_id, output);
                true
            }
            _ => false,
        }
    }

    fn handle_turn_item(&mut self, client: &SessionClient, item: &TurnItem) -> bool {
        match item {
            TurnItem::CollabAgentToolCall(item) => {
                self.handle_collab_item(client, item);
                true
            }
            TurnItem::SubAgentActivity(item) => {
                self.handle_activity_item(client, item);
                true
            }
            _ => false,
        }
    }

    fn handle_collab_item(&mut self, client: &SessionClient, item: &CollabAgentToolCallItem) {
        let tool = NativeSubagentTool::from_collab_tool(item.tool);
        let mut targets = item.receiver_thread_ids.clone();
        targets.extend(item.receiver_agents.iter().map(|agent| agent.thread_id));
        targets.extend(item.agents_states.keys().copied());
        let targets = sorted_thread_ids(targets);
        let parent = self.parent_for_targets(&targets);
        let raw = serde_json::to_value(item).ok();
        self.start_operation(client, &item.id, tool, targets.clone(), parent, raw.clone());

        if item.status == CollabAgentToolCallStatus::InProgress {
            return;
        }

        if tool == NativeSubagentTool::Spawn {
            for thread_id in &targets {
                self.parents_by_thread.insert(*thread_id, item.id.clone());
            }
            let status = if item.status == CollabAgentToolCallStatus::Failed || targets.is_empty() {
                ToolCallStatus::Failed
            } else {
                targets
                    .first()
                    .and_then(|thread_id| item.agents_states.get(thread_id))
                    .map(native_agent_status)
                    .unwrap_or(ToolCallStatus::InProgress)
            };
            let title = item
                .receiver_agents
                .first()
                .map(|agent| {
                    native_agent_title(agent.agent_nickname.as_deref(), agent.agent_role.as_deref())
                })
                .unwrap_or_else(|| "Subagent".to_string());
            self.finish_operation(client, &item.id, status, Some(title), raw);
            return;
        }

        let failed_agent = tool != NativeSubagentTool::Close
            && item
                .agents_states
                .values()
                .any(|status| native_operation_status(status) == ToolCallStatus::Failed);
        let status = if item.status == CollabAgentToolCallStatus::Failed || failed_agent {
            ToolCallStatus::Failed
        } else {
            native_collab_tool_status(item.status)
        };
        self.finish_operation(client, &item.id, status, None, raw);
        // Close status describes the agent before the close. Only the raw
        // FunctionCallOutput confirms whether the close itself succeeded.
        if tool != NativeSubagentTool::Close {
            for (thread_id, agent_status) in sorted_agent_statuses(&item.agents_states) {
                self.update_parent_for_thread(client, thread_id, agent_status);
            }
        }
    }

    fn handle_activity_item(&mut self, client: &SessionClient, item: &SubAgentActivityItem) {
        self.handle_activity(
            client,
            &item.id,
            item.agent_thread_id,
            item.agent_path.as_ref(),
            item.kind,
            serde_json::to_value(item).ok(),
        );
    }

    fn handle_activity(
        &mut self,
        client: &SessionClient,
        event_id: &str,
        thread_id: ThreadId,
        agent_path: &str,
        kind: SubAgentActivityKind,
        raw_output: Option<Value>,
    ) {
        let event_key = format!("{event_id}:{thread_id}:{kind:?}");
        if !self.seen_activities.insert(event_key) {
            return;
        }

        let existing_parent = self
            .parents_by_thread
            .get(&thread_id)
            .or_else(|| {
                self.agents_by_path
                    .get(agent_path)
                    .map(|(_, parent)| parent)
            })
            .cloned();
        let parent = existing_parent.unwrap_or_else(|| {
            if kind == SubAgentActivityKind::Started {
                event_id.to_string()
            } else {
                format!("native-subagent:{thread_id}")
            }
        });
        self.ensure_parent(client, &parent, thread_id, agent_path);

        let activity_id = if parent == event_id {
            format!("{event_id}:subagent_activity")
        } else {
            event_id.to_string()
        };
        let (title, status, parent_status) = match kind {
            SubAgentActivityKind::Started => (
                "Subagent started",
                ToolCallStatus::Completed,
                ToolCallStatus::InProgress,
            ),
            SubAgentActivityKind::Interacted => (
                "Subagent progressed",
                ToolCallStatus::Completed,
                ToolCallStatus::InProgress,
            ),
            SubAgentActivityKind::Interrupted => (
                "Subagent interrupted",
                ToolCallStatus::Failed,
                ToolCallStatus::Failed,
            ),
        };
        let mut call = ToolCall::new(activity_id, title)
            .kind(ToolKind::Think)
            .status(status)
            .meta(native_subagent_meta("subagent_activity", Some(&parent)));
        if let Some(raw_output) = raw_output {
            call = call.raw_output(raw_output);
        }
        client.send_tool_call(call);
        self.update_parent_status(
            client,
            &parent,
            parent_status,
            json!({
                "subagentId": thread_id,
                "agentPath": agent_path,
                "activity": kind,
            }),
        );
    }

    fn handle_response_output(
        &mut self,
        client: &SessionClient,
        call_id: &str,
        output: &FunctionCallOutputPayload,
    ) {
        let Some(operation) = self.operations.get(call_id).cloned() else {
            return;
        };
        let parsed = output
            .body
            .to_text()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        let raw_output = parsed.clone().or_else(|| serde_json::to_value(output).ok());
        let failed = output.success == Some(false);

        match operation.tool {
            NativeSubagentTool::Spawn => {
                let thread_id = parsed
                    .as_ref()
                    .and_then(|value| value.get("agent_id"))
                    .and_then(Value::as_str)
                    .and_then(|value| ThreadId::from_string(value).ok());
                if let Some(thread_id) = thread_id {
                    self.parents_by_thread
                        .insert(thread_id, call_id.to_string());
                }
                let title = parsed.as_ref().and_then(|value| {
                    value
                        .get("nickname")
                        .or_else(|| value.get("agent_nickname"))
                        .or_else(|| value.get("agent_role"))
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                });
                self.finish_operation(
                    client,
                    call_id,
                    if failed || thread_id.is_none() {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::InProgress
                    },
                    title,
                    raw_output,
                );
            }
            NativeSubagentTool::SendInput => {
                self.finish_operation(
                    client,
                    call_id,
                    if failed {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    },
                    None,
                    raw_output,
                );
            }
            NativeSubagentTool::Wait => {
                let status_values = parsed
                    .as_ref()
                    .and_then(|value| value.get("status"))
                    .and_then(Value::as_object);
                let sole_target = (status_values.is_some_and(|values| values.len() == 1)
                    && operation.target_thread_ids.len() == 1)
                    .then_some(operation.target_thread_ids[0]);
                let statuses = status_values
                    .into_iter()
                    .flatten()
                    .filter_map(|(target, status)| {
                        let thread_id = ThreadId::from_string(target)
                            .ok()
                            .or_else(|| {
                                self.agents_by_path
                                    .get(target)
                                    .map(|(thread_id, _)| *thread_id)
                            })
                            .or(sole_target)?;
                        Some((
                            thread_id,
                            serde_json::from_value::<AgentStatus>(status.clone()).ok()?,
                        ))
                    })
                    .collect::<HashMap<_, _>>();
                let statuses = sorted_agent_statuses(&statuses);
                let operation_failed = failed
                    || statuses.iter().any(|(_, status)| {
                        native_operation_status(status) == ToolCallStatus::Failed
                    });
                self.finish_operation(
                    client,
                    call_id,
                    if operation_failed {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    },
                    None,
                    raw_output,
                );
                for (thread_id, status) in statuses {
                    self.update_parent_for_thread(client, thread_id, status);
                }
            }
            NativeSubagentTool::Resume => {
                let status = parsed
                    .as_ref()
                    .and_then(|value| value.get("status"))
                    .and_then(|value| serde_json::from_value::<AgentStatus>(value.clone()).ok());
                self.finish_operation(
                    client,
                    call_id,
                    if failed {
                        ToolCallStatus::Failed
                    } else {
                        status
                            .as_ref()
                            .map(native_operation_status)
                            .unwrap_or(ToolCallStatus::Completed)
                    },
                    None,
                    raw_output,
                );
                if let (Some(thread_id), Some(status)) =
                    (operation.target_thread_ids.first(), status.as_ref())
                {
                    self.update_parent_for_thread(client, *thread_id, status);
                }
            }
            NativeSubagentTool::Close => {
                let operation_status = if failed {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                self.finish_operation(client, call_id, operation_status, None, raw_output);
                if !failed && let Some(thread_id) = operation.target_thread_ids.first() {
                    self.update_parent_for_thread_status(
                        client,
                        *thread_id,
                        ToolCallStatus::Completed,
                        json!({
                            "subagentId": thread_id,
                            "previousStatus": parsed
                                .as_ref()
                                .and_then(|value| value.get("previous_status")),
                        }),
                    );
                }
            }
        }
    }

    fn start_thread_operation(
        &mut self,
        client: &SessionClient,
        call_id: &str,
        tool: NativeSubagentTool,
        thread_id: ThreadId,
        raw_input: Option<Value>,
    ) {
        let targets = vec![thread_id];
        let parent = self.parent_for_targets(&targets);
        self.start_operation(client, call_id, tool, targets, parent, raw_input);
    }

    fn start_operation(
        &mut self,
        client: &SessionClient,
        call_id: &str,
        tool: NativeSubagentTool,
        target_thread_ids: Vec<ThreadId>,
        parent_tool_call_id: Option<String>,
        raw_input: Option<Value>,
    ) {
        if let Some(operation) = self.operations.get_mut(call_id) {
            // Parent metadata is immutable once the ACP ToolCall is emitted.
            // In particular, a root-level multi-agent wait must not become
            // nested merely because only one target appears in its result.
            if operation.target_thread_ids.is_empty() {
                operation.target_thread_ids = target_thread_ids;
            }
            return;
        }

        let parent_tool_call_id = parent_tool_call_id.or_else(|| {
            if target_thread_ids.len() == 1 {
                self.parents_by_thread.get(&target_thread_ids[0]).cloned()
            } else {
                None
            }
        });
        let mut call = ToolCall::new(call_id.to_string(), tool.title())
            .kind(ToolKind::Think)
            .status(ToolCallStatus::InProgress)
            .meta(native_subagent_meta(
                tool.native_name(),
                parent_tool_call_id.as_deref(),
            ));
        if let Some(raw_input) = raw_input {
            call = call.raw_input(raw_input);
        }
        client.send_tool_call(call);

        if tool == NativeSubagentTool::Spawn {
            self.parent_tool_calls.insert(call_id.to_string());
            self.parent_statuses
                .insert(call_id.to_string(), ToolCallStatus::InProgress);
        }
        self.operations.insert(
            call_id.to_string(),
            NativeSubagentOperation {
                tool,
                parent_tool_call_id,
                target_thread_ids,
                emitted_status: None,
            },
        );
    }

    fn finish_operation(
        &mut self,
        client: &SessionClient,
        call_id: &str,
        status: ToolCallStatus,
        title: Option<String>,
        raw_output: Option<Value>,
    ) {
        let Some(operation) = self.operations.get_mut(call_id) else {
            return;
        };
        if operation.emitted_status == Some(status) {
            return;
        }
        let tool = operation.tool;
        let parent = operation.parent_tool_call_id.clone();
        operation.emitted_status = Some(status);

        let mut fields = ToolCallUpdateFields::new().status(status);
        if let Some(title) = title {
            fields = fields.title(title);
        }
        if let Some(raw_output) = raw_output {
            fields = fields.raw_output(raw_output);
        }
        client.send_tool_call_update(
            ToolCallUpdate::new(call_id.to_string(), fields)
                .meta(native_subagent_meta(tool.native_name(), parent.as_deref())),
        );
        if tool == NativeSubagentTool::Spawn {
            self.parent_statuses.insert(call_id.to_string(), status);
        }
    }

    fn finish_agent_operation(
        &mut self,
        client: &SessionClient,
        call_id: &str,
        tool: NativeSubagentTool,
        thread_id: ThreadId,
        status: &AgentStatus,
        raw_output: Option<Value>,
    ) {
        let targets = vec![thread_id];
        let parent = self.parent_for_targets(&targets);
        self.start_operation(client, call_id, tool, targets, parent, None);
        let operation_status = if tool == NativeSubagentTool::Close {
            native_close_status(status)
        } else {
            native_operation_status(status)
        };
        self.finish_operation(client, call_id, operation_status, None, raw_output);
        // See handle_collab_item: a legacy close-end status is pre-close state.
        if tool != NativeSubagentTool::Close {
            self.update_parent_for_thread(client, thread_id, status);
        }
    }

    fn ensure_parent(
        &mut self,
        client: &SessionClient,
        parent: &str,
        thread_id: ThreadId,
        agent_path: &str,
    ) {
        self.parents_by_thread.insert(thread_id, parent.to_string());
        self.agents_by_path
            .insert(agent_path.to_string(), (thread_id, parent.to_string()));
        if !self.parent_tool_calls.insert(parent.to_string()) {
            return;
        }

        let label = agent_path
            .rsplit('/')
            .find(|part| !part.is_empty())
            .unwrap_or("Subagent");
        client.send_tool_call(
            ToolCall::new(parent.to_string(), label.to_string())
                .kind(ToolKind::Think)
                .status(ToolCallStatus::InProgress)
                .raw_input(json!({
                    "subagentId": thread_id,
                    "agentPath": agent_path,
                }))
                .meta(native_subagent_meta("Agent", None)),
        );
        self.parent_statuses
            .insert(parent.to_string(), ToolCallStatus::InProgress);
    }

    fn parent_for_targets(&self, targets: &[ThreadId]) -> Option<String> {
        if targets.len() != 1 {
            return None;
        }
        self.parents_by_thread.get(&targets[0]).cloned()
    }

    fn update_parent_for_thread(
        &mut self,
        client: &SessionClient,
        thread_id: ThreadId,
        status: &AgentStatus,
    ) {
        let Some(parent) = self.parents_by_thread.get(&thread_id).cloned() else {
            return;
        };
        self.update_parent_status(
            client,
            &parent,
            native_agent_status(status),
            json!({
                "subagentId": thread_id,
                "status": status,
            }),
        );
    }

    fn update_parent_for_thread_status(
        &mut self,
        client: &SessionClient,
        thread_id: ThreadId,
        status: ToolCallStatus,
        raw_output: Value,
    ) {
        let Some(parent) = self.parents_by_thread.get(&thread_id).cloned() else {
            return;
        };
        self.update_parent_status(client, &parent, status, raw_output);
    }

    fn update_parent_status(
        &mut self,
        client: &SessionClient,
        parent: &str,
        status: ToolCallStatus,
        raw_output: Value,
    ) {
        if self.parent_statuses.get(parent) == Some(&status) {
            return;
        }
        self.parent_statuses.insert(parent.to_string(), status);
        client.send_tool_call_update(
            ToolCallUpdate::new(
                parent.to_string(),
                ToolCallUpdateFields::new()
                    .status(status)
                    .raw_output(raw_output),
            )
            .meta(native_subagent_meta("Agent", None)),
        );
    }
}

fn native_subagent_meta(native_tool_name: &str, parent_tool_call_id: Option<&str>) -> Meta {
    let mut anyharness = json!({
        "nativeToolName": native_tool_name,
        "toolKind": "subagent",
    });
    if let Some(parent_tool_call_id) = parent_tool_call_id {
        anyharness["parentToolCallId"] = json!(parent_tool_call_id);
    }
    Meta::from_iter([(ANYHARNESS_META_KEY.to_string(), anyharness)])
}

fn native_agent_status(status: &AgentStatus) -> ToolCallStatus {
    match status {
        AgentStatus::Errored(_) | AgentStatus::Interrupted | AgentStatus::NotFound => {
            ToolCallStatus::Failed
        }
        AgentStatus::Completed(_) | AgentStatus::Shutdown => ToolCallStatus::Completed,
        AgentStatus::PendingInit | AgentStatus::Running => ToolCallStatus::InProgress,
    }
}

fn native_operation_status(status: &AgentStatus) -> ToolCallStatus {
    match status {
        AgentStatus::Errored(_) | AgentStatus::Interrupted | AgentStatus::NotFound => {
            ToolCallStatus::Failed
        }
        AgentStatus::PendingInit
        | AgentStatus::Running
        | AgentStatus::Completed(_)
        | AgentStatus::Shutdown => ToolCallStatus::Completed,
    }
}

fn native_close_status(status: &AgentStatus) -> ToolCallStatus {
    match status {
        AgentStatus::Errored(_) | AgentStatus::NotFound => ToolCallStatus::Failed,
        AgentStatus::PendingInit
        | AgentStatus::Running
        | AgentStatus::Interrupted
        | AgentStatus::Completed(_)
        | AgentStatus::Shutdown => ToolCallStatus::Completed,
    }
}

fn native_collab_tool_status(status: CollabAgentToolCallStatus) -> ToolCallStatus {
    match status {
        CollabAgentToolCallStatus::InProgress => ToolCallStatus::InProgress,
        CollabAgentToolCallStatus::Completed => ToolCallStatus::Completed,
        CollabAgentToolCallStatus::Failed => ToolCallStatus::Failed,
    }
}

fn native_agent_title(nickname: Option<&str>, role: Option<&str>) -> String {
    nickname.or(role).unwrap_or("Subagent").to_string()
}

fn native_targets_from_input(tool: NativeSubagentTool, input: &Value) -> Vec<ThreadId> {
    let values = match tool {
        NativeSubagentTool::Spawn => Vec::new(),
        NativeSubagentTool::SendInput | NativeSubagentTool::Close => input
            .get("target")
            .and_then(Value::as_str)
            .into_iter()
            .collect(),
        NativeSubagentTool::Resume => input
            .get("id")
            .and_then(Value::as_str)
            .into_iter()
            .collect(),
        NativeSubagentTool::Wait => input
            .get("targets")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect(),
    };
    sorted_thread_ids(
        values
            .into_iter()
            .filter_map(|value| ThreadId::from_string(value).ok())
            .collect(),
    )
}

fn sorted_thread_ids(mut thread_ids: Vec<ThreadId>) -> Vec<ThreadId> {
    thread_ids.sort_by_key(ToString::to_string);
    thread_ids.dedup();
    thread_ids
}

fn sorted_agent_statuses(
    statuses: &HashMap<ThreadId, AgentStatus>,
) -> Vec<(ThreadId, &AgentStatus)> {
    statuses
        .iter()
        .sorted_by_key(|(thread_id, _)| thread_id.to_string())
        .map(|(thread_id, status)| (*thread_id, status))
        .collect()
}

#[cfg(test)]
#[path = "native_subagents_tests.rs"]
mod tests;
