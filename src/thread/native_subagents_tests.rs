use super::*;
use crate::thread::{ClientSender, SessionClient};
use agent_client_protocol::{
    Error,
    schema::{
        ExtRequest, ExtResponse, RequestPermissionRequest, RequestPermissionResponse, SessionId,
        SessionNotification, SessionUpdate,
    },
};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

#[derive(Default)]
struct NativeStubClient {
    notifications: Mutex<Vec<SessionNotification>>,
}

impl ClientSender for NativeStubClient {
    fn send_session_notification(&self, notification: SessionNotification) -> Result<(), Error> {
        self.notifications.lock().unwrap().push(notification);
        Ok(())
    }

    fn request_permission(
        &self,
        _request: RequestPermissionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<RequestPermissionResponse, Error>> + Send + '_>> {
        Box::pin(async { Err(Error::internal_error()) })
    }

    fn ext_method(
        &self,
        _request: ExtRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ExtResponse, Error>> + Send + '_>> {
        Box::pin(async { Err(Error::internal_error()) })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct NativeNotice {
    phase: &'static str,
    id: String,
    status: ToolCallStatus,
    native_tool_name: String,
    parent_tool_call_id: Option<String>,
}

fn native_notices(client: &NativeStubClient) -> Vec<NativeNotice> {
    client
        .notifications
        .lock()
        .unwrap()
        .iter()
        .filter_map(|notification| match &notification.update {
            SessionUpdate::ToolCall(tool) => {
                let anyharness = tool.meta.as_ref()?.get(ANYHARNESS_META_KEY)?;
                Some(NativeNotice {
                    phase: "start",
                    id: tool.tool_call_id.0.to_string(),
                    status: tool.status,
                    native_tool_name: anyharness.get("nativeToolName")?.as_str()?.to_string(),
                    parent_tool_call_id: anyharness
                        .get("parentToolCallId")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                })
            }
            SessionUpdate::ToolCallUpdate(update) => {
                let anyharness = update.meta.as_ref()?.get(ANYHARNESS_META_KEY)?;
                Some(NativeNotice {
                    phase: "update",
                    id: update.tool_call_id.0.to_string(),
                    status: update.fields.status?,
                    native_tool_name: anyharness.get("nativeToolName")?.as_str()?.to_string(),
                    parent_tool_call_id: anyharness
                        .get("parentToolCallId")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                })
            }
            _ => None,
        })
        .collect()
}

fn native_test_client() -> (Arc<NativeStubClient>, SessionClient) {
    let client = Arc::new(NativeStubClient::default());
    let session_client = SessionClient::with_client(
        SessionId::new("native-subagent-test"),
        client.clone(),
        Arc::default(),
    );
    (client, session_client)
}

fn apply_event(state: &mut NativeSubagentState, client: &SessionClient, event: EventMsg) {
    assert!(state.handle_event(client, &event));
}

fn apply_response(state: &mut NativeSubagentState, client: &SessionClient, item: ResponseItem) {
    assert!(state.handle_response_item(client, &item));
}

fn notice_shapes(notices: &[NativeNotice]) -> Vec<String> {
    notices
        .iter()
        .map(|notice| {
            format!(
                "{}:{}:{:?}:{}:{}",
                notice.phase,
                notice.id,
                notice.status,
                notice.native_tool_name,
                notice.parent_tool_call_id.as_deref().unwrap_or("-")
            )
        })
        .collect()
}

fn fixed_thread_id(suffix: u8) -> ThreadId {
    ThreadId::from_string(&format!("00000000-0000-7000-8000-{suffix:012x}"))
        .expect("fixed test thread id")
}

fn response_call(
    call_id: &str,
    name: &str,
    namespace: Option<&str>,
    arguments: Value,
) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: name.to_string(),
        namespace: namespace.map(ToOwned::to_owned),
        arguments: arguments.to_string(),
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn v1_call(call_id: &str, name: &str, arguments: Value) -> ResponseItem {
    response_call(call_id, name, Some(MULTI_AGENT_V1_NAMESPACE), arguments)
}

fn response_output(call_id: &str, output: Value) -> ResponseItem {
    response_output_with_success(call_id, output, true)
}

fn response_output_with_success(call_id: &str, output: Value, success: bool) -> ResponseItem {
    response_text_output_with_success(call_id, output.to_string(), Some(success))
}

fn response_text_output_with_success(
    call_id: &str,
    output: impl Into<String>,
    success: Option<bool>,
) -> ResponseItem {
    let mut payload = codex_protocol::models::FunctionCallOutputPayload::from_text(output.into());
    payload.success = success;
    ResponseItem::FunctionCallOutput {
        id: None,
        call_id: call_id.to_string(),
        output: payload,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn completed_native_item(thread_id: ThreadId, item: TurnItem) -> EventMsg {
    EventMsg::ItemCompleted(ItemCompletedEvent {
        thread_id,
        turn_id: "turn-fixture".to_string(),
        item,
        completed_at_ms: 1,
    })
}

fn completed_collab_event(
    id: &str,
    tool: CollabAgentTool,
    sender_thread_id: ThreadId,
    agents_states: HashMap<ThreadId, AgentStatus>,
) -> EventMsg {
    use codex_protocol::protocol::CollabAgentRef;

    let receiver_thread_ids = sorted_thread_ids(agents_states.keys().copied().collect());
    let receiver_agents = receiver_thread_ids
        .iter()
        .copied()
        .map(|thread_id| CollabAgentRef {
            thread_id,
            agent_nickname: Some(format!("agent-{thread_id}")),
            agent_role: Some("explorer".to_string()),
        })
        .collect();
    completed_native_item(
        sender_thread_id,
        TurnItem::CollabAgentToolCall(CollabAgentToolCallItem {
            id: id.to_string(),
            tool,
            status: CollabAgentToolCallStatus::Completed,
            sender_thread_id,
            receiver_thread_ids,
            receiver_agents,
            prompt: matches!(
                tool,
                CollabAgentTool::SpawnAgent | CollabAgentTool::SendInput
            )
            .then(|| "Inspect README.md".to_string()),
            model: (tool == CollabAgentTool::SpawnAgent).then(|| "gpt-5".to_string()),
            reasoning_effort: None,
            agents_states,
        }),
    )
}

#[test]
fn native_v1_live_and_paginated_replay_have_identical_identity_and_metadata() {
    use codex_protocol::protocol::{
        CollabAgentInteractionBeginEvent, CollabAgentInteractionEndEvent,
        CollabAgentSpawnBeginEvent, CollabAgentSpawnEndEvent, CollabAgentStatusEntry,
        CollabWaitingBeginEvent, CollabWaitingEndEvent,
    };

    let sender = fixed_thread_id(1);
    let child = fixed_thread_id(2);
    let (live_client, live_session) = native_test_client();
    let mut live = NativeSubagentState::default();
    let live_events = vec![
        EventMsg::CollabAgentSpawnBegin(CollabAgentSpawnBeginEvent {
            call_id: "spawn-1".to_string(),
            started_at_ms: 1,
            sender_thread_id: sender,
            prompt: "Inspect README.md".to_string(),
            model: "gpt-5".to_string(),
            reasoning_effort: Default::default(),
        }),
        EventMsg::CollabAgentSpawnEnd(CollabAgentSpawnEndEvent {
            call_id: "spawn-1".to_string(),
            completed_at_ms: 2,
            sender_thread_id: sender,
            new_thread_id: Some(child),
            new_agent_nickname: Some(format!("agent-{child}")),
            new_agent_role: Some("explorer".to_string()),
            prompt: "Inspect README.md".to_string(),
            model: "gpt-5".to_string(),
            reasoning_effort: Default::default(),
            status: AgentStatus::Running,
        }),
        EventMsg::CollabAgentInteractionBegin(CollabAgentInteractionBeginEvent {
            call_id: "send-1".to_string(),
            started_at_ms: 3,
            sender_thread_id: sender,
            receiver_thread_id: child,
            prompt: "Inspect README.md".to_string(),
        }),
        EventMsg::CollabAgentInteractionEnd(CollabAgentInteractionEndEvent {
            call_id: "send-1".to_string(),
            completed_at_ms: 4,
            sender_thread_id: sender,
            receiver_thread_id: child,
            receiver_agent_nickname: Some(format!("agent-{child}")),
            receiver_agent_role: Some("explorer".to_string()),
            prompt: "Inspect README.md".to_string(),
            status: AgentStatus::Running,
        }),
        EventMsg::CollabWaitingBegin(CollabWaitingBeginEvent {
            started_at_ms: 5,
            sender_thread_id: sender,
            receiver_thread_ids: vec![child],
            receiver_agents: Vec::new(),
            call_id: "wait-1".to_string(),
        }),
        EventMsg::CollabWaitingEnd(CollabWaitingEndEvent {
            sender_thread_id: sender,
            call_id: "wait-1".to_string(),
            completed_at_ms: 6,
            agent_statuses: vec![CollabAgentStatusEntry {
                thread_id: child,
                agent_nickname: Some(format!("agent-{child}")),
                agent_role: Some("explorer".to_string()),
                status: AgentStatus::Completed(Some("done".to_string())),
            }],
            statuses: [(child, AgentStatus::Completed(Some("done".to_string())))]
                .into_iter()
                .collect(),
        }),
    ];
    for event in &live_events {
        assert!(live.handle_event(&live_session, event));
    }

    let (replay_client, replay_session) = native_test_client();
    let mut replay = NativeSubagentState::default();
    apply_response(
        &mut replay,
        &replay_session,
        v1_call(
            "spawn-1",
            "spawn_agent",
            json!({"message": "Inspect README.md"}),
        ),
    );
    apply_event(
        &mut replay,
        &replay_session,
        completed_collab_event(
            "spawn-1",
            CollabAgentTool::SpawnAgent,
            sender,
            [(child, AgentStatus::Running)].into_iter().collect(),
        ),
    );
    apply_response(
        &mut replay,
        &replay_session,
        response_output(
            "spawn-1",
            json!({"agent_id": child, "nickname": format!("agent-{child}")}),
        ),
    );
    apply_response(
        &mut replay,
        &replay_session,
        v1_call(
            "send-1",
            "send_input",
            json!({"target": child, "message": "Inspect README.md"}),
        ),
    );
    apply_event(
        &mut replay,
        &replay_session,
        completed_collab_event(
            "send-1",
            CollabAgentTool::SendInput,
            sender,
            [(child, AgentStatus::Running)].into_iter().collect(),
        ),
    );
    apply_response(
        &mut replay,
        &replay_session,
        response_output("send-1", json!({"submission_id": "turn-child"})),
    );
    apply_response(
        &mut replay,
        &replay_session,
        v1_call(
            "wait-1",
            "wait_agent",
            json!({"targets": [child], "timeout_ms": 1000}),
        ),
    );
    apply_event(
        &mut replay,
        &replay_session,
        completed_collab_event(
            "wait-1",
            CollabAgentTool::Wait,
            sender,
            [(child, AgentStatus::Completed(Some("done".to_string())))]
                .into_iter()
                .collect(),
        ),
    );
    apply_response(
        &mut replay,
        &replay_session,
        response_output(
            "wait-1",
            json!({
                "status": {(child.to_string()): {"completed": "done"}},
                "timed_out": false,
            }),
        ),
    );

    assert_eq!(native_notices(&live_client), native_notices(&replay_client));
    let notices = native_notices(&live_client);
    assert_eq!(
        notice_shapes(&notices),
        [
            "start:spawn-1:InProgress:Agent:-",
            "update:spawn-1:InProgress:Agent:-",
            "start:send-1:InProgress:send_input:spawn-1",
            "update:send-1:Completed:send_input:spawn-1",
            "start:wait-1:InProgress:wait:spawn-1",
            "update:wait-1:Completed:wait:spawn-1",
            "update:spawn-1:Completed:Agent:-",
        ]
    );
}

#[test]
fn native_v2_activity_live_and_paginated_replay_match_without_raw_call_duplicates() {
    use codex_protocol::{
        AgentPath,
        protocol::{RawResponseItemEvent, SubAgentActivityEvent},
    };

    let sender = fixed_thread_id(1);
    let child = fixed_thread_id(2);
    let path = AgentPath::try_from("/root/reader").expect("fixed agent path");
    let activities = [
        (
            "spawn-v2",
            "spawn_agent",
            SubAgentActivityKind::Started,
            json!({
                "message": "Inspect README.md",
                "task_name": "reader",
                "fork_turns": "none",
            }),
            response_output(
                "spawn-v2",
                json!({"task_name": "/root/reader", "nickname": "reader"}),
            ),
        ),
        (
            "message-v2",
            "send_message",
            SubAgentActivityKind::Interacted,
            json!({"target": "/root/reader", "message": "Continue"}),
            // A post-tool hook can fail after the collaboration handler has
            // already emitted its canonical activity. The later failed output
            // must not create a duplicate failure tool.
            response_output_with_success(
                "message-v2",
                json!({"error": "post-tool hook rejected the result"}),
                false,
            ),
        ),
    ];

    let (live_client, live_session) = native_test_client();
    let mut live = NativeSubagentState::default();
    for (event_id, name, kind, arguments, output) in &activities {
        apply_event(
            &mut live,
            &live_session,
            EventMsg::RawResponseItem(RawResponseItemEvent {
                item: response_call(
                    event_id,
                    name,
                    Some(MULTI_AGENT_V2_NAMESPACE),
                    arguments.clone(),
                ),
            }),
        );
        apply_event(
            &mut live,
            &live_session,
            completed_native_item(
                sender,
                TurnItem::SubAgentActivity(SubAgentActivityItem {
                    id: event_id.to_string(),
                    kind: *kind,
                    agent_thread_id: child,
                    agent_path: path.clone(),
                }),
            ),
        );
        let notice_count = native_notices(&live_client).len();
        apply_event(
            &mut live,
            &live_session,
            EventMsg::SubAgentActivity(SubAgentActivityEvent {
                event_id: event_id.to_string(),
                occurred_at_ms: 1,
                agent_thread_id: child,
                agent_path: path.clone(),
                kind: *kind,
            }),
        );
        assert_eq!(native_notices(&live_client).len(), notice_count);
        apply_event(
            &mut live,
            &live_session,
            EventMsg::RawResponseItem(RawResponseItemEvent {
                item: output.clone(),
            }),
        );
    }

    let (replay_client, replay_session) = native_test_client();
    let mut replay = NativeSubagentState::default();
    for (event_id, name, kind, arguments, output) in activities {
        apply_response(
            &mut replay,
            &replay_session,
            response_call(event_id, name, Some(MULTI_AGENT_V2_NAMESPACE), arguments),
        );
        apply_event(
            &mut replay,
            &replay_session,
            completed_native_item(
                sender,
                TurnItem::SubAgentActivity(SubAgentActivityItem {
                    id: event_id.to_string(),
                    kind,
                    agent_thread_id: child,
                    agent_path: path.clone(),
                }),
            ),
        );
        apply_response(&mut replay, &replay_session, output);
    }

    assert_eq!(native_notices(&live_client), native_notices(&replay_client));
    let notices = native_notices(&replay_client);
    assert_eq!(
        notices
            .iter()
            .map(|notice| notice.id.as_str())
            .collect::<Vec<_>>(),
        vec!["spawn-v2", "spawn-v2:subagent_activity", "message-v2",],
    );
    assert_eq!(notices[0].native_tool_name, "Agent");
    assert_eq!(notices[1].parent_tool_call_id.as_deref(), Some("spawn-v2"),);
    assert_eq!(notices[2].parent_tool_call_id.as_deref(), Some("spawn-v2"),);
}

#[test]
fn native_v2_failed_calls_survive_live_and_paginated_replay_without_success_flag() {
    use codex_protocol::{
        AgentPath,
        protocol::{RawResponseItemEvent, SubAgentActivityEvent},
    };

    fn apply_raw(
        state: &mut NativeSubagentState,
        session: &SessionClient,
        item: ResponseItem,
        live: bool,
    ) {
        if live {
            apply_event(
                state,
                session,
                EventMsg::RawResponseItem(RawResponseItemEvent { item }),
            );
        } else {
            apply_response(state, session, item);
        }
    }

    fn fixture(live: bool) -> Vec<NativeNotice> {
        let sender = fixed_thread_id(1);
        let child = fixed_thread_id(2);
        let path = AgentPath::try_from("/root/reader").expect("fixed agent path");
        let (client, session) = native_test_client();
        let mut state = NativeSubagentState::default();

        apply_raw(
            &mut state,
            &session,
            response_call(
                "spawn-seed",
                "spawn_agent",
                Some(MULTI_AGENT_V2_NAMESPACE),
                json!({
                    "message": "Inspect README.md",
                    "task_name": "reader",
                    "fork_turns": "none",
                }),
            ),
            live,
        );
        apply_event(
            &mut state,
            &session,
            completed_native_item(
                sender,
                TurnItem::SubAgentActivity(SubAgentActivityItem {
                    id: "spawn-seed".to_string(),
                    kind: SubAgentActivityKind::Started,
                    agent_thread_id: child,
                    agent_path: path.clone(),
                }),
            ),
        );
        if live {
            let notice_count = native_notices(&client).len();
            apply_event(
                &mut state,
                &session,
                EventMsg::SubAgentActivity(SubAgentActivityEvent {
                    event_id: "spawn-seed".to_string(),
                    occurred_at_ms: 1,
                    agent_thread_id: child,
                    agent_path: path,
                    kind: SubAgentActivityKind::Started,
                }),
            );
            assert_eq!(native_notices(&client).len(), notice_count);
        }
        apply_raw(
            &mut state,
            &session,
            response_text_output_with_success(
                "spawn-seed",
                r#"{"task_name":"/root/reader","nickname":"reader"}"#,
                live.then_some(true),
            ),
            live,
        );
        assert!(state.suppressed_v2_calls.is_empty());
        assert!(state.suppressed_v2_order.is_empty());
        assert!(state.early_v2_activities.ids.is_empty());
        assert!(state.completed_v2_calls.contains("spawn-seed"));

        let failures = [
            (
                "spawn-invalid",
                "spawn_agent",
                json!({
                    "message": "Inspect README.md",
                    "task_name": "invalid",
                    "fork_turns": "0",
                }),
                "fork_turns must be `none`, `all`, or a positive integer string",
            ),
            (
                "message-failed",
                "send_message",
                json!({"target": "/root/reader", "message": ""}),
                "Empty message can't be sent to an agent",
            ),
            (
                "followup-failed",
                "followup_task",
                json!({"target": "reader", "message": ""}),
                "Empty message can't be sent to an agent",
            ),
            (
                "interrupt-failed",
                "interrupt_agent",
                json!({"target": child}),
                "failed to interrupt agent",
            ),
        ];
        for (call_id, name, arguments, error) in failures {
            apply_raw(
                &mut state,
                &session,
                response_call(call_id, name, Some(MULTI_AGENT_V2_NAMESPACE), arguments),
                live,
            );
            let output = response_text_output_with_success(call_id, error, live.then_some(false));
            apply_raw(&mut state, &session, output.clone(), live);
            assert!(!state.suppressed_v2_calls.contains_key(call_id));
            assert!(!state.suppressed_v2_order.iter().any(|id| id == call_id));
            assert!(!state.early_v2_activities.contains(call_id));
            assert!(state.completed_v2_calls.contains(call_id));
            let notice_count = native_notices(&client).len();
            let tombstone_count = state.completed_v2_calls.ids.len();
            // Duplicate delivery must not create a second logical item/output.
            apply_raw(&mut state, &session, output, live);
            assert_eq!(native_notices(&client).len(), notice_count);
            assert_eq!(state.completed_v2_calls.ids.len(), tombstone_count);
        }
        assert_eq!(state.completed_v2_calls.ids.len(), 5);

        native_notices(&client)
    }

    let live = fixture(true);
    let replay = fixture(false);
    assert_eq!(live, replay);
    assert_eq!(
        notice_shapes(&live),
        [
            "start:spawn-seed:InProgress:Agent:-",
            "start:spawn-seed:subagent_activity:Completed:subagent_activity:spawn-seed",
            "start:spawn-invalid:InProgress:Agent:-",
            "update:spawn-invalid:Failed:Agent:-",
            "start:message-failed:InProgress:send_input:spawn-seed",
            "update:message-failed:Failed:send_input:spawn-seed",
            "start:followup-failed:InProgress:send_input:spawn-seed",
            "update:followup-failed:Failed:send_input:spawn-seed",
            "start:interrupt-failed:InProgress:close_agent:spawn-seed",
            "update:interrupt-failed:Failed:close_agent:spawn-seed",
        ]
    );
}

#[test]
fn native_v2_only_claims_the_configured_collaboration_namespace() {
    let (client, session) = native_test_client();
    let mut state = NativeSubagentState::default();
    let arguments = json!({
        "message": "Inspect README.md",
        "task_name": "reader",
        "fork_turns": "none",
    });

    for (call_id, namespace) in [("plain-v2", None), ("foreign-v2", Some("foreign"))] {
        assert!(!state.handle_response_item(
            &session,
            &response_call(call_id, "spawn_agent", namespace, arguments.clone()),
        ));
    }
    assert!(state.suppressed_v2_calls.is_empty());
    assert!(native_notices(&client).is_empty());

    apply_response(
        &mut state,
        &session,
        v1_call(
            "legacy-v1",
            "spawn_agent",
            json!({"message": "Inspect README.md"}),
        ),
    );
    assert!(state.operations.contains_key("legacy-v1"));
    assert!(!state.suppressed_v2_calls.contains_key("legacy-v1"));

    apply_response(
        &mut state,
        &session,
        response_call(
            "collaboration-v2",
            "spawn_agent",
            Some(MULTI_AGENT_V2_NAMESPACE),
            arguments,
        ),
    );
    assert!(state.suppressed_v2_calls.contains_key("collaboration-v2"));

    assert!(!state.handle_response_item(
        &session,
        &response_call("plain-wait", "wait_agent", None, json!({})),
    ));
    apply_response(
        &mut state,
        &session,
        response_call(
            "collaboration-wait",
            "wait_agent",
            Some(MULTI_AGENT_V2_NAMESPACE),
            json!({}),
        ),
    );
    assert!(state.operations.contains_key("collaboration-wait"));
}

#[test]
fn native_v2_wait_live_and_replay_classify_plain_failures_and_structured_success() {
    use codex_protocol::protocol::RawResponseItemEvent;

    fn apply_raw(
        state: &mut NativeSubagentState,
        session: &SessionClient,
        item: ResponseItem,
        live: bool,
    ) {
        if live {
            apply_event(
                state,
                session,
                EventMsg::RawResponseItem(RawResponseItemEvent { item }),
            );
        } else {
            apply_response(state, session, item);
        }
    }

    fn fixture(live: bool) -> Vec<NativeNotice> {
        let sender = fixed_thread_id(1);
        let (client, session) = native_test_client();
        let mut state = NativeSubagentState::default();

        for (call_id, arguments, error) in [
            (
                "wait-invalid-timeout",
                json!({"timeout_ms": 0}),
                "timeout_ms must be at least 50",
            ),
            (
                "wait-handler-failed",
                json!({"targets": []}),
                "failed to parse function arguments: unknown field `targets`, expected `timeout_ms`",
            ),
        ] {
            apply_raw(
                &mut state,
                &session,
                response_call(
                    call_id,
                    "wait_agent",
                    Some(MULTI_AGENT_V2_NAMESPACE),
                    arguments,
                ),
                live,
            );
            apply_raw(
                &mut state,
                &session,
                response_text_output_with_success(call_id, error, live.then_some(false)),
                live,
            );
        }

        apply_raw(
            &mut state,
            &session,
            response_call(
                "wait-success",
                "wait_agent",
                Some(MULTI_AGENT_V2_NAMESPACE),
                json!({"timeout_ms": 50}),
            ),
            live,
        );
        apply_event(
            &mut state,
            &session,
            completed_collab_event(
                "wait-success",
                CollabAgentTool::Wait,
                sender,
                HashMap::new(),
            ),
        );
        apply_raw(
            &mut state,
            &session,
            response_text_output_with_success(
                "wait-success",
                r#"{"message":"Wait timed out.","timed_out":true}"#,
                None,
            ),
            live,
        );

        apply_raw(
            &mut state,
            &session,
            response_call(
                "wait-post-hook-failed",
                "wait_agent",
                Some(MULTI_AGENT_V2_NAMESPACE),
                json!({"timeout_ms": 50}),
            ),
            live,
        );
        apply_event(
            &mut state,
            &session,
            completed_collab_event(
                "wait-post-hook-failed",
                CollabAgentTool::Wait,
                sender,
                HashMap::new(),
            ),
        );
        apply_raw(
            &mut state,
            &session,
            response_text_output_with_success(
                "wait-post-hook-failed",
                "PostToolUse hook blocked the completed result",
                live.then_some(false),
            ),
            live,
        );

        native_notices(&client)
    }

    let live = fixture(true);
    let replay = fixture(false);
    assert_eq!(live, replay);
    assert_eq!(
        notice_shapes(&live),
        [
            "start:wait-invalid-timeout:InProgress:wait:-",
            "update:wait-invalid-timeout:Failed:wait:-",
            "start:wait-handler-failed:InProgress:wait:-",
            "update:wait-handler-failed:Failed:wait:-",
            "start:wait-success:InProgress:wait:-",
            "update:wait-success:Completed:wait:-",
            "start:wait-post-hook-failed:InProgress:wait:-",
            "update:wait-post-hook-failed:Completed:wait:-",
        ]
    );

    let (v1_client, v1_session) = native_test_client();
    let mut v1 = NativeSubagentState::default();
    apply_response(
        &mut v1,
        &v1_session,
        v1_call(
            "wait-v1",
            "wait_agent",
            json!({"targets": [], "timeout_ms": 50}),
        ),
    );
    apply_response(
        &mut v1,
        &v1_session,
        response_text_output_with_success("wait-v1", r#"{"status":{},"timed_out":true}"#, None),
    );
    assert_eq!(
        notice_shapes(&native_notices(&v1_client)),
        [
            "start:wait-v1:InProgress:wait:-",
            "update:wait-v1:Completed:wait:-",
        ]
    );
}

#[test]
fn native_v2_early_activity_marker_is_consumed_and_terminal_state_is_payload_free() {
    use codex_protocol::AgentPath;

    let sender = fixed_thread_id(1);
    let child = fixed_thread_id(2);
    let path = AgentPath::try_from("/root/reader").expect("fixed agent path");
    let (client, session) = native_test_client();
    let mut state = NativeSubagentState::default();

    apply_event(
        &mut state,
        &session,
        completed_native_item(
            sender,
            TurnItem::SubAgentActivity(SubAgentActivityItem {
                id: "activity-first".to_string(),
                kind: SubAgentActivityKind::Started,
                agent_thread_id: child,
                agent_path: path,
            }),
        ),
    );
    assert!(state.early_v2_activities.contains("activity-first"));
    let notice_count = native_notices(&client).len();

    apply_response(
        &mut state,
        &session,
        response_call(
            "activity-first",
            "spawn_agent",
            Some(MULTI_AGENT_V2_NAMESPACE),
            json!({
                "message": "Inspect README.md",
                "task_name": "reader",
                "fork_turns": "none",
            }),
        ),
    );
    assert!(!state.early_v2_activities.contains("activity-first"));
    assert!(
        state
            .suppressed_v2_calls
            .get("activity-first")
            .is_some_and(|call| call.canonical_activity)
    );
    apply_response(
        &mut state,
        &session,
        response_text_output_with_success("activity-first", "{}", None),
    );

    assert_eq!(native_notices(&client).len(), notice_count);
    assert!(state.suppressed_v2_calls.is_empty());
    assert!(state.suppressed_v2_order.is_empty());
    assert!(state.early_v2_activities.ids.is_empty());
    assert!(state.completed_v2_calls.contains("activity-first"));
}

#[test]
fn native_v2_pending_orphans_and_id_only_tombstones_are_fifo_bounded() {
    let (_client, session) = native_test_client();
    let mut state = NativeSubagentState::default();

    for index in 0..MAX_PENDING_V2_CALLS + 2 {
        apply_response(
            &mut state,
            &session,
            response_call(
                &format!("orphan-{index}"),
                "spawn_agent",
                Some(MULTI_AGENT_V2_NAMESPACE),
                json!({
                    "message": format!("orphan payload {index}"),
                    "task_name": format!("orphan-{index}"),
                    "fork_turns": "none",
                }),
            ),
        );
    }

    assert_eq!(state.suppressed_v2_calls.len(), MAX_PENDING_V2_CALLS);
    assert_eq!(state.suppressed_v2_order.len(), MAX_PENDING_V2_CALLS);
    assert_eq!(
        state.suppressed_v2_order.front().map(String::as_str),
        Some("orphan-2")
    );
    assert!(!state.suppressed_v2_calls.contains_key("orphan-0"));
    assert!(!state.suppressed_v2_calls.contains_key("orphan-1"));
    assert!(state.completed_v2_calls.contains("orphan-0"));
    assert!(state.completed_v2_calls.contains("orphan-1"));

    for index in 0..MAX_EARLY_V2_ACTIVITIES + 2 {
        drop(
            state
                .early_v2_activities
                .insert(format!("early-{index}"), MAX_EARLY_V2_ACTIVITIES),
        );
    }
    assert_eq!(state.early_v2_activities.ids.len(), MAX_EARLY_V2_ACTIVITIES);
    assert_eq!(
        state.early_v2_activities.order.front().map(String::as_str),
        Some("early-2")
    );

    for index in 0..MAX_COMPLETED_V2_CALLS + 2 {
        drop(
            state
                .completed_v2_calls
                .insert(format!("completed-{index}"), MAX_COMPLETED_V2_CALLS),
        );
    }
    assert_eq!(state.completed_v2_calls.ids.len(), MAX_COMPLETED_V2_CALLS);
    assert_eq!(
        state.completed_v2_calls.order.front().map(String::as_str),
        Some("completed-2")
    );
    assert!(!state.completed_v2_calls.contains("orphan-0"));
    assert!(!state.completed_v2_calls.contains("completed-0"));
    assert!(!state.completed_v2_calls.contains("completed-1"));
}

#[test]
fn native_v2_evicted_early_activity_is_tombstoned_before_delayed_call_and_output() {
    let child = fixed_thread_id(2);
    let (client, session) = native_test_client();
    let mut state = NativeSubagentState::default();

    state.handle_activity(
        &session,
        "delayed-activity",
        child,
        "/root/reader",
        SubAgentActivityKind::Started,
        None,
    );
    for index in 1..=MAX_EARLY_V2_ACTIVITIES {
        state.handle_activity(
            &session,
            &format!("later-activity-{index}"),
            child,
            "/root/reader",
            SubAgentActivityKind::Interacted,
            None,
        );
    }

    assert!(!state.early_v2_activities.contains("delayed-activity"));
    assert!(state.completed_v2_calls.contains("delayed-activity"));
    let notice_count = native_notices(&client).len();

    apply_response(
        &mut state,
        &session,
        response_call(
            "delayed-activity",
            "spawn_agent",
            Some(MULTI_AGENT_V2_NAMESPACE),
            json!({
                "message": "Inspect README.md",
                "task_name": "reader",
                "fork_turns": "none",
            }),
        ),
    );
    apply_response(
        &mut state,
        &session,
        response_text_output_with_success("delayed-activity", "delayed failure output", None),
    );

    assert_eq!(native_notices(&client).len(), notice_count);
    assert!(!state.suppressed_v2_calls.contains_key("delayed-activity"));
    assert!(state.completed_v2_calls.contains("delayed-activity"));
}

#[test]
fn native_legacy_path_wait_and_structured_interrupt_fail_consistently() {
    let sender = fixed_thread_id(1);
    let child = fixed_thread_id(2);
    let (client, session) = native_test_client();
    let mut state = NativeSubagentState::default();

    apply_response(
        &mut state,
        &session,
        v1_call(
            "spawn-legacy",
            "spawn_agent",
            json!({"message": "Inspect README.md"}),
        ),
    );
    apply_response(
        &mut state,
        &session,
        response_output(
            "spawn-legacy",
            json!({"agent_id": child, "nickname": "reader"}),
        ),
    );
    apply_response(
        &mut state,
        &session,
        v1_call(
            "wait-legacy",
            "wait_agent",
            json!({"targets": [child], "timeout_ms": 1000}),
        ),
    );
    apply_response(
        &mut state,
        &session,
        response_output(
            "wait-legacy",
            json!({
                "status": {"/root/reader": {"completed": "done"}},
                "timed_out": false,
            }),
        ),
    );
    apply_event(
        &mut state,
        &session,
        completed_collab_event(
            "send-interrupted",
            CollabAgentTool::SendInput,
            sender,
            [(child, AgentStatus::Interrupted)].into_iter().collect(),
        ),
    );
    apply_response(
        &mut state,
        &session,
        v1_call("close-failed", "close_agent", json!({"target": child})),
    );
    apply_response(
        &mut state,
        &session,
        response_output_with_success("close-failed", json!({"error": "still running"}), false),
    );
    assert_eq!(
        native_notices(&client)
            .iter()
            .rev()
            .find(|notice| notice.id == "spawn-legacy")
            .map(|notice| notice.status),
        Some(ToolCallStatus::Failed),
        "a failed close must preserve the parent's prior status",
    );
    apply_response(
        &mut state,
        &session,
        v1_call("close-legacy", "close_agent", json!({"target": child})),
    );
    apply_response(
        &mut state,
        &session,
        response_output("close-legacy", json!({"previous_status": "running"})),
    );

    let notices = native_notices(&client);
    let wait = notices
        .iter()
        .find(|notice| notice.id == "wait-legacy" && notice.phase == "start")
        .expect("legacy wait start");
    assert_eq!(wait.parent_tool_call_id.as_deref(), Some("spawn-legacy"));
    assert!(notices.iter().any(|notice| {
        notice.id == "spawn-legacy" && notice.status == ToolCallStatus::Completed
    }));
    assert!(notices.iter().any(|notice| {
        notice.id == "send-interrupted" && notice.status == ToolCallStatus::Failed
    }));
    assert!(notices.iter().any(|notice| {
        notice.id == "close-legacy" && notice.status == ToolCallStatus::Completed
    }));
    assert!(
        notices.iter().any(|notice| {
            notice.id == "close-failed" && notice.status == ToolCallStatus::Failed
        })
    );
    assert_eq!(
        notices
            .iter()
            .rev()
            .find(|notice| notice.id == "spawn-legacy")
            .map(|notice| notice.status),
        Some(ToolCallStatus::Completed),
    );
}

#[test]
fn native_parent_survives_replay_to_live_and_multi_wait_is_deterministic() {
    use codex_protocol::protocol::{
        CollabAgentInteractionBeginEvent, CollabWaitingBeginEvent, CollabWaitingEndEvent,
    };

    let sender = fixed_thread_id(1);
    let first = fixed_thread_id(2);
    let second = fixed_thread_id(3);
    let (client, session) = native_test_client();
    let mut state = NativeSubagentState::default();

    for (spawn_id, child) in [("spawn-a", first), ("spawn-b", second)] {
        apply_event(
            &mut state,
            &session,
            completed_collab_event(
                spawn_id,
                CollabAgentTool::SpawnAgent,
                sender,
                [(child, AgentStatus::Running)].into_iter().collect(),
            ),
        );
    }

    apply_event(
        &mut state,
        &session,
        EventMsg::CollabAgentInteractionBegin(CollabAgentInteractionBeginEvent {
            call_id: "next-turn-send".to_string(),
            started_at_ms: 2,
            sender_thread_id: sender,
            receiver_thread_id: first,
            prompt: "Continue".to_string(),
        }),
    );
    apply_event(
        &mut state,
        &session,
        EventMsg::CollabWaitingBegin(CollabWaitingBeginEvent {
            started_at_ms: 3,
            sender_thread_id: sender,
            receiver_thread_ids: vec![second, first],
            receiver_agents: Vec::new(),
            call_id: "wait-many".to_string(),
        }),
    );
    apply_event(
        &mut state,
        &session,
        EventMsg::CollabWaitingEnd(CollabWaitingEndEvent {
            sender_thread_id: sender,
            call_id: "wait-many".to_string(),
            completed_at_ms: 4,
            agent_statuses: Vec::new(),
            statuses: [(first, AgentStatus::Interrupted)].into_iter().collect(),
        }),
    );
    apply_event(
        &mut state,
        &session,
        EventMsg::CollabWaitingBegin(CollabWaitingBeginEvent {
            started_at_ms: 5,
            sender_thread_id: sender,
            receiver_thread_ids: vec![second, first],
            receiver_agents: Vec::new(),
            call_id: "wait-order".to_string(),
        }),
    );
    apply_event(
        &mut state,
        &session,
        EventMsg::CollabWaitingEnd(CollabWaitingEndEvent {
            sender_thread_id: sender,
            call_id: "wait-order".to_string(),
            completed_at_ms: 6,
            agent_statuses: Vec::new(),
            statuses: [
                (second, AgentStatus::Completed(None)),
                (first, AgentStatus::Completed(None)),
            ]
            .into_iter()
            .collect(),
        }),
    );

    let notices = native_notices(&client);
    let send = notices
        .iter()
        .find(|notice| notice.id == "next-turn-send")
        .expect("cross-turn send notice");
    assert_eq!(send.parent_tool_call_id.as_deref(), Some("spawn-a"));
    let waits = notices
        .iter()
        .filter(|notice| notice.id == "wait-many")
        .collect::<Vec<_>>();
    assert_eq!(waits.len(), 2);
    assert!(
        waits
            .iter()
            .all(|notice| notice.parent_tool_call_id.is_none())
    );
    assert_eq!(
        notices
            .iter()
            .rev()
            .take(2)
            .map(|notice| notice.id.as_str())
            .collect::<Vec<_>>(),
        vec!["spawn-b", "spawn-a"],
        "parent terminal updates must be emitted in sorted thread-id order",
    );
    assert_eq!(
        native_agent_status(&AgentStatus::Interrupted),
        ToolCallStatus::Failed,
    );
}
