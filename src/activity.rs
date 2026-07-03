//! Anyharness ActivityPort wire types: read-only process + subagent rosters.
//!
//! Unlike goals (a persisted, queryable sqlite row we can always re-snapshot
//! in full), codex has no native roster store for `commandExecution` items
//! or collab sub-agents. These types are built entirely event-sourced, from
//! state tracked locally in `thread.rs` as the lifecycle events arrive. See
//! `codex/session-activity-architecture.md` for the pinned `ActivityProcess`
//! / `ActivitySubagent` contract these mirror, and
//! `codex/harness-runtime-mechanics.md` §4.2/§7 for the verified wire facts
//! (`commandExecution` items are strictly in-turn; collab sub-agents are
//! full child threads whose lifecycle crosses the wire via
//! `collabAgentToolCall` items on the parent).
//!
//! Wire conventions follow `goals.rs`'s `GoalWire` precedent: camelCase,
//! all fields serialize (nulls included), a `native: true` marker, and an
//! `updatedAtMs` freshness stamp on every emission.

use agent_client_protocol::schema::{ContentChunk, Meta, SessionUpdate};
use codex_protocol::protocol::{AgentStatus, EventMsg};
use serde::Serialize;
use serde_json::json;

use crate::goals::ANYHARNESS_META_KEY;

pub(crate) const ANYHARNESS_SCHEMA_VERSION: u32 = 1;

// Transcript event names for up-channel notifications (task a/b/c wire tags).
pub(crate) const PROCESS_UPSERTED_EVENT: &str = "process_upserted";
pub(crate) const SUBAGENT_UPSERTED_EVENT: &str = "subagent_upserted";
/// Feed-scoped envelope carrying a raw child-thread `EventMsg`. Never a
/// roster mutation itself -- purely a transport for the demuxed child feed
/// (see `child_feed_event_update`).
pub(crate) const CHILD_DEMUX_EVENT: &str = "acp_child_demux";

pub(crate) const PROCESS_STATUS_RUNNING: &str = "running";
pub(crate) const PROCESS_STATUS_EXITED: &str = "exited";

pub(crate) const SUBAGENT_STATUS_RUNNING: &str = "running";
pub(crate) const SUBAGENT_STATUS_COMPLETED: &str = "completed";
pub(crate) const SUBAGENT_STATUS_FAILED: &str = "failed";

/// Wire shape for `ActivityProcess`, built from codex's `commandExecution`
/// item lifecycle (`ExecCommandBegin`/`ExecCommandEnd`). Codex commands are
/// strictly in-turn (harness-runtime-mechanics.md §4.2 -- "nothing runs
/// between turns"), so any entry still `Running` when a turn ends is a
/// leftover that must be force-exited ("expired") with a null exit code --
/// see `PromptState::expire_active_processes` in `thread.rs`.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProcessWire {
    pub id: String,
    pub command: String,
    pub cwd: Option<String>,
    pub status: &'static str,
    pub exit_code: Option<i32>,
    pub started_at_ms: Option<i64>,
    pub ended_at_ms: Option<i64>,
    pub native: bool,
    pub updated_at_ms: i64,
}

impl ProcessWire {
    pub(crate) fn running(
        id: String,
        command: String,
        cwd: Option<String>,
        started_at_ms: i64,
    ) -> Self {
        Self {
            id,
            command,
            cwd,
            status: PROCESS_STATUS_RUNNING,
            exit_code: None,
            started_at_ms: Some(started_at_ms),
            ended_at_ms: None,
            native: true,
            updated_at_ms: started_at_ms,
        }
    }

    /// `exit_code: None` covers both codex's "no exit code available" case
    /// and the turn-end expiry case, where the command never received an
    /// `ExecCommandEnd` at all.
    pub(crate) fn exited(
        id: String,
        command: String,
        cwd: Option<String>,
        started_at_ms: Option<i64>,
        ended_at_ms: i64,
        exit_code: Option<i32>,
    ) -> Self {
        Self {
            id,
            command,
            cwd,
            status: PROCESS_STATUS_EXITED,
            exit_code,
            started_at_ms,
            ended_at_ms: Some(ended_at_ms),
            native: true,
            updated_at_ms: ended_at_ms,
        }
    }
}

/// Wire shape for `ActivitySubagent`. `status`/`summary` are flattened
/// siblings (mirroring how `GoalWire` flattens Rust enum payloads onto the
/// JSON object rather than nesting a tagged enum) instead of following the
/// pinned Rust contract's `SubagentStatus::Completed { summary }` shape
/// verbatim. The contract's `Failed` variant carries no payload, so
/// `summary` is always null when `status == "failed"`.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SubagentWire {
    /// Codex child `ThreadId`, stringified.
    pub id: String,
    pub agent_type: Option<String>,
    pub description: Option<String>,
    pub model: Option<String>,
    /// Always `false` for codex v1: collab sub-agents run inside a
    /// held-open parent turn (harness-runtime-mechanics.md §4.2) and carry
    /// no `isBackground`-style signal on the wire, unlike Cursor's
    /// `cursor/task`.
    pub background: bool,
    pub status: &'static str,
    pub summary: Option<String>,
    /// Transport descriptor the runtime uses to construct this roster
    /// entry's `FeedRef` -- see harness-runtime-mechanics.md §5/§7 and the
    /// per-harness membrane table in session-activity-architecture.md.
    /// Structured (not the `acp_child_demux:<threadId>` colon-delimited
    /// string) to match anyharness's `feed: Option<FeedTransportWire>`
    /// field on `ActivitySubagentWire` (domains/activity/wire.rs) verbatim;
    /// the colon-delimited string form still exists (`child_feed_transport`)
    /// but is scoped to the internal `acp_child_demux` feed-buffer key on
    /// `child_feed_event_update`, never this roster wire payload.
    pub feed: FeedTransportWire,
    pub native: bool,
    pub updated_at_ms: i64,
}

/// Wire shape for `SubagentWire.feed`. Matches anyharness's
/// `FeedTransportWire::AcpChildDemux` tagged-enum variant exactly
/// (`{"kind": "acp_child_demux", "threadId": "<id>"}` --
/// domains/activity/wire.rs). Codex only ever produces this one transport
/// kind for subagents (harness-runtime-mechanics.md §5/§7: "acp_child_demux
/// (subagents); none needed for terminals (pure wire)"), so this type
/// intentionally models only that variant rather than replicating
/// anyharness's full `tail_file | acp_child_demux | http_sse` enum --
/// codex-acp has no code path that could ever construct the other two, and
/// unused variants would be flagged dead code.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum FeedTransportWire {
    AcpChildDemux {
        #[serde(rename = "threadId")]
        thread_id: String,
    },
}

pub(crate) fn process_notification_update(process: &ProcessWire) -> SessionUpdate {
    tagged_update(json!({
        "schemaVersion": ANYHARNESS_SCHEMA_VERSION,
        "transcriptEvent": PROCESS_UPSERTED_EVENT,
        "process": process,
    }))
}

pub(crate) fn subagent_notification_update(subagent: &SubagentWire) -> SessionUpdate {
    tagged_update(json!({
        "schemaVersion": ANYHARNESS_SCHEMA_VERSION,
        "transcriptEvent": SUBAGENT_UPSERTED_EVENT,
        "subagent": subagent,
    }))
}

/// Feed-scoped envelope for a raw child-thread `EventMsg`, kept out of the
/// parent transcript entirely (it never goes through the per-type
/// `PromptState::handle_event` dispatch -- see `thread.rs`'s child feed
/// pump). The runtime buffers these per `feedTransport` id and only
/// replays/parses them while a client has the corresponding subagent feed
/// open (session-activity-architecture.md: "no byte flow unless a panel is
/// open").
pub(crate) fn child_feed_event_update(thread_id: &str, event: &EventMsg) -> SessionUpdate {
    tagged_update(json!({
        "schemaVersion": ANYHARNESS_SCHEMA_VERSION,
        "transcriptEvent": CHILD_DEMUX_EVENT,
        "feedTransport": child_feed_transport(thread_id),
        "threadId": thread_id,
        "event": event,
    }))
}

pub(crate) fn child_feed_transport(thread_id: &str) -> String {
    format!("acp_child_demux:{thread_id}")
}

/// Builds the structured `SubagentWire.feed` value for a child thread id.
/// See `FeedTransportWire`'s doc comment for why this is a single-variant
/// type rather than a colon-delimited string like `child_feed_transport`.
pub(crate) fn child_feed_wire(thread_id: &str) -> FeedTransportWire {
    FeedTransportWire::AcpChildDemux {
        thread_id: thread_id.to_string(),
    }
}

fn tagged_update(payload: serde_json::Value) -> SessionUpdate {
    SessionUpdate::AgentMessageChunk(
        ContentChunk::new("".into())
            .meta(Meta::from_iter([(ANYHARNESS_META_KEY.to_string(), payload)])),
    )
}

/// Map a codex `AgentStatus` onto the subagent roster's normalized status +
/// optional completion summary. `Failed` carries no payload per the pinned
/// `ActivitySubagent` contract, so any terminal-detail text (an errored
/// agent's message) is intentionally dropped here -- surfacing it would
/// need a wire change (an extra field, mirroring `GoalWire::nativeStatus`)
/// that's out of scope for this task.
pub(crate) fn map_agent_status(status: &AgentStatus) -> (&'static str, Option<String>) {
    match status {
        AgentStatus::PendingInit | AgentStatus::Running | AgentStatus::Interrupted => {
            (SUBAGENT_STATUS_RUNNING, None)
        }
        AgentStatus::Completed(summary) => (SUBAGENT_STATUS_COMPLETED, summary.clone()),
        // An explicit close/shutdown (the `closeAgent` terminal case) is a
        // normal terminal, not a failure.
        AgentStatus::Shutdown => (SUBAGENT_STATUS_COMPLETED, None),
        AgentStatus::Errored(_) | AgentStatus::NotFound => (SUBAGENT_STATUS_FAILED, None),
    }
}

/// Truncate a spawn prompt for the roster's `description` field so a large
/// prompt doesn't balloon every upsert. Codex has no existing truncation
/// helper for this in the fork; chosen to keep a reasonable single-line
/// preview length, matching the order of magnitude used for tool-call
/// titles elsewhere in `thread.rs`.
pub(crate) fn truncate_description(prompt: &str) -> Option<String> {
    const MAX_CHARS: usize = 240;
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut truncated: String = trimmed.chars().take(MAX_CHARS).collect();
    if trimmed.chars().count() > MAX_CHARS {
        truncated.push('\u{2026}'); // …
    }
    Some(truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_meta(update: &SessionUpdate) -> serde_json::Value {
        match update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                let meta = chunk.meta.as_ref().expect("chunk should carry _meta");
                meta.get(ANYHARNESS_META_KEY)
                    .expect("anyharness key present")
                    .clone()
            }
            other => panic!("unexpected update variant: {other:?}"),
        }
    }

    #[test]
    fn process_wire_running_then_exited_serializes_camel_case() {
        let running = ProcessWire::running(
            "call-1".to_string(),
            "sleep 30 && echo OK".to_string(),
            Some("/work".to_string()),
            1_000,
        );
        let value = serde_json::to_value(&running).unwrap();
        assert_eq!(
            value,
            json!({
                "id": "call-1",
                "command": "sleep 30 && echo OK",
                "cwd": "/work",
                "status": "running",
                "exitCode": null,
                "startedAtMs": 1000,
                "endedAtMs": null,
                "native": true,
                "updatedAtMs": 1000,
            })
        );

        let exited = ProcessWire::exited(
            "call-1".to_string(),
            "sleep 30 && echo OK".to_string(),
            Some("/work".to_string()),
            Some(1_000),
            2_500,
            Some(0),
        );
        let value = serde_json::to_value(&exited).unwrap();
        assert_eq!(value["status"], json!("exited"));
        assert_eq!(value["exitCode"], json!(0));
        assert_eq!(value["endedAtMs"], json!(2500));
    }

    #[test]
    fn process_notification_update_tags_transcript_event_and_excludes_transcript() {
        let process = ProcessWire::running("c".to_string(), "ls".to_string(), None, 1);
        let update = process_notification_update(&process);
        let meta = extract_meta(&update);
        assert_eq!(meta["transcriptEvent"], json!(PROCESS_UPSERTED_EVENT));
        assert_eq!(meta["process"]["id"], json!("c"));
    }

    #[test]
    fn map_agent_status_running_variants() {
        assert_eq!(
            map_agent_status(&AgentStatus::PendingInit),
            (SUBAGENT_STATUS_RUNNING, None)
        );
        assert_eq!(
            map_agent_status(&AgentStatus::Running),
            (SUBAGENT_STATUS_RUNNING, None)
        );
        assert_eq!(
            map_agent_status(&AgentStatus::Interrupted),
            (SUBAGENT_STATUS_RUNNING, None)
        );
    }

    #[test]
    fn map_agent_status_completed_carries_summary() {
        assert_eq!(
            map_agent_status(&AgentStatus::Completed(Some("did the thing".to_string()))),
            (
                SUBAGENT_STATUS_COMPLETED,
                Some("did the thing".to_string())
            )
        );
        assert_eq!(
            map_agent_status(&AgentStatus::Completed(None)),
            (SUBAGENT_STATUS_COMPLETED, None)
        );
    }

    #[test]
    fn map_agent_status_shutdown_is_completed_not_failed() {
        assert_eq!(
            map_agent_status(&AgentStatus::Shutdown),
            (SUBAGENT_STATUS_COMPLETED, None)
        );
    }

    #[test]
    fn map_agent_status_errored_and_not_found_drop_message_per_contract() {
        assert_eq!(
            map_agent_status(&AgentStatus::Errored("boom".to_string())),
            (SUBAGENT_STATUS_FAILED, None)
        );
        assert_eq!(
            map_agent_status(&AgentStatus::NotFound),
            (SUBAGENT_STATUS_FAILED, None)
        );
    }

    #[test]
    fn subagent_wire_serializes_camel_case_with_nulls() {
        let wire = SubagentWire {
            id: "thread-2".to_string(),
            agent_type: None,
            description: None,
            model: Some("gpt-5.5".to_string()),
            background: false,
            status: SUBAGENT_STATUS_RUNNING,
            summary: None,
            feed: child_feed_wire("thread-2"),
            native: true,
            updated_at_ms: 42,
        };
        let value = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            value,
            json!({
                "id": "thread-2",
                "agentType": null,
                "description": null,
                "model": "gpt-5.5",
                "background": false,
                "status": "running",
                "summary": null,
                "feed": { "kind": "acp_child_demux", "threadId": "thread-2" },
                "native": true,
                "updatedAtMs": 42,
            })
        );
    }

    #[test]
    fn subagent_notification_update_tags_transcript_event() {
        let wire = SubagentWire {
            id: "thread-3".to_string(),
            agent_type: Some("reviewer".to_string()),
            description: Some("review the diff".to_string()),
            model: Some("gpt-5.5".to_string()),
            background: false,
            status: SUBAGENT_STATUS_COMPLETED,
            summary: Some("looks good".to_string()),
            feed: child_feed_wire("thread-3"),
            native: true,
            updated_at_ms: 99,
        };
        let update = subagent_notification_update(&wire);
        let meta = extract_meta(&update);
        assert_eq!(meta["transcriptEvent"], json!(SUBAGENT_UPSERTED_EVENT));
        assert_eq!(meta["subagent"]["id"], json!("thread-3"));
        assert_eq!(meta["subagent"]["summary"], json!("looks good"));
        assert_eq!(
            meta["subagent"]["feed"],
            json!({ "kind": "acp_child_demux", "threadId": "thread-3" })
        );
    }

    #[test]
    fn subagent_feed_wire_matches_anyharness_tagged_shape() {
        // Structural round-trip proof for the finding this fixes: the wire
        // must be a `{kind, threadId}` object under the `feed` key, not a
        // colon-delimited string under `feedTransport` -- see
        // `FeedTransportWire`'s doc comment.
        let value = serde_json::to_value(child_feed_wire("thread-9")).unwrap();
        assert_eq!(
            value,
            json!({ "kind": "acp_child_demux", "threadId": "thread-9" })
        );
    }

    #[test]
    fn child_feed_transport_matches_documented_form() {
        assert_eq!(
            child_feed_transport("018f-abc"),
            "acp_child_demux:018f-abc"
        );
    }

    #[test]
    fn child_feed_event_update_wraps_raw_event_and_never_looks_like_transcript() {
        let event = EventMsg::AgentMessage(codex_protocol::protocol::AgentMessageEvent {
            message: "hello from child".to_string(),
            phase: None,
            memory_citation: None,
        });
        let update = child_feed_event_update("child-1", &event);
        match &update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                // The visible content must stay empty -- all payload travels
                // in `_meta`, exactly like goal/loop notifications, so a
                // membrane that doesn't know about `acp_child_demux` still
                // renders nothing instead of leaking child text.
                match &chunk.content {
                    agent_client_protocol::schema::ContentBlock::Text(text) => {
                        assert_eq!(text.text, "");
                    }
                    other => panic!("unexpected content block: {other:?}"),
                }
            }
            other => panic!("unexpected update variant: {other:?}"),
        }
        let meta = extract_meta(&update);
        assert_eq!(meta["transcriptEvent"], json!(CHILD_DEMUX_EVENT));
        assert_eq!(meta["threadId"], json!("child-1"));
        assert_eq!(meta["feedTransport"], json!("acp_child_demux:child-1"));
        assert_eq!(meta["event"]["type"], json!("agent_message"));
    }

    #[test]
    fn truncate_description_passes_short_prompts_through() {
        assert_eq!(
            truncate_description("review the diff"),
            Some("review the diff".to_string())
        );
        assert_eq!(truncate_description("   "), None);
        assert_eq!(truncate_description(""), None);
    }

    #[test]
    fn truncate_description_truncates_long_prompts() {
        let long = "a".repeat(400);
        let truncated = truncate_description(&long).unwrap();
        assert_eq!(truncated.chars().count(), 241); // 240 + ellipsis
        assert!(truncated.ends_with('\u{2026}'));
    }
}
