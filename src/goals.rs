//! Anyharness GoalPort wire types (contract v1).
//!
//! Implements the pinned GoalPort wire contract:
//! - normalized `GoalWire` shape returned from ext methods and carried on
//!   tagged `AgentMessageChunk` notifications,
//! - the `_anyharness/goal/*` ext-method request types, and
//! - helpers for building the `_meta.anyharness` payloads.

use agent_client_protocol::{
    self as acp, Error, JsonRpcMessage, JsonRpcRequest, UntypedMessage,
    schema::{ContentChunk, Meta, SessionUpdate},
};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// `_meta` key under which all anyharness payloads are nested.
pub(crate) const ANYHARNESS_META_KEY: &str = "anyharness";
/// Version of the anyharness wire contract implemented here.
pub(crate) const ANYHARNESS_SCHEMA_VERSION: u32 = 1;

// Wire method names. ACP requires inbound extension methods to be
// `_`-prefixed on the wire; we register a typed request that matches the raw
// wire names directly.
pub(crate) const GOAL_SET_WIRE_METHOD: &str = "_anyharness/goal/set";
pub(crate) const GOAL_GET_WIRE_METHOD: &str = "_anyharness/goal/get";
pub(crate) const GOAL_CLEAR_WIRE_METHOD: &str = "_anyharness/goal/clear";

// Transcript event names for up-channel notifications.
pub(crate) const GOAL_UPDATED_EVENT: &str = "goal_updated";
pub(crate) const GOAL_MET_EVENT: &str = "goal_met";
pub(crate) const GOAL_CLEARED_EVENT: &str = "goal_cleared";

/// Normalized goal statuses per the wire contract.
pub(crate) const GOAL_STATUS_ACTIVE: &str = "active";
pub(crate) const GOAL_STATUS_PAUSED: &str = "paused";
pub(crate) const GOAL_STATUS_BLOCKED: &str = "blocked";
pub(crate) const GOAL_STATUS_MET: &str = "met";
pub(crate) const GOAL_STATUS_FAILED: &str = "failed";

/// Normalized goal shape shared by ext-method results and notifications.
///
/// All fields serialize (nullable fields as `null`) so clients can rely on a
/// stable shape.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GoalWire {
    pub objective: String,
    /// Normalized status: active|paused|blocked|met|failed|cleared.
    pub status: &'static str,
    /// Raw codex status string, verbatim (camelCase, e.g. "usageLimited").
    pub native_status: &'static str,
    pub token_budget: Option<i64>,
    pub tokens_used: Option<i64>,
    pub time_used_seconds: Option<i64>,
    /// Codex has no terminal-detail reason; always null here (the detail is
    /// preserved in `nativeStatus`).
    pub met_reason: Option<String>,
    /// Claude-only; always null for codex.
    pub iterations: Option<i64>,
    pub native: bool,
    pub updated_at_ms: i64,
}

/// (normalized status, verbatim native status) for a codex goal status.
fn normalize_state_status(status: codex_state::ThreadGoalStatus) -> (&'static str, &'static str) {
    use codex_state::ThreadGoalStatus as S;
    match status {
        S::Active => (GOAL_STATUS_ACTIVE, "active"),
        S::Paused => (GOAL_STATUS_PAUSED, "paused"),
        S::Blocked => (GOAL_STATUS_BLOCKED, "blocked"),
        S::UsageLimited => (GOAL_STATUS_FAILED, "usageLimited"),
        S::BudgetLimited => (GOAL_STATUS_FAILED, "budgetLimited"),
        S::Complete => (GOAL_STATUS_MET, "complete"),
    }
}

fn normalize_protocol_status(
    status: codex_protocol::protocol::ThreadGoalStatus,
) -> (&'static str, &'static str) {
    use codex_protocol::protocol::ThreadGoalStatus as S;
    match status {
        S::Active => (GOAL_STATUS_ACTIVE, "active"),
        S::Paused => (GOAL_STATUS_PAUSED, "paused"),
        S::Blocked => (GOAL_STATUS_BLOCKED, "blocked"),
        S::UsageLimited => (GOAL_STATUS_FAILED, "usageLimited"),
        S::BudgetLimited => (GOAL_STATUS_FAILED, "budgetLimited"),
        S::Complete => (GOAL_STATUS_MET, "complete"),
    }
}

impl GoalWire {
    /// Build from a state-db goal row (ext-method path).
    pub(crate) fn from_state(goal: &codex_state::ThreadGoal) -> Self {
        let (status, native_status) = normalize_state_status(goal.status);
        Self {
            objective: goal.objective.clone(),
            status,
            native_status,
            token_budget: goal.token_budget,
            tokens_used: Some(goal.tokens_used),
            time_used_seconds: Some(goal.time_used_seconds),
            met_reason: None,
            iterations: None,
            native: true,
            updated_at_ms: goal.updated_at.timestamp_millis(),
        }
    }

    /// Build from a protocol goal (EventMsg::ThreadGoalUpdated path). Protocol
    /// timestamps are epoch seconds.
    pub(crate) fn from_protocol(goal: &codex_protocol::protocol::ThreadGoal) -> Self {
        let (status, native_status) = normalize_protocol_status(goal.status);
        Self {
            objective: goal.objective.clone(),
            status,
            native_status,
            token_budget: goal.token_budget,
            tokens_used: Some(goal.tokens_used),
            time_used_seconds: Some(goal.time_used_seconds),
            met_reason: None,
            iterations: None,
            native: true,
            updated_at_ms: goal.updated_at.saturating_mul(1000),
        }
    }

    /// Notification event name for this goal snapshot: `goal_met` when the
    /// normalized status maps to met, otherwise `goal_updated`.
    pub(crate) fn transcript_event(&self) -> &'static str {
        if self.status == GOAL_STATUS_MET {
            GOAL_MET_EVENT
        } else {
            GOAL_UPDATED_EVENT
        }
    }
}

/// Build the zero-length `AgentMessageChunk` update tagged with
/// `_meta.anyharness` for a goal transcript event. `goal` is omitted for
/// `goal_cleared`.
pub(crate) fn goal_notification_update(
    transcript_event: &str,
    goal: Option<&GoalWire>,
) -> SessionUpdate {
    let mut payload = json!({
        "schemaVersion": ANYHARNESS_SCHEMA_VERSION,
        "transcriptEvent": transcript_event,
    });
    if let Some(goal) = goal {
        payload["goal"] = serde_json::to_value(goal).unwrap_or(serde_json::Value::Null);
    }
    SessionUpdate::AgentMessageChunk(ContentChunk::new("".into()).meta(Meta::from_iter([(
        ANYHARNESS_META_KEY.to_string(),
        payload,
    )])))
}

/// `_meta.anyharness` value advertised on the initialize response.
pub(crate) fn initialize_capability_meta() -> Meta {
    Meta::from_iter([(
        ANYHARNESS_META_KEY.to_string(),
        json!({
            "schemaVersion": ANYHARNESS_SCHEMA_VERSION,
            "goals": { "supported": true, "native": true },
        }),
    )])
}

// --- Ext-method request types ---

/// Requested goal status on `goal/set`. The contract only allows arming
/// (`active`) or pausing (`paused`) from the outside.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum GoalSetStatusParam {
    Active,
    Paused,
}

impl GoalSetStatusParam {
    pub(crate) fn to_state(self) -> codex_state::ThreadGoalStatus {
        match self {
            GoalSetStatusParam::Active => codex_state::ThreadGoalStatus::Active,
            GoalSetStatusParam::Paused => codex_state::ThreadGoalStatus::Paused,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GoalSetParams {
    pub session_id: String,
    /// Omitted objective = status/budget-only patch (codex semantics).
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub status: Option<GoalSetStatusParam>,
    /// Absent = keep, `null` = clear the budget, number = set it.
    #[serde(default, deserialize_with = "deserialize_double_option")]
    pub token_budget: Option<Option<i64>>,
}

/// Deserialize a present field (including an explicit `null`) into
/// `Some(inner)` so absent (`None`) and `null` (`Some(None)`) stay distinct.
fn deserialize_double_option<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GoalSessionParams {
    pub session_id: String,
}

/// Typed request covering the three `_anyharness/goal/*` wire methods.
///
/// Registered directly on the ACP builder; `matches_method` sees the raw wire
/// method names so unrelated ext methods fall through to other handlers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum AnyharnessGoalRequest {
    Set(GoalSetParams),
    Get(GoalSessionParams),
    Clear(GoalSessionParams),
}

impl JsonRpcMessage for AnyharnessGoalRequest {
    fn matches_method(method: &str) -> bool {
        matches!(
            method,
            GOAL_SET_WIRE_METHOD | GOAL_GET_WIRE_METHOD | GOAL_CLEAR_WIRE_METHOD
        )
    }

    fn method(&self) -> &str {
        match self {
            AnyharnessGoalRequest::Set(_) => GOAL_SET_WIRE_METHOD,
            AnyharnessGoalRequest::Get(_) => GOAL_GET_WIRE_METHOD,
            AnyharnessGoalRequest::Clear(_) => GOAL_CLEAR_WIRE_METHOD,
        }
    }

    fn to_untyped_message(&self) -> Result<UntypedMessage, Error> {
        UntypedMessage::new(self.method(), self)
    }

    fn parse_message(method: &str, params: &impl Serialize) -> Result<Self, Error> {
        match method {
            GOAL_SET_WIRE_METHOD => acp::util::json_cast_params(params).map(Self::Set),
            GOAL_GET_WIRE_METHOD => acp::util::json_cast_params(params).map(Self::Get),
            GOAL_CLEAR_WIRE_METHOD => acp::util::json_cast_params(params).map(Self::Clear),
            _ => Err(Error::method_not_found()),
        }
    }
}

impl JsonRpcRequest for AnyharnessGoalRequest {
    type Response = serde_json::Value;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_set_params_distinguish_absent_and_null_budget() {
        let absent: GoalSetParams =
            serde_json::from_str(r#"{"sessionId":"t","objective":"o"}"#).unwrap();
        assert_eq!(absent.token_budget, None);

        let cleared: GoalSetParams =
            serde_json::from_str(r#"{"sessionId":"t","tokenBudget":null}"#).unwrap();
        assert_eq!(cleared.token_budget, Some(None));

        let set: GoalSetParams =
            serde_json::from_str(r#"{"sessionId":"t","tokenBudget":50000}"#).unwrap();
        assert_eq!(set.token_budget, Some(Some(50000)));
    }

    #[test]
    fn status_normalization_matches_contract() {
        use codex_state::ThreadGoalStatus as S;
        assert_eq!(normalize_state_status(S::Active), ("active", "active"));
        assert_eq!(normalize_state_status(S::Paused), ("paused", "paused"));
        assert_eq!(normalize_state_status(S::Blocked), ("blocked", "blocked"));
        assert_eq!(
            normalize_state_status(S::UsageLimited),
            ("failed", "usageLimited")
        );
        assert_eq!(
            normalize_state_status(S::BudgetLimited),
            ("failed", "budgetLimited")
        );
        assert_eq!(normalize_state_status(S::Complete), ("met", "complete"));
    }

    #[test]
    fn goal_wire_serializes_camel_case_with_nulls() {
        let wire = GoalWire {
            objective: "probe".to_string(),
            status: GOAL_STATUS_PAUSED,
            native_status: "paused",
            token_budget: None,
            tokens_used: Some(0),
            time_used_seconds: Some(0),
            met_reason: None,
            iterations: None,
            native: true,
            updated_at_ms: 1234,
        };
        let value = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "objective": "probe",
                "status": "paused",
                "nativeStatus": "paused",
                "tokenBudget": null,
                "tokensUsed": 0,
                "timeUsedSeconds": 0,
                "metReason": null,
                "iterations": null,
                "native": true,
                "updatedAtMs": 1234,
            })
        );
    }

    #[test]
    fn wire_methods_match_and_parse() {
        assert!(AnyharnessGoalRequest::matches_method("_anyharness/goal/set"));
        assert!(AnyharnessGoalRequest::matches_method("_anyharness/goal/get"));
        assert!(AnyharnessGoalRequest::matches_method(
            "_anyharness/goal/clear"
        ));
        assert!(!AnyharnessGoalRequest::matches_method("anyharness/goal/set"));
        assert!(!AnyharnessGoalRequest::matches_method("session/prompt"));

        let parsed = AnyharnessGoalRequest::parse_message(
            "_anyharness/goal/set",
            &serde_json::json!({"sessionId": "abc", "objective": "probe", "status": "paused"}),
        )
        .unwrap();
        match parsed {
            AnyharnessGoalRequest::Set(params) => {
                assert_eq!(params.session_id, "abc");
                assert_eq!(params.objective.as_deref(), Some("probe"));
                assert!(matches!(params.status, Some(GoalSetStatusParam::Paused)));
            }
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    /// Task (d): codex-acp does not implement `LoopPort` -- loops are
    /// runtime-emulated by anyharness for codex (per the pinned wire
    /// contract's codex-acp integration addendum). Locking that the
    /// capability stays absent (not merely `false`) so a stale membrane on
    /// the anyharness side degrades to "unsupported" rather than seeing a
    /// key it doesn't expect.
    #[test]
    fn initialize_capability_meta_advertises_goals_only_no_loops() {
        let meta = initialize_capability_meta();
        let anyharness = meta
            .get(ANYHARNESS_META_KEY)
            .expect("anyharness key present");
        assert!(anyharness.get("goals").is_some());
        assert!(
            anyharness.get("loops").is_none(),
            "codex-acp must not advertise loops support: {anyharness:?}"
        );
        assert_eq!(anyharness["goals"]["native"], serde_json::json!(true));
    }
}
