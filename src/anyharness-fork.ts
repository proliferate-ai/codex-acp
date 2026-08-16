import { RequestError } from "@agentclientprotocol/sdk";

/** The Proliferate ("AnyHarness") thin delta over canonical `codex-acp`:
 *  registration of the ACP `session/fork` method, mapped onto the native App
 *  Server `thread/fork` (canonical registers `_session/goal` and
 *  `_session/steering` already, but NOT `session/fork`).
 *
 *  The inclusive fork anchor rides in the `session/fork` request
 *  `_meta.anyharness.lastTurnId`. Per the pinned App Server schema
 *  (fixtures/contracts/codex-app-server-schema, native codex 0.147.0),
 *  `ThreadForkParams.lastTurnId` is the ONLY fork anchor — "last turn id to
 *  fork through, inclusive; turns after it are omitted; the referenced turn
 *  cannot be in progress". There is no `beforeTurnId`/`excludeTurns`. The
 *  runtime resolves the product boundary ("immediately before the selected
 *  user message") to the preceding turn id and passes it here.
 *
 *  Absent anchor → an unanchored (tip) fork. A present-but-malformed anchor is
 *  a HARD `invalidParams`, never a silent tip fork (Forks ADR §5 cardinal
 *  sin). */
export const ANYHARNESS_META_NAMESPACE = "anyharness";

export function forkAnchorLastTurnId(meta: unknown): string | undefined {
  if (!meta || typeof meta !== "object") return undefined;
  const ns = (meta as Record<string, unknown>)[ANYHARNESS_META_NAMESPACE];
  if (!ns || typeof ns !== "object") return undefined;
  const anchor = (ns as Record<string, unknown>)["lastTurnId"];
  if (anchor === undefined || anchor === null) return undefined;
  if (typeof anchor !== "string" || anchor.length === 0) {
    throw RequestError.invalidParams(
      undefined,
      "_meta.anyharness.lastTurnId must be a non-empty string when present",
    );
  }
  return anchor;
}
