import { describe, it, expect } from "vitest";
import { PassThrough } from "node:stream";
import * as acp from "@agentclientprotocol/sdk";
import { CodexAcpServer } from "../../CodexAcpServer";
import { CodexAcpClient } from "../../CodexAcpClient";
import { CodexAppServerClient } from "../../CodexAppServerClient";
import { createMockConnections } from "./test-utils";
import { createJsonStream } from "../../StdUtils";
import { ANYHARNESS_TARGETED_FORK_CAPABILITY_META } from "../../anyharness-fork";

/** Wire-contract tests for the versioned targeted-fork capability
 *  advertisement (`sessionCapabilities.fork._meta.anyharness`).
 *
 *  Two consumers are pinned here:
 *  1. The raw ndjson wire: the AnyHarness runtime reads the initialize
 *     response through its pinned Rust ACP client, so the `_meta` must
 *     survive the JS SDK's serialization of the initialize result byte-level
 *     (the rung-1 JS-SDK-vs-Rust-client interop qualification, applied to
 *     this bridge).
 *  2. The runtime probe `has_anyharness_targeted_fork_extension`
 *     (anyharness-lib/src/live/sessions/driver/native_session.rs), mirrored
 *     below field-for-field. The probe's accepted `target` values gain
 *     "turn_id" in the runtime wire-in lane (ruled Q-B1); this suite pins the
 *     shape the adapter ships. */

/** Field-for-field TS mirror of the Rust runtime probe
 *  `has_anyharness_targeted_fork_extension`, with the Q-B1-ruled "turn_id"
 *  member included in the accepted target set. */
function hasAnyharnessTargetedForkExtension(
  meta: unknown,
  acceptedTargets: readonly string[],
): boolean {
  if (!meta || typeof meta !== "object") return false;
  const anyharness = (meta as Record<string, unknown>)["anyharness"];
  if (!anyharness || typeof anyharness !== "object") return false;
  const ns = anyharness as Record<string, unknown>;
  if (ns["schemaVersion"] !== 1) return false;
  const targetedFork = ns["targetedFork"];
  if (!targetedFork || typeof targetedFork !== "object") return false;
  const fork = targetedFork as Record<string, unknown>;
  if (fork["fileEffects"] !== "none") return false;
  return (
    typeof fork["target"] === "string" && acceptedTargets.includes(fork["target"])
  );
}

async function initializeOverWire(): Promise<{
  rawResult: Record<string, unknown>;
}> {
  const clientToAgent = new PassThrough();
  const agentToClient = new PassThrough();

  const mocks = createMockConnections();
  const codexAppServerClient = new CodexAppServerClient(mocks.mockCodexConnection);
  const codexAcpClient = new CodexAcpClient(codexAppServerClient);

  acp
    .agent({ name: "codex-acp-test" })
    .onConnect((connection) => {
      void connection;
    })
    .onRequest(acp.methods.agent.initialize, (ctx) =>
      new CodexAcpServer(mocks.mockAcpConnection, codexAcpClient).initialize(ctx.params),
    )
    .connect(createJsonStream(clientToAgent, agentToClient));

  const request = {
    jsonrpc: "2.0",
    id: 1,
    method: acp.methods.agent.initialize,
    params: { protocolVersion: acp.PROTOCOL_VERSION },
  };

  const rawResponse = new Promise<Record<string, unknown>>((resolve, reject) => {
    let buffered = "";
    agentToClient.on("data", (chunk: Buffer) => {
      buffered += chunk.toString("utf8");
      const newlineIndex = buffered.indexOf("\n");
      if (newlineIndex >= 0) {
        resolve(JSON.parse(buffered.slice(0, newlineIndex)));
      }
    });
    agentToClient.on("error", reject);
    setTimeout(() => reject(new Error("initialize response timeout")), 5_000);
  });

  clientToAgent.write(JSON.stringify(request) + "\n");
  const response = await rawResponse;
  expect(response["id"]).toBe(1);
  expect(response["error"]).toBeUndefined();
  return { rawResult: response["result"] as Record<string, unknown> };
}

function forkCapabilityMetaFrom(rawResult: Record<string, unknown>): unknown {
  const agentCapabilities = rawResult["agentCapabilities"] as Record<string, unknown>;
  const sessionCapabilities = agentCapabilities["sessionCapabilities"] as Record<string, unknown>;
  const fork = sessionCapabilities["fork"] as Record<string, unknown>;
  expect(fork).toBeDefined();
  return fork["_meta"];
}

describe("fork capability advertisement (versioned targetedFork _meta)", () => {
  it("preserves the fork capability _meta on the raw ndjson wire", async () => {
    const { rawResult } = await initializeOverWire();
    const meta = forkCapabilityMetaFrom(rawResult);
    expect(meta).toEqual({
      anyharness: {
        schemaVersion: 1,
        targetedFork: {
          fileEffects: "none",
          target: "turn_id",
        },
      },
    });
  });

  it("satisfies the runtime probe with turn_id in the accepted target set (Q-B1 ruling)", async () => {
    const { rawResult } = await initializeOverWire();
    const meta = forkCapabilityMetaFrom(rawResult);
    expect(
      hasAnyharnessTargetedForkExtension(meta, [
        "message_id",
        "user_message_index",
        "turn_id",
      ]),
    ).toBe(true);
  });

  it("is NOT accepted by the pre-Q-B1 runtime probe enum (documents the Lane D dependency)", async () => {
    const { rawResult } = await initializeOverWire();
    const meta = forkCapabilityMetaFrom(rawResult);
    expect(
      hasAnyharnessTargetedForkExtension(meta, ["message_id", "user_message_index"]),
    ).toBe(false);
  });

  it("negative controls: the probe mirror rejects near-miss shapes", () => {
    const base = () =>
      JSON.parse(JSON.stringify(ANYHARNESS_TARGETED_FORK_CAPABILITY_META)) as {
        anyharness: {
          schemaVersion: number;
          targetedFork: { fileEffects: string; target: string };
        };
      };
    const accepted = ["message_id", "user_message_index", "turn_id"];

    expect(hasAnyharnessTargetedForkExtension(base(), accepted)).toBe(true);

    const wrongVersion = base();
    wrongVersion.anyharness.schemaVersion = 2;
    expect(hasAnyharnessTargetedForkExtension(wrongVersion, accepted)).toBe(false);

    const wrongFileEffects = base();
    wrongFileEffects.anyharness.targetedFork.fileEffects = "workspace";
    expect(hasAnyharnessTargetedForkExtension(wrongFileEffects, accepted)).toBe(false);

    const wrongTarget = base();
    wrongTarget.anyharness.targetedFork.target = "message_index";
    expect(hasAnyharnessTargetedForkExtension(wrongTarget, accepted)).toBe(false);

    expect(hasAnyharnessTargetedForkExtension({}, accepted)).toBe(false);
    expect(hasAnyharnessTargetedForkExtension(undefined, accepted)).toBe(false);
  });
});
