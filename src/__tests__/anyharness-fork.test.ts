import { describe, it, expect } from "vitest";
import { RequestError } from "@agentclientprotocol/sdk";
import {
  ANYHARNESS_TARGETED_FORK_CAPABILITY_META,
  forkAnchorLastTurnId,
} from "../anyharness-fork";

describe("forkAnchorLastTurnId", () => {
  it("returns undefined when no meta is present (tip fork)", () => {
    expect(forkAnchorLastTurnId(undefined)).toBeUndefined();
    expect(forkAnchorLastTurnId(null)).toBeUndefined();
    expect(forkAnchorLastTurnId({})).toBeUndefined();
    expect(forkAnchorLastTurnId({ anyharness: {} })).toBeUndefined();
    expect(forkAnchorLastTurnId({ anyharness: { lastTurnId: null } })).toBeUndefined();
    expect(forkAnchorLastTurnId("not-an-object")).toBeUndefined();
  });

  it("returns the anchor when present and well-formed", () => {
    expect(
      forkAnchorLastTurnId({ anyharness: { lastTurnId: "turn-abc" } }),
    ).toBe("turn-abc");
  });

  it("throws invalidParams on a malformed anchor, never a silent tip fork", () => {
    for (const malformed of ["", 42, {}, [], true]) {
      expect(() =>
        forkAnchorLastTurnId({ anyharness: { lastTurnId: malformed } }),
      ).toThrow(RequestError);
    }
    try {
      forkAnchorLastTurnId({ anyharness: { lastTurnId: "" } });
      expect.unreachable("empty-string anchor must throw");
    } catch (error) {
      expect((error as RequestError).code).toBe(-32602);
    }
  });
});

describe("ANYHARNESS_TARGETED_FORK_CAPABILITY_META", () => {
  it("pins the versioned advertisement shape (runtime probe wire contract)", () => {
    expect(ANYHARNESS_TARGETED_FORK_CAPABILITY_META).toEqual({
      anyharness: {
        schemaVersion: 1,
        targetedFork: {
          fileEffects: "none",
          target: "turn_id",
        },
      },
    });
  });
});
