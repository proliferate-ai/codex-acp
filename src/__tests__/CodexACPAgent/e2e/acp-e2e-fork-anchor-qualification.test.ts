import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {afterAll, beforeAll, expect, it} from "vitest";
import {CodexAppServerClient} from "../../../CodexAppServerClient";
import {startCodexConnection, type CodexConnection} from "../../../CodexJsonRpcConnection";
import {removeDirectoryWithRetry, writeCodexHomeConfig} from "../../acp-test-utils";
import {describeE2E, DEFAULT_TEST_MODEL_ID} from "./acp-e2e-test-utils";
import type {Turn} from "../../../app-server/v2/Turn";
import type {ThreadItem} from "../../../app-server/v2/ThreadItem";

/**
 * Qualification fixture pinning upstream Codex App Server (pinned
 * `@openai/codex` 0.147.0) `thread/fork` anchor behavior, driven directly at
 * the App Server JSON-RPC level (not through the ACP adapter). This exists to
 * settle Forks ADR open questions Q-B2 (does each committed user submission
 * open a new native turn, and is steering the sole counterexample?) and Q-B3
 * (does an anchored fork on a valid `lastTurnId` copy exactly the inclusive
 * prefix, and does an unknown/stale `lastTurnId` avoid a silent full copy?).
 *
 * This suite requires live OpenAI credentials and runs real model turns; it
 * is gated behind RUN_E2E_TESTS and is not intended to run in ordinary CI.
 */

const NO_TOOLS_SUFFIX = " Do not use any tools.";

// ChatGPT-account auth only serves the -codex model line; allow overriding
// the default API-key-account model when running under CODEX_AUTH_JSON_PATH.
const TEST_MODEL = process.env["CODEX_TEST_MODEL"] ?? DEFAULT_TEST_MODEL_ID.model;
const TEST_EFFORT = process.env["CODEX_TEST_EFFORT"] ?? DEFAULT_TEST_MODEL_ID.effort;

function isUserMessage(item: ThreadItem): boolean {
    return item.type === "userMessage";
}

function countUserMessages(turn: Turn): number {
    return turn.items.filter(isUserMessage).length;
}

describeE2E("Fork anchor qualification (pinned upstream @openai/codex 0.147.0)", () => {
    let codexHome: string;
    let workspaceDir: string;
    let rootDir: string;
    let connectionHandle: CodexConnection;
    let client: CodexAppServerClient;
    let threadId: string;
    let firstTurnId: string;
    let secondTurnId: string;
    let thirdTurnId: string;

    beforeAll(async () => {
        rootDir = fs.mkdtempSync(path.join(os.tmpdir(), "codex-acp-fork-anchor-qual-"));
        codexHome = path.join(rootDir, "codex-home");
        workspaceDir = path.join(rootDir, "workspace");
        fs.mkdirSync(codexHome, {recursive: true});
        fs.mkdirSync(workspaceDir, {recursive: true});
        writeCodexHomeConfig(codexHome, {
            model: TEST_MODEL,
            model_reasoning_effort: TEST_EFFORT,
            web_search: "disabled",
        });

        const seedAuthJsonPath = process.env["CODEX_AUTH_JSON_PATH"];
        if (seedAuthJsonPath) {
            fs.copyFileSync(seedAuthJsonPath, path.join(codexHome, "auth.json"));
            fs.chmodSync(path.join(codexHome, "auth.json"), 0o600);
        }

        connectionHandle = startCodexConnection(undefined, {
            ...process.env,
            CODEX_HOME: codexHome,
        });
        client = new CodexAppServerClient(connectionHandle.connection);

        await client.initialize({
            clientInfo: {
                name: "codex-acp-fork-anchor-qualification",
                title: "Codex ACP Fork Anchor Qualification",
                version: "1.0.0",
            },
            capabilities: {
                experimentalApi: true,
                requestAttestation: false,
            },
        });

        // The bundled app server does not read OPENAI_API_KEY from the
        // environment on its own. Authenticate either from a pre-existing
        // codex auth.json (CODEX_AUTH_JSON_PATH, copied into the isolated
        // CODEX_HOME before use) or through the native account flow the
        // adapter itself uses (account/login/start type=apiKey), polling
        // account/read until the login is effective.
        const authJsonPath = process.env["CODEX_AUTH_JSON_PATH"];
        if (!authJsonPath) {
            const apiKey = process.env["OPENAI_API_KEY"];
            if (!apiKey) {
                throw new Error("OPENAI_API_KEY or CODEX_AUTH_JSON_PATH is required for the fork anchor qualification suite");
            }
            await client.accountLogin({type: "apiKey", apiKey});
        }
        const loginDeadline = Date.now() + 30_000;
        for (;;) {
            const accountResponse = await client.accountRead({refreshToken: false});
            if (accountResponse.account) {
                break;
            }
            if (Date.now() > loginDeadline) {
                throw new Error("codex login did not complete in time");
            }
            await new Promise((resolve) => setTimeout(resolve, 250));
        }

        const threadStartResponse = await client.threadStart({
            cwd: workspaceDir,
            model: TEST_MODEL,
        });
        threadId = threadStartResponse.thread.id;
    }, 60_000);

    afterAll(async () => {
        connectionHandle.process.kill();
        removeDirectoryWithRetry(rootDir);
    });

    it("Q-B2: each committed user submission opens a new native turn", async () => {
        const appleCompletion = await client.runTurn({
            threadId,
            input: [{type: "text", text: `Reply with exactly the word APPLE.${NO_TOOLS_SUFFIX}`, text_elements: []}],
        }, (turnId) => {
            firstTurnId = turnId;
        });
        expect(appleCompletion.turn.error, JSON.stringify(appleCompletion.turn.error)).toBeNull();
        expect(appleCompletion.turn.status).toBe("completed");
        expect(appleCompletion.turn.id).toBe(firstTurnId);

        const bananaCompletion = await client.runTurn({
            threadId,
            input: [{type: "text", text: `Reply with exactly the word BANANA.${NO_TOOLS_SUFFIX}`, text_elements: []}],
        }, (turnId) => {
            secondTurnId = turnId;
        });
        expect(bananaCompletion.turn.status).toBe("completed");
        expect(bananaCompletion.turn.id).toBe(secondTurnId);

        const readResponse = await client.threadRead({threadId, includeTurns: true});
        const turns = readResponse.thread.turns;

        expect(turns).toHaveLength(2);

        const turnIds = turns.map((turn) => turn.id);
        expect(new Set(turnIds).size).toBe(2);
        expect(turnIds).toEqual([firstTurnId, secondTurnId]);

        for (const turn of turns) {
            expect(countUserMessages(turn)).toBe(1);
        }
    }, 120_000);

    it("Q-B2 counterexample hunt: a steered message lands inside the in-flight turn, not a new one", async () => {
        const countingTurnPromise = client.turnStart({
            threadId,
            input: [{
                type: "text",
                text: `Count slowly from 1 to 30, one number per line.${NO_TOOLS_SUFFIX}`,
                text_elements: [],
            }],
        });

        const countingTurnStarted = await countingTurnPromise;
        thirdTurnId = countingTurnStarted.turn.id;

        let steerLanded = false;
        const steerDeadline = Date.now() + 20_000;
        while (!steerLanded && Date.now() < steerDeadline) {
            try {
                await client.turnSteer({
                    threadId,
                    input: [{type: "text", text: `Actually stop and reply DONE.${NO_TOOLS_SUFFIX}`, text_elements: []}],
                    expectedTurnId: thirdTurnId,
                });
                steerLanded = true;
            } catch {
                await new Promise((resolve) => setTimeout(resolve, 250));
            }
        }

        const completion = await client.awaitTurnCompleted(threadId, thirdTurnId);
        expect(completion.turn.status).toBe("completed");

        const readResponse = await client.threadRead({threadId, includeTurns: true});
        const turns = readResponse.thread.turns;
        expect(turns).toHaveLength(3);

        const thirdTurn = turns.find((turn) => turn.id === thirdTurnId);
        expect(thirdTurn).toBeDefined();

        const userMessageCount = countUserMessages(thirdTurn!);
        console.info(`[qualification] Q-B2 steer landed in-flight: ${steerLanded}; user messages in steered turn: ${userMessageCount}`);
        if (steerLanded) {
            // PINS the counterexample: a mid-flight steer is appended to the
            // SAME native turn rather than opening a new one, so the
            // "each committed submission = one new turn" heuristic in Q-B2
            // is confined to non-steered submissions.
            expect(userMessageCount).toBeGreaterThanOrEqual(2);
        } else {
            // The steer never landed in-flight within the retry budget (the
            // turn likely completed too quickly on this model/prompt). Do
            // not fabricate a counterexample: pin what was actually
            // observed instead.
            expect(userMessageCount).toBeGreaterThanOrEqual(1);
        }
    }, 120_000);

    it("Q-B3: anchored fork on a valid lastTurnId copies exactly the inclusive prefix", async () => {
        const forkResponse = await client.threadFork({
            threadId,
            lastTurnId: firstTurnId,
            cwd: workspaceDir,
        });

        expect(forkResponse.thread.id).not.toBe(threadId);

        const forkedThreadId = forkResponse.thread.id;
        const forkedRead = await client.threadRead({threadId: forkedThreadId, includeTurns: true});
        const forkedTurns = forkedRead.thread.turns;

        expect(forkedTurns).toHaveLength(1);
        expect(forkedTurns[0]!.id).toBe(firstTurnId);

        const forkedText = JSON.stringify(forkedTurns[0]);
        expect(forkedText).not.toContain("BANANA");

        // Second boundary, distinct prefix: anchor on the second turn keeps
        // APPLE and BANANA but drops the steered counting turn.
        const secondBoundaryFork = await client.threadFork({
            threadId,
            lastTurnId: secondTurnId,
            cwd: workspaceDir,
        });
        const secondBoundaryRead = await client.threadRead({
            threadId: secondBoundaryFork.thread.id,
            includeTurns: true,
        });
        expect(secondBoundaryRead.thread.turns.map((turn) => turn.id)).toEqual([firstTurnId, secondTurnId]);
        expect(JSON.stringify(secondBoundaryRead.thread.turns)).not.toContain("Count slowly");

        const tipForkResponse = await client.threadFork({
            threadId,
            cwd: workspaceDir,
        });
        const tipForkedThreadId = tipForkResponse.thread.id;
        const tipForkedRead = await client.threadRead({threadId: tipForkedThreadId, includeTurns: true});

        expect(tipForkedRead.thread.turns).toHaveLength(3);
        const tipForkedTurnIds = tipForkedRead.thread.turns.map((turn) => turn.id);
        expect(tipForkedTurnIds).toEqual(expect.arrayContaining([firstTurnId, secondTurnId, thirdTurnId]));
    }, 60_000);

    it("Q-B3: unknown/stale lastTurnId must NOT silently full-copy", async () => {
        const unknownLastTurnIds = [
            "00000000-0000-0000-0000-000000000000",
            crypto.randomUUID(),
        ];

        for (const unknownLastTurnId of unknownLastTurnIds) {
            let forkResponse;
            let thrown: unknown = null;
            try {
                forkResponse = await client.threadFork({
                    threadId,
                    lastTurnId: unknownLastTurnId,
                    cwd: workspaceDir,
                });
            } catch (error) {
                thrown = error;
            }

            if (thrown !== null) {
                // Pinned: upstream 0.147.0 rejects an unknown/stale
                // lastTurnId outright rather than silently full-copying.
                console.info(`[qualification] Q-B3 unknown lastTurnId ${unknownLastTurnId}: REJECTED (${String(thrown)})`);
                expect(thrown).toBeDefined();
                continue;
            }

            expect(forkResponse).toBeDefined();
            const forkedThreadId = forkResponse!.thread.id;
            const forkedRead = await client.threadRead({threadId: forkedThreadId, includeTurns: true});
            const forkedTurnIds = forkedRead.thread.turns.map((turn) => turn.id);
            const sourceTurnIds = [firstTurnId, secondTurnId, thirdTurnId];
            const containsAllSourceTurns = sourceTurnIds.every((turnId) => forkedTurnIds.includes(turnId));

            // Any non-error outcome on an unknown anchor is a silent failure:
            // a full copy is the Forks ADR §5 cardinal sin (silent tip
            // downgrade) and an empty/partial fork is the same sin in the
            // opposite direction (silent content loss). Either way adapter
            // pre-validation of the turn id before dispatch becomes
            // mandatory.
            console.info(`[qualification] Q-B3 unknown lastTurnId ${unknownLastTurnId}: ACCEPTED with ${forkedTurnIds.length} turns copied (full copy: ${containsAllSourceTurns})`);
            expect.unreachable(
                containsAllSourceTurns
                    ? "SILENT FULL COPY: upstream forked the full thread on an unknown lastTurnId — adapter pre-validation required (Forks ADR §5 cardinal sin)"
                    : `SILENT PARTIAL/EMPTY FORK: upstream accepted an unknown lastTurnId and copied ${forkedTurnIds.length} turns — adapter pre-validation required (silent content loss)`,
            );
        }
    }, 60_000);
});
