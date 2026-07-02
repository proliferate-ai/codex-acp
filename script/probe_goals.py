#!/usr/bin/env python3
"""Integration probe for the anyharness GoalPort ext methods.

Launches the built codex-acp binary over stdio (newline-delimited JSON-RPC
2.0, i.e. the ACP transport), then exercises:

  initialize                 -> asserts _meta.anyharness capability
  session/new                -> creates a persisted thread
  _anyharness/goal/set       -> objective "probe", status "paused"
  _anyharness/goal/get       -> returns the stored goal
  _anyharness/goal/set       -> objective edit ("probe v2")
  _anyharness/goal/clear     -> cleared: true
  _anyharness/goal/get       -> goal: null
  _anyharness/goal/get       -> unknown session id errors cleanly

It also asserts that each successful mutation emits the tagged zero-length
AgentMessageChunk notification, and that the goal row lands in the goals
sqlite inside the temp CODEX_HOME.

Usage: python3 script/probe_goals.py [path-to-codex-acp-binary]
"""

import json
import os
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading

TIMEOUT_S = 60


class Probe:
    def __init__(self, binary: str, codex_home: str):
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "CODEX_HOME": codex_home,
                "OPENAI_API_KEY": "sk-probe-key",
                "RUST_LOG": "error",
            },
            text=True,
            bufsize=1,
        )
        self.next_id = 0
        self.responses: "queue.Queue[dict]" = queue.Queue()
        self.notifications: list[dict] = []
        self.notification_lock = threading.Lock()
        self.transcript: list[str] = []
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        self.stderr_lines: list[str] = []
        self.stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self.stderr_reader.start()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            self.transcript.append(f"<- {line}")
            msg = json.loads(line)
            if "id" in msg and ("result" in msg or "error" in msg):
                self.responses.put(msg)
            else:
                with self.notification_lock:
                    self.notifications.append(msg)

    def _stderr_loop(self):
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip())

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        req = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        line = json.dumps(req)
        self.transcript.append(f"-> {line}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        try:
            resp = self.responses.get(timeout=TIMEOUT_S)
        except queue.Empty:
            self.dump(f"timed out waiting for response to {method}")
        if resp.get("id") != self.next_id:
            self.dump(f"response id mismatch for {method}: {resp}")
        return resp

    def goal_notifications(self) -> list[dict]:
        """Extract anyharness goal_* payloads from session/update notifications."""
        out = []
        with self.notification_lock:
            for msg in self.notifications:
                if msg.get("method") != "session/update":
                    continue
                update = msg.get("params", {}).get("update", {})
                if update.get("sessionUpdate") != "agent_message_chunk":
                    continue
                meta = (update.get("_meta") or {}).get("anyharness") or {}
                if str(meta.get("transcriptEvent", "")).startswith("goal_"):
                    out.append(meta)
        return out

    def dump(self, reason: str):
        print(f"PROBE FAILURE: {reason}", file=sys.stderr)
        print("--- transcript ---", file=sys.stderr)
        for line in self.transcript:
            print(line, file=sys.stderr)
        print("--- stderr ---", file=sys.stderr)
        for line in self.stderr_lines[-40:]:
            print(line, file=sys.stderr)
        self.proc.kill()
        sys.exit(1)

    def check(self, cond: bool, label: str):
        if cond:
            print(f"ok: {label}")
        else:
            self.dump(f"assertion failed: {label}")


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "target/debug/codex-acp"
    codex_home = tempfile.mkdtemp(prefix="codex-acp-goal-probe-home-")
    workdir = tempfile.mkdtemp(prefix="codex-acp-goal-probe-cwd-")
    with open(os.path.join(codex_home, "auth.json"), "w") as f:
        json.dump({"OPENAI_API_KEY": "sk-probe-key", "tokens": None, "last_refresh": None}, f)

    p = Probe(binary, codex_home)

    # 1. initialize: capability advertisement
    resp = p.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    anyharness = resp["result"].get("_meta", {}).get("anyharness")
    p.check(anyharness is not None, "initialize response carries _meta.anyharness")
    p.check(anyharness.get("schemaVersion") == 1, "capability schemaVersion == 1")
    p.check(
        anyharness.get("goals") == {"supported": True, "native": True},
        "goals capability is {supported:true, native:true}",
    )
    p.check("loops" not in anyharness, "loops capability omitted (runtime-emulated)")

    # 2. session/new
    resp = p.request("session/new", {"cwd": workdir, "mcpServers": []})
    session_id = resp["result"]["sessionId"]
    p.check(bool(session_id), f"session/new returned sessionId {session_id}")

    # 3. goal/set (create, paused)
    resp = p.request(
        "_anyharness/goal/set",
        {"sessionId": session_id, "objective": "probe", "status": "paused"},
    )
    goal = resp["result"]["goal"]
    p.check(goal["objective"] == "probe", "set: objective == probe")
    p.check(goal["status"] == "paused", "set: normalized status == paused")
    p.check(goal["nativeStatus"] == "paused", "set: nativeStatus == paused")
    p.check(goal["native"] is True, "set: native == true")
    p.check(goal["tokenBudget"] is None, "set: tokenBudget null")
    p.check(isinstance(goal["updatedAtMs"], int) and goal["updatedAtMs"] > 0, "set: updatedAtMs present")

    notifs = p.goal_notifications()
    p.check(len(notifs) >= 1, "set: goal notification chunk arrived")
    p.check(notifs[-1]["transcriptEvent"] == "goal_updated", "set: notification is goal_updated")
    p.check(notifs[-1]["schemaVersion"] == 1, "set: notification schemaVersion == 1")
    p.check(notifs[-1]["goal"]["objective"] == "probe", "set: notification carries goal")

    # 4. goal/get
    resp = p.request("_anyharness/goal/get", {"sessionId": session_id})
    goal = resp["result"]["goal"]
    p.check(goal is not None and goal["objective"] == "probe", "get: returns stored goal")
    p.check(goal["status"] == "paused", "get: status still paused")

    # 5. goal/set edit (objective changes, status stays paused)
    resp = p.request("_anyharness/goal/set", {"sessionId": session_id, "objective": "probe v2"})
    goal = resp["result"]["goal"]
    p.check(goal["objective"] == "probe v2", "edit: objective == probe v2")
    p.check(goal["status"] == "paused", "edit: status preserved (paused)")
    notifs = p.goal_notifications()
    p.check(
        len(notifs) >= 2 and notifs[-1]["goal"]["objective"] == "probe v2",
        "edit: second goal_updated notification arrived",
    )

    # 6. goals sqlite row (before clear)
    goals_db = os.path.join(codex_home, "goals_1.sqlite")
    p.check(os.path.exists(goals_db), "goals sqlite exists in temp CODEX_HOME")
    with sqlite3.connect(f"file:{goals_db}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT thread_id, objective, status FROM thread_goals").fetchall()
    p.check(len(rows) == 1, "goals sqlite has exactly one goal row")
    p.check(rows[0][0] == session_id, "goal row thread_id == sessionId")
    p.check(rows[0][1] == "probe v2", "goal row objective == probe v2")
    p.check(rows[0][2] == "paused", "goal row status == paused")

    # 7. goal/clear
    resp = p.request("_anyharness/goal/clear", {"sessionId": session_id})
    p.check(resp["result"] == {"cleared": True}, "clear: cleared == true")
    notifs = p.goal_notifications()
    p.check(notifs[-1]["transcriptEvent"] == "goal_cleared", "clear: goal_cleared notification")
    p.check("goal" not in notifs[-1], "clear: notification omits goal")

    # 8. goal/get after clear
    resp = p.request("_anyharness/goal/get", {"sessionId": session_id})
    p.check(resp["result"]["goal"] is None, "get after clear: goal is null")

    with sqlite3.connect(f"file:{goals_db}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM thread_goals").fetchone()
    p.check(rows[0] == 0, "goals sqlite row removed after clear")

    # 9. unknown session id errors cleanly
    resp = p.request(
        "_anyharness/goal/get",
        {"sessionId": "00000000-0000-0000-0000-00000000dead"},
    )
    p.check("error" in resp, "unknown session id: error response")
    p.check("thread not found" in json.dumps(resp["error"]), "unknown session id: thread not found")

    # 10. malformed session id errors cleanly
    resp = p.request("_anyharness/goal/get", {"sessionId": "not-a-thread-id"})
    p.check("error" in resp, "malformed session id: error response")

    p.proc.terminate()
    if os.environ.get("PROBE_VERBOSE"):
        print("\n--- wire transcript ---")
        for line in p.transcript:
            print(line)
    print("\nPROBE PASSED")
    print(f"codex_home={codex_home}")


if __name__ == "__main__":
    main()
