import type {
  ChatMessage,
  ChatTurnRequest,
  ChatTurnResponse,
  CheckpointVersion,
  ServerChatSession
} from "@/src/types";
import { execFile } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const DEFAULT_CHAT_AUDIT_DIR = "audit/chat";

type SessionTurnRecord = {
  turn_index?: number;
  user_input?: string;
  assistant_output?: string;
  run_audit_dir?: string;
  lucidity_decision?: string;
};

type SessionRecord = {
  session_id?: string;
  turns?: SessionTurnRecord[];
};

function backendUrl() {
  return (process.env.LUCID_CHAT_API_URL || "").trim().replace(/\/+$/, "");
}

function repoRoot() {
  return path.resolve(process.cwd(), "..");
}

function pythonBin() {
  return process.env.LUCID_PYTHON || "py";
}

function trainRoot() {
  const override = (process.env.LUCID_TRAIN_ROOT || "").trim();
  return override ? path.resolve(override) : path.join(repoRoot(), "lucid", "training", "tree");
}

export function chatAuditDir() {
  return (process.env.LUCID_CHAT_AUDIT_DIR || DEFAULT_CHAT_AUDIT_DIR).trim();
}

export function resolveChatAuditRoot() {
  const text = chatAuditDir().replace(/\\/g, "/");
  if (text.startsWith("train/")) {
    return path.join(repoRoot(), "lucid", "training", "tree", text.slice("train/".length));
  }
  if (path.isAbsolute(text)) {
    return path.resolve(text);
  }
  return path.join(trainRoot(), text);
}

function shortTitle(text: string) {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 34 ? `${clean.slice(0, 34).trim()}...` : clean || "Untitled session";
}

async function runLucidCli(args: string[]) {
  const { stdout, stderr } = await execFileAsync(pythonBin(), ["-m", "lucid.cli", ...args], {
    cwd: repoRoot(),
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8"
    },
    windowsHide: true,
    timeout: Number(process.env.LUCID_WEB_TIMEOUT_MS || 120000),
    maxBuffer: 1024 * 1024 * 8
  });
  return { stdout: stdout.trim(), stderr: stderr.trim() };
}

function appendCheckpointArgs(args: string[], checkpointVersion: string) {
  if (checkpointVersion === "cold") {
    return;
  }
  if (checkpointVersion === "loaded") {
    args.push("--checkpoint", "loaded");
    return;
  }
  if (checkpointVersion) {
    args.push("--checkpoint", checkpointVersion);
  }
}

export async function sendChatTurn(payload: ChatTurnRequest): Promise<ChatTurnResponse> {
  const url = backendUrl();
  if (!url) {
    return runLocalChatTurn(payload);
  }

  const response = await fetch(`${url}/chat/turn`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(process.env.LUCID_CHAT_API_TOKEN
        ? { Authorization: `Bearer ${process.env.LUCID_CHAT_API_TOKEN}` }
        : {})
    },
    body: JSON.stringify({
      text: payload.message,
      message: payload.message,
      session_id: payload.sessionId,
      checkpoint: payload.checkpointVersion,
      checkpoint_version: payload.checkpointVersion
    }),
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const data = (await response.json()) as Record<string, unknown>;
  return normalizeChatTurnResponse(data, payload);
}

async function runLocalChatTurn(payload: ChatTurnRequest): Promise<ChatTurnResponse> {
  const args = [
    "chat",
    "send",
    payload.message,
    "--session-id",
    payload.sessionId,
    "--audit-dir",
    chatAuditDir(),
    "--perception",
    process.env.LUCID_WEB_PERCEPTION || "rule",
    "--json"
  ];
  appendCheckpointArgs(args, payload.checkpointVersion);

  const { stdout, stderr } = await runLucidCli(args);
  if (!stdout) {
    throw new Error(stderr || "chat send returned no output");
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(stdout) as Record<string, unknown>;
  } catch {
    throw new Error(stderr || stdout || "chat send returned invalid JSON");
  }

  return normalizeChatTurnResponse(data, payload);
}

function normalizeChatTurnResponse(
  data: Record<string, unknown>,
  payload: ChatTurnRequest
): ChatTurnResponse {
  const runAuditDir =
    typeof data.run_audit_dir === "string"
      ? data.run_audit_dir
      : typeof data.runAuditDir === "string"
        ? data.runAuditDir
        : "";

  return {
    sessionId:
      typeof data.session_id === "string"
        ? data.session_id
        : typeof data.sessionId === "string"
          ? data.sessionId
          : payload.sessionId,
    assistantOutput:
      typeof data.assistant_output === "string"
        ? data.assistant_output
        : typeof data.assistantOutput === "string"
          ? data.assistantOutput
          : typeof data.answer === "string"
            ? data.answer
            : "",
    auditLog:
      typeof data.audit_log === "string"
        ? data.audit_log
        : typeof data.auditLog === "string"
          ? data.auditLog
          : "",
    checkpointVersion:
      typeof data.checkpoint_version === "string"
        ? data.checkpoint_version
        : typeof data.checkpointVersion === "string"
          ? data.checkpointVersion
          : payload.checkpointVersion,
    lucidityDecision:
      typeof data.lucidity_decision === "string"
        ? data.lucidity_decision
        : typeof data.lucidityDecision === "string"
          ? data.lucidityDecision
          : "",
    runAuditDir,
    turnIndex:
      typeof data.turn_index === "number"
        ? data.turn_index
        : typeof data.turnIndex === "number"
          ? data.turnIndex
          : undefined,
    sessionAuditPath:
      typeof data.session_audit_path === "string"
        ? data.session_audit_path
        : typeof data.sessionAuditPath === "string"
          ? data.sessionAuditPath
          : ""
  };
}

export async function enrichChatTurnResponse(result: ChatTurnResponse): Promise<ChatTurnResponse> {
  const runAuditDir = result.runAuditDir;
  if (!runAuditDir) {
    return result;
  }

  const auditLog = result.auditLog || (await loadRunAuditSummary(runAuditDir));
  const lucidityDecision = result.lucidityDecision || extractLucidityDecision(auditLog);
  return {
    ...result,
    auditLog,
    lucidityDecision
  };
}

export async function startLocalSession(sessionId?: string) {
  const args = ["chat", "start", "--audit-dir", chatAuditDir()];
  if (sessionId) {
    args.push("--session-id", sessionId);
  }
  const { stdout, stderr } = await runLucidCli(args);
  const id = stdout.trim();
  if (!id) {
    throw new Error(stderr || "chat start returned no session id");
  }
  return id;
}

export async function listLocalChatSessionIds() {
  const { stdout } = await runLucidCli(["chat", "list", "--audit-dir", chatAuditDir()]);
  return stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

export async function loadLocalSessionRecord(sessionId: string): Promise<SessionRecord | null> {
  const sessionPath = path.join(resolveChatAuditRoot(), sessionId, "session.json");
  try {
    const raw = await readFile(sessionPath, "utf-8");
    return JSON.parse(raw) as SessionRecord;
  } catch {
    return null;
  }
}

export function turnsToMessages(turns: SessionTurnRecord[], checkpointVersion = "loaded"): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    const turnIndex = typeof turn.turn_index === "number" ? turn.turn_index : messages.length / 2 + 1;
    const userInput = String(turn.user_input || "").trim();
    const assistantOutput = String(turn.assistant_output || "").trim();
    if (userInput) {
      messages.push({
        id: `${turnIndex}-user`,
        role: "user",
        content: userInput,
        turnIndex,
        checkpointVersion
      });
    }
    if (assistantOutput) {
      messages.push({
        id: `${turnIndex}-assistant`,
        role: "assistant",
        content: assistantOutput,
        turnIndex,
        runAuditDir: turn.run_audit_dir || "",
        lucidityDecision: turn.lucidity_decision || "",
        checkpointVersion
      });
    }
  }
  return messages;
}

export async function loadLocalChatSession(
  sessionId: string,
  checkpointVersion = "loaded"
): Promise<ServerChatSession> {
  const record = await loadLocalSessionRecord(sessionId);
  const turns = Array.isArray(record?.turns) ? record.turns : [];
  const messages = await enrichSessionMessages(turnsToMessages(turns, checkpointVersion));
  const firstUser = turns.find((turn) => String(turn.user_input || "").trim())?.user_input || "";
  return {
    id: sessionId,
    title: firstUser ? shortTitle(String(firstUser)) : "Untitled session",
    checkpointVersion,
    turnCount: turns.length,
    messages
  };
}

async function enrichSessionMessages(messages: ChatMessage[]): Promise<ChatMessage[]> {
  return Promise.all(
    messages.map(async (message) => {
      if (message.role !== "assistant" || message.auditLog || !message.runAuditDir) {
        return message;
      }
      const auditLog = await loadRunAuditSummary(message.runAuditDir);
      return {
        ...message,
        auditLog,
        lucidityDecision: message.lucidityDecision || extractLucidityDecision(auditLog)
      };
    })
  );
}

export async function listLocalChatSessions(
  checkpointPrefs: Record<string, string> = {}
): Promise<ServerChatSession[]> {
  const ids = await listLocalChatSessionIds();
  const sessions = await Promise.all(
    ids.map((id) => loadLocalChatSession(id, checkpointPrefs[id] || "loaded"))
  );
  return sessions.sort((a, b) => {
    const aTurn = a.turnCount;
    const bTurn = b.turnCount;
    if (aTurn !== bTurn) {
      return bTurn - aTurn;
    }
    return b.id.localeCompare(a.id);
  });
}

async function loadRunAuditSummary(runAuditDir: string) {
  try {
    const manifestPath = path.join(runAuditDir, "manifest.json");
    const raw = await readFile(manifestPath, "utf-8");
    const manifest = JSON.parse(raw) as {
      summary?: { headline?: string; lines?: string[] };
      lucidity_decision?: string;
    };
    const summary = manifest.summary;
    if (summary?.lines?.length) {
      return [summary.headline || "", ...summary.lines].filter(Boolean).join("\n");
    }
    return JSON.stringify(manifest, null, 2);
  } catch {
    return "";
  }
}

function extractLucidityDecision(auditLog: string) {
  const headline = auditLog.split("\n")[0]?.trim() || "";
  const match = headline.match(/^(\w+)\s·/);
  return match?.[1] || "";
}

export async function listLocalCheckpoints() {
  const remoteCheckpoints = await listRemoteCheckpoints();
  if (remoteCheckpoints.length) {
    return markDefault(remoteCheckpoints);
  }

  const root = repoRoot();
  const registryPath = path.join(root, "lucid", "training", "tree", "checkpoints", "saves", "registry.json");
  const savesDir = path.join(root, "lucid", "training", "tree", "checkpoints", "saves");

  const versions = [
    {
      id: "loaded",
      label: "Loaded",
      detail: "Pinned inference slot"
    },
    {
      id: "cold",
      label: "Cold",
      detail: "No checkpoint"
    }
  ];

  try {
    const raw = await readFile(registryPath, "utf-8");
    const parsed = JSON.parse(raw) as { checkpoints?: Array<Record<string, unknown>> };
    const rows = Array.isArray(parsed.checkpoints) ? parsed.checkpoints : [];
    const checkpoints = rows
      .map((row) => ({
        id: typeof row.name === "string" ? row.name : "",
        label: typeof row.name === "string" ? row.name : "",
        detail:
          typeof row.command === "string" && row.command
            ? row.command
            : typeof row.training_steps === "number"
              ? `${row.training_steps} training steps`
              : "Saved checkpoint"
      }))
      .filter((row) => row.id)
      .sort((a, b) => b.id.localeCompare(a.id, undefined, { numeric: true }));

    return checkpoints.length ? markDefault([...checkpoints, ...versions]) : markDefault(versions);
  } catch {
    try {
      const entries = await readdir(savesDir, { withFileTypes: true });
      const checkpoints = entries
        .filter((entry) => entry.isDirectory() && /^cp_\d+$/.test(entry.name))
        .map((entry) => ({
          id: entry.name,
          label: entry.name,
          detail: "Saved checkpoint"
        }))
        .sort((a, b) => b.id.localeCompare(a.id, undefined, { numeric: true }));
      return checkpoints.length ? markDefault([...checkpoints, ...versions]) : markDefault(versions);
    } catch {
      return markDefault(versions);
    }
  }
}

function markDefault<T extends { id: string; isDefault?: boolean }>(versions: T[]) {
  return versions.map((version, index) => ({
    ...version,
    isDefault: index === 0
  }));
}

async function listRemoteCheckpoints(): Promise<CheckpointVersion[]> {
  const url = backendUrl();
  if (!url) {
    return [];
  }

  try {
    const response = await fetch(`${url}/checkpoints`, {
      headers: process.env.LUCID_CHAT_API_TOKEN
        ? { Authorization: `Bearer ${process.env.LUCID_CHAT_API_TOKEN}` }
        : undefined,
      cache: "no-store"
    });
    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as Record<string, unknown>;
    const raw = Array.isArray(data.checkpoints)
      ? data.checkpoints
      : Array.isArray(data.checkpoint_versions)
        ? data.checkpoint_versions
        : Array.isArray(data.versions)
          ? data.versions
          : [];

    return raw
      .map((item) => normalizeRemoteCheckpoint(item))
      .filter((item): item is CheckpointVersion => Boolean(item));
  } catch {
    return [];
  }
}

function normalizeRemoteCheckpoint(item: unknown): CheckpointVersion | null {
  if (typeof item === "string") {
    return { id: item, label: item, detail: "Hosted checkpoint" };
  }

  if (!item || typeof item !== "object") {
    return null;
  }

  const row = item as Record<string, unknown>;
  const id =
    typeof row.id === "string"
      ? row.id
      : typeof row.name === "string"
        ? row.name
        : typeof row.checkpoint === "string"
          ? row.checkpoint
          : "";
  if (!id) {
    return null;
  }

  return {
    id,
    label: typeof row.label === "string" ? row.label : id,
    detail: typeof row.detail === "string" ? row.detail : "Hosted checkpoint"
  };
}
