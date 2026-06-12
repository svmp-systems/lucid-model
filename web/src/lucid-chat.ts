import type { ChatTurnRequest, ChatTurnResponse, CheckpointVersion } from "@/src/types";
import { execFile } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

function backendUrl() {
  return (process.env.LUCID_CHAT_API_URL || "").trim().replace(/\/+$/, "");
}

function repoRoot() {
  return path.resolve(process.cwd(), "..");
}

export async function sendChatTurn(payload: ChatTurnRequest): Promise<ChatTurnResponse> {
  const url = backendUrl();
  if (!url) {
    return runLocalLucid(payload);
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
      checkpoint_version: payload.checkpointVersion,
      history: payload.history
    }),
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const data = (await response.json()) as Record<string, unknown>;
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
    runAuditDir:
      typeof data.run_audit_dir === "string"
        ? data.run_audit_dir
        : typeof data.runAuditDir === "string"
          ? data.runAuditDir
          : ""
  };
}

async function runLocalLucid(payload: ChatTurnRequest): Promise<ChatTurnResponse> {
  const args = [
    "-m",
    "lucid.cli",
    "ask",
    payload.message,
    "--perception",
    process.env.LUCID_WEB_PERCEPTION || "rule",
    "--no-latest"
  ];
  if (payload.checkpointVersion === "cold") {
    args.push("--cold");
  } else if (payload.checkpointVersion && payload.checkpointVersion !== "loaded") {
    args.push("--checkpoint", payload.checkpointVersion);
  }
  const { stdout } = await execFileAsync(process.env.LUCID_PYTHON || "py", args, {
    cwd: repoRoot(),
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8"
    },
    windowsHide: true,
    timeout: Number(process.env.LUCID_WEB_TIMEOUT_MS || 60000),
    maxBuffer: 1024 * 1024 * 4
  });
  return {
    sessionId: payload.sessionId,
    assistantOutput: extractAnswer(stdout),
    auditLog: stdout.trim(),
    checkpointVersion: payload.checkpointVersion,
    lucidityDecision: extractLucidityDecision(stdout),
    runAuditDir: extractRunAuditDir(stdout)
  };
}

function extractAnswer(output: string) {
  const normalized = output.replace(/\r\n/g, "\n");
  const match = normalized.match(/\nanswer\n([\s\S]*?)(?:\n\naudit\n|$)/);
  const answer = match?.[1]?.trim();
  return answer || "(no decoder output)";
}

function extractLucidityDecision(output: string) {
  const match = output.match(/\nlucidity:\s*([^\n]+)/);
  return match?.[1]?.trim() || "";
}

function extractRunAuditDir(output: string) {
  const match = output.match(/\nrun:\s*([^\n]+)/);
  return match?.[1]?.trim() || "";
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
    const raw =
      Array.isArray(data.checkpoints)
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
