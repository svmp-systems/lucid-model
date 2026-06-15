import { NextResponse } from "next/server";

import { listLocalChatSessions, loadLocalChatSession, startLocalSession } from "@/src/lucid-chat";
import type { ServerChatSession } from "@/src/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type CheckpointPrefs = Record<string, string>;

function parseCheckpointPrefs(value: unknown): CheckpointPrefs {
  if (!value || typeof value !== "object") {
    return {};
  }
  const prefs: CheckpointPrefs = {};
  for (const [key, checkpoint] of Object.entries(value as Record<string, unknown>)) {
    if (typeof key === "string" && typeof checkpoint === "string" && checkpoint) {
      prefs[key] = checkpoint;
    }
  }
  return prefs;
}

function toChatSession(session: ServerChatSession) {
  return {
    id: session.id,
    title: session.title,
    checkpointVersion: session.checkpointVersion,
    updatedAt: session.updatedAt,
    messages: session.messages
  };
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const prefsRaw = url.searchParams.get("checkpointPrefs");
  let checkpointPrefs: CheckpointPrefs = {};
  if (prefsRaw) {
    try {
      checkpointPrefs = parseCheckpointPrefs(JSON.parse(prefsRaw));
    } catch {
      checkpointPrefs = {};
    }
  }

  try {
    const sessions = await listLocalChatSessions(checkpointPrefs);
    return NextResponse.json({
      sessions: sessions.map(toChatSession)
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "session list failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}

export async function POST(request: Request) {
  const body = (await request.json()) as {
    sessionId?: string;
    checkpointVersion?: string;
  };

  try {
    const sessionId = await startLocalSession(body.sessionId);
    const session = await loadLocalChatSession(sessionId, body.checkpointVersion || "loaded");
    return NextResponse.json({ session: toChatSession(session) });
  } catch (error) {
    const message = error instanceof Error ? error.message : "session start failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
