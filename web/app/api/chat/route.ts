import { NextResponse } from "next/server";

import { sendChatTurn } from "@/src/lucid-chat";
import type { ChatMessage, ChatTurnRequest } from "@/src/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function isMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    (item.role === "user" || item.role === "assistant") &&
    typeof item.content === "string"
  );
}

export async function POST(request: Request) {
  const body = (await request.json()) as Partial<ChatTurnRequest>;

  if (!body.message || !body.sessionId) {
    return NextResponse.json({ error: "message and sessionId are required" }, { status: 400 });
  }

  try {
    const result = await sendChatTurn({
      message: body.message,
      sessionId: body.sessionId,
      checkpointVersion: body.checkpointVersion || "loaded",
      history: Array.isArray(body.history) ? body.history.filter(isMessage) : []
    });
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "chat request failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
