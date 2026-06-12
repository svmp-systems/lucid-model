import { NextResponse } from "next/server";

import { enrichChatTurnResponse, sendChatTurn } from "@/src/lucid-chat";
import type { ChatTurnRequest } from "@/src/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = (await request.json()) as Partial<ChatTurnRequest>;

  if (!body.message || !body.sessionId) {
    return NextResponse.json({ error: "message and sessionId are required" }, { status: 400 });
  }

  try {
    const result = await enrichChatTurnResponse(
      await sendChatTurn({
        message: body.message,
        sessionId: body.sessionId,
        checkpointVersion: body.checkpointVersion || "loaded"
      })
    );
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "chat request failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
