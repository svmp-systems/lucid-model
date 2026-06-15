import { NextResponse } from "next/server";

import { loadLocalChatSession } from "@/src/lucid-chat";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ id: string }>;
};

export async function GET(request: Request, context: RouteContext) {
  const { id } = await context.params;
  const url = new URL(request.url);
  const checkpointVersion = url.searchParams.get("checkpointVersion") || "loaded";

  try {
    const session = await loadLocalChatSession(id, checkpointVersion);
    return NextResponse.json({
      session: {
        id: session.id,
        title: session.title,
        checkpointVersion: session.checkpointVersion,
        updatedAt: session.updatedAt,
        messages: session.messages
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "session load failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
