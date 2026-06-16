import { NextResponse } from "next/server";

import { loadRunAuditLog } from "@/src/lucid-chat";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const runDir = url.searchParams.get("runDir") || "";

  if (!runDir.trim()) {
    return NextResponse.json({ error: "runDir is required" }, { status: 400 });
  }

  try {
    const auditLog = await loadRunAuditLog(runDir);
    if (!auditLog) {
      return NextResponse.json({ error: "audit log not found" }, { status: 404 });
    }
    return NextResponse.json({ auditLog });
  } catch (error) {
    const message = error instanceof Error ? error.message : "audit load failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
