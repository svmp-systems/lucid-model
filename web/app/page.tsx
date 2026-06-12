"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { MessageCircle, Plus, ScrollText, Send, Trash2 } from "lucide-react";

import type { ChatMessage, ChatSession, CheckpointVersion } from "@/src/types";

const STORAGE_KEY = "etoise-sessions-v1";
const DEFAULT_CHECKPOINT = "loaded";

const STARTER_SESSIONS: ChatSession[] = [
  {
    id: "etoise-session",
    title: "Untitled session",
    checkpointVersion: DEFAULT_CHECKPOINT,
    messages: []
  }
];

function createEmptySession(checkpointVersion = DEFAULT_CHECKPOINT): ChatSession {
  return {
    id: `session-${Date.now()}`,
    title: "Untitled session",
    checkpointVersion,
    messages: []
  };
}

function shortTitle(text: string) {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 34 ? `${clean.slice(0, 34).trim()}...` : clean || "Untitled session";
}

function makeMessage(role: ChatMessage["role"], content: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    ...extra
  };
}

function normalizeSessions(value: unknown): ChatSession[] {
  if (!Array.isArray(value)) {
    return STARTER_SESSIONS;
  }

  const sessions = value
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
    .map((item) => ({
      id: typeof item.id === "string" ? item.id : `session-${Date.now()}`,
      title: typeof item.title === "string" ? item.title : "Untitled session",
      checkpointVersion:
        typeof item.checkpointVersion === "string" ? item.checkpointVersion : DEFAULT_CHECKPOINT,
      messages: Array.isArray(item.messages)
        ? item.messages.filter((message): message is ChatMessage => {
            return Boolean(
              message &&
                typeof message === "object" &&
                typeof (message as ChatMessage).id === "string" &&
                ((message as ChatMessage).role === "user" ||
                  (message as ChatMessage).role === "assistant") &&
                typeof (message as ChatMessage).content === "string"
            );
          })
        : []
    }));

  return sessions.length ? sessions : STARTER_SESSIONS;
}

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>(STARTER_SESSIONS);
  const [activeId, setActiveId] = useState(STARTER_SESSIONS[0].id);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [checkpoints, setCheckpoints] = useState<CheckpointVersion[]>([
    { id: DEFAULT_CHECKPOINT, label: "Loaded", detail: "Pinned inference slot", isDefault: true }
  ]);
  const [auditOpen, setAuditOpen] = useState(false);
  const [selectedAuditId, setSelectedAuditId] = useState<string>("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const nextSessions = normalizeSessions(JSON.parse(saved));
        setSessions(nextSessions);
        setActiveId(nextSessions[0].id);
      } catch {
        window.localStorage.removeItem(STORAGE_KEY);
      }
    }
    setLoaded(true);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadCheckpoints() {
      try {
        const response = await fetch("/api/checkpoints");
        if (!response.ok) {
          return;
        }
        const data = (await response.json()) as { checkpoints?: CheckpointVersion[] };
        if (!cancelled && Array.isArray(data.checkpoints) && data.checkpoints.length) {
          setCheckpoints(data.checkpoints);
          setSessions((current) =>
            current.map((session) =>
              session.checkpointVersion === DEFAULT_CHECKPOINT && data.checkpoints?.[0]?.id
                ? { ...session, checkpointVersion: data.checkpoints[0].id }
                : session
            )
          );
        }
      } catch {
        // Keep the local default when checkpoint discovery is unavailable.
      }
    }
    loadCheckpoints();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (loaded) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }
  }, [loaded, sessions]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeId) || sessions[0],
    [activeId, sessions]
  );

  const selectedAudit = useMemo(() => {
    const messages = activeSession?.messages || [];
    return (
      messages.find((message) => message.id === selectedAuditId && message.auditLog) ||
      [...messages].reverse().find((message) => message.auditLog)
    );
  }, [activeSession, selectedAuditId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [activeSession?.messages.length]);

  function updateSession(nextSession: ChatSession) {
    setSessions((current) =>
      current.map((session) => (session.id === nextSession.id ? nextSession : session))
    );
  }

  function startNewChat() {
    const session = createEmptySession(checkpoints[0]?.id || DEFAULT_CHECKPOINT);
    setSessions((current) => [session, ...current]);
    setActiveId(session.id);
    setSelectedAuditId("");
    setInput("");
  }

  function deleteSession(sessionId: string) {
    const remaining = sessions.filter((session) => session.id !== sessionId);
    if (remaining.length === 0) {
      const replacement = createEmptySession(checkpoints[0]?.id || DEFAULT_CHECKPOINT);
      setSessions([replacement]);
      setActiveId(replacement.id);
      setSelectedAuditId("");
      return;
    }

    setSessions(remaining);
    if (sessionId === activeId) {
      setActiveId(remaining[0].id);
      setSelectedAuditId("");
    }
  }

  function setCheckpointVersion(checkpointVersion: string) {
    if (!activeSession) {
      return;
    }
    updateSession({ ...activeSession, checkpointVersion });
  }

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    if (!activeSession || !input.trim() || isSending) {
      return;
    }

    const userMessage = makeMessage("user", input.trim(), {
      checkpointVersion: activeSession.checkpointVersion
    });
    const nextSession: ChatSession = {
      ...activeSession,
      title: activeSession.messages.length === 0 ? shortTitle(userMessage.content) : activeSession.title,
      messages: [...activeSession.messages, userMessage]
    };

    updateSession(nextSession);
    setInput("");
    setIsSending(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: userMessage.content,
          sessionId: nextSession.id,
          checkpointVersion: nextSession.checkpointVersion,
          history: nextSession.messages
        })
      });

      if (!response.ok) {
        throw new Error("Request failed");
      }

      const data = (await response.json()) as {
        assistantOutput?: string;
        auditLog?: string;
        runAuditDir?: string;
        lucidityDecision?: string;
        checkpointVersion?: string;
      };
      const assistantMessage = makeMessage("assistant", data.assistantOutput || "(no decoder output)", {
        auditLog: data.auditLog,
        runAuditDir: data.runAuditDir,
        lucidityDecision: data.lucidityDecision,
        checkpointVersion: data.checkpointVersion || nextSession.checkpointVersion
      });
      updateSession({
        ...nextSession,
        messages: [...nextSession.messages, assistantMessage]
      });
      setSelectedAuditId(assistantMessage.id);
    } catch {
      updateSession({
        ...nextSession,
        messages: [
          ...nextSession.messages,
          makeMessage("assistant", "The local runtime did not return a response.", {
            checkpointVersion: nextSession.checkpointVersion
          })
        ]
      });
    } finally {
      setIsSending(false);
    }
  }

  if (!activeSession) {
    return null;
  }

  const isEmpty = activeSession.messages.length === 0;
  const activeCheckpoint =
    checkpoints.find((checkpoint) => checkpoint.id === activeSession.checkpointVersion) ||
    checkpoints[0];
  const turnCount = activeSession.messages.filter((message) => message.role === "user").length;

  return (
    <main className={`etoise-app ${auditOpen ? "audit-open" : ""}`}>
      <header className="top-header">
        <div className="brand">
          <span className="crescent" aria-hidden="true" />
          <span>Etoise</span>
        </div>
      </header>

      <div className="body">
        <aside className="sidebar">
          <button className="new-chat" type="button" onClick={startNewChat}>
            <Plus size={20} strokeWidth={2} />
            <span>New Session</span>
          </button>

          <div className="history">
            <div className="history-title">Sessions</div>
            <div className="history-list">
              {sessions.map((session) => (
                <div
                  className={`history-item ${session.id === activeId ? "active" : ""}`}
                  key={session.id}
                >
                  <button
                    className="history-select"
                    type="button"
                    onClick={() => {
                      setActiveId(session.id);
                      setSelectedAuditId("");
                    }}
                  >
                    <MessageCircle size={18} strokeWidth={1.8} />
                    <span>
                      <strong>{session.title}</strong>
                      <small>{session.checkpointVersion}</small>
                    </span>
                  </button>
                  <button
                    aria-label={`Delete ${session.title}`}
                    className="delete-session"
                    type="button"
                    onClick={() => deleteSession(session.id)}
                  >
                    <Trash2 size={16} strokeWidth={1.8} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-toolbar">
            <div>
              <h1>{activeSession.title}</h1>
              <p>
                {turnCount === 1 ? "1 turn" : `${turnCount} turns`} /{" "}
                {activeCheckpoint?.label || activeSession.checkpointVersion}
              </p>
            </div>
            <button
              className="toolbar-audit"
              type="button"
              onClick={() => setAuditOpen((open) => !open)}
            >
              <ScrollText size={18} strokeWidth={1.8} />
              <span>{auditOpen ? "Hide Audit" : "Audit Log"}</span>
            </button>
          </div>

          <div className="messages" ref={scrollRef}>
            {isEmpty ? (
              <div className="empty-state">
                <span className="assistant-mark" aria-hidden="true" />
                <div className="assistant-bubble">Ready for a new session.</div>
              </div>
            ) : (
              <div className="message-list">
                {activeSession.messages.map((message) =>
                  message.role === "user" ? (
                    <div className="user-row" key={message.id}>
                      <div className="user-bubble">{message.content}</div>
                    </div>
                  ) : (
                    <div className="assistant-row" key={message.id}>
                      <span className="assistant-mark" aria-hidden="true" />
                      <div className="assistant-stack">
                        <div className="assistant-bubble">{message.content}</div>
                        {(message.auditLog || message.lucidityDecision || message.checkpointVersion) && (
                          <button
                            className="message-audit"
                            type="button"
                            onClick={() => {
                              setSelectedAuditId(message.id);
                              setAuditOpen(true);
                            }}
                          >
                            Audit
                            <span>{message.checkpointVersion}</span>
                            {message.lucidityDecision ? <span>{message.lucidityDecision}</span> : null}
                          </button>
                        )}
                      </div>
                    </div>
                  )
                )}
              </div>
            )}
          </div>

          <form className="composer" onSubmit={submitMessage}>
            <input
              aria-label="Message"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Message Etoise..."
            />
            <label className="composer-checkpoint">
              <span>Checkpoint</span>
              <select
                aria-label="Checkpoint"
                className="checkpoint-select"
                value={activeSession.checkpointVersion}
                onChange={(event) => setCheckpointVersion(event.target.value)}
              >
                {checkpoints.map((checkpoint) => (
                  <option key={checkpoint.id} value={checkpoint.id}>
                    {checkpoint.label}
                  </option>
                ))}
              </select>
            </label>
            <button aria-label="Send" type="submit" disabled={isSending || !input.trim()}>
              <Send size={22} strokeWidth={2} />
            </button>
          </form>
        </section>

        <aside className="audit-drawer">
          <div className="audit-header">
            <div>
              <h2>Audit Log</h2>
              <p>{selectedAudit?.runAuditDir || "No run selected"}</p>
            </div>
            <button type="button" onClick={() => setAuditOpen(false)}>
              Close
            </button>
          </div>
          <pre>{selectedAudit?.auditLog || "Run a turn, then open its audit here."}</pre>
        </aside>
      </div>
    </main>
  );
}
