"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MessageCircle, Plus, ScrollText, Send, Trash2 } from "lucide-react";

import type { ChatMessage, ChatSession, CheckpointVersion } from "@/src/types";

const CHECKPOINT_PREFS_KEY = "etoise-checkpoint-prefs-v1";
const DEFAULT_CHECKPOINT = "loaded";

function createEmptySession(checkpointVersion = DEFAULT_CHECKPOINT): ChatSession {
  return {
    id: "",
    title: "Untitled session",
    checkpointVersion,
    messages: []
  };
}

function shortTitle(text: string) {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 34 ? `${clean.slice(0, 34).trim()}...` : clean || "Untitled session";
}

function formatUpdatedAt(value?: string) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function loadCheckpointPrefs(): Record<string, string> {
  if (typeof window === "undefined") {
    return {};
  }
  const saved = window.localStorage.getItem(CHECKPOINT_PREFS_KEY);
  if (!saved) {
    return {};
  }
  try {
    const parsed = JSON.parse(saved) as unknown;
    if (!parsed || typeof parsed !== "object") {
      return {};
    }
    const prefs: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof key === "string" && typeof value === "string" && value) {
        prefs[key] = value;
      }
    }
    return prefs;
  } catch {
    return {};
  }
}

function saveCheckpointPrefs(prefs: Record<string, string>) {
  window.localStorage.setItem(CHECKPOINT_PREFS_KEY, JSON.stringify(prefs));
}

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState("");
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [checkpointPrefs, setCheckpointPrefs] = useState<Record<string, string>>({});
  const [checkpoints, setCheckpoints] = useState<CheckpointVersion[]>([
    { id: DEFAULT_CHECKPOINT, label: "Loaded", detail: "Pinned inference slot", isDefault: true }
  ]);
  const [auditOpen, setAuditOpen] = useState(false);
  const [selectedAuditId, setSelectedAuditId] = useState<string>("");
  const [auditContent, setAuditContent] = useState("");
  const [auditLoading, setAuditLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const reloadSessions = useCallback(async (prefs: Record<string, string>) => {
    const query = new URLSearchParams({
      checkpointPrefs: JSON.stringify(prefs)
    });
    const response = await fetch(`/api/sessions?${query.toString()}`);
    if (!response.ok) {
      throw new Error("Failed to load sessions");
    }
    const data = (await response.json()) as { sessions?: ChatSession[] };
    const nextSessions = Array.isArray(data.sessions) ? data.sessions : [];
    setSessions(nextSessions);
    setActiveId((current) => {
      if (current && nextSessions.some((session) => session.id === current)) {
        return current;
      }
      return nextSessions[0]?.id || "";
    });
  }, []);

  useEffect(() => {
    const prefs = loadCheckpointPrefs();
    setCheckpointPrefs(prefs);
    reloadSessions(prefs)
      .catch(() => {
        setSessions([]);
        setActiveId("");
      })
      .finally(() => setLoaded(true));
  }, [reloadSessions]);

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
      saveCheckpointPrefs(checkpointPrefs);
    }
  }, [loaded, checkpointPrefs]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeId) || sessions[0],
    [activeId, sessions]
  );

  const selectedAudit = useMemo(() => {
    const messages = activeSession?.messages || [];
    if (selectedAuditId) {
      const chosen = messages.find((message) => message.id === selectedAuditId);
      if (chosen && (chosen.auditLog || chosen.runAuditDir)) {
        return chosen;
      }
    }
    return [...messages].reverse().find((message) => message.auditLog || message.runAuditDir);
  }, [activeSession, selectedAuditId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [activeSession?.messages.length]);

  useEffect(() => {
    const runDir = selectedAudit?.runAuditDir;
    if (!auditOpen || !runDir) {
      setAuditContent("");
      setAuditLoading(false);
      return;
    }

    let cancelled = false;
    setAuditLoading(true);
    setAuditContent("");

    fetch(`/api/audit?runDir=${encodeURIComponent(runDir)}`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Failed to load audit log");
        }
        const data = (await response.json()) as { auditLog?: string };
        if (!cancelled) {
          setAuditContent(data.auditLog || "");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAuditContent("Could not load the audit log for this turn.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setAuditLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [auditOpen, selectedAudit?.runAuditDir, selectedAuditId]);

  function updateSession(nextSession: ChatSession) {
    setSessions((current) =>
      current.map((session) => (session.id === nextSession.id ? nextSession : session))
    );
  }

  async function startNewChat() {
    const checkpointVersion = checkpoints[0]?.id || DEFAULT_CHECKPOINT;
    try {
      const response = await fetch("/api/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ checkpointVersion })
      });
      if (!response.ok) {
        throw new Error("Failed to create session");
      }
      const data = (await response.json()) as { session?: ChatSession };
      const session = data.session || createEmptySession(checkpointVersion);
      const nextPrefs = { ...checkpointPrefs, [session.id]: session.checkpointVersion || checkpointVersion };
      setCheckpointPrefs(nextPrefs);
      setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)]);
      setActiveId(session.id);
      setSelectedAuditId("");
      setInput("");
    } catch {
      // Keep the UI responsive even if session creation fails.
    }
  }

  async function deleteSession(sessionId: string) {
    try {
      const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE"
      });
      if (!response.ok) {
        throw new Error("Failed to delete session");
      }
    } catch {
      return;
    }

    const remaining = sessions.filter((session) => session.id !== sessionId);
    const nextPrefs = { ...checkpointPrefs };
    delete nextPrefs[sessionId];
    setCheckpointPrefs(nextPrefs);

    if (remaining.length === 0) {
      setSessions([]);
      setActiveId("");
      setSelectedAuditId("");
      void startNewChat();
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
    const nextPrefs = { ...checkpointPrefs, [activeSession.id]: checkpointVersion };
    setCheckpointPrefs(nextPrefs);
    updateSession({ ...activeSession, checkpointVersion });
  }

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    if (!activeSession?.id || !input.trim() || isSending) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
      checkpointVersion: activeSession.checkpointVersion
    };
    const optimisticSession: ChatSession = {
      ...activeSession,
      title: activeSession.messages.length === 0 ? shortTitle(userMessage.content) : activeSession.title,
      messages: [...activeSession.messages, userMessage]
    };

    updateSession(optimisticSession);
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
          sessionId: optimisticSession.id,
          checkpointVersion: optimisticSession.checkpointVersion
        })
      });

      if (!response.ok) {
        const errorBody = (await response.json()) as { error?: string };
        throw new Error(errorBody.error || "Request failed");
      }

      const data = (await response.json()) as {
        assistantOutput?: string;
        auditLog?: string;
        runAuditDir?: string;
        lucidityDecision?: string;
        checkpointVersion?: string;
        turnIndex?: number;
      };

      const sessionResponse = await fetch(
        `/api/sessions/${encodeURIComponent(optimisticSession.id)}?checkpointVersion=${encodeURIComponent(
          optimisticSession.checkpointVersion
        )}`
      );
      if (sessionResponse.ok) {
        const sessionData = (await sessionResponse.json()) as { session?: ChatSession };
        if (sessionData.session) {
          updateSession(sessionData.session);
          const lastAssistant = [...sessionData.session.messages]
            .reverse()
            .find((message) => message.role === "assistant");
          if (lastAssistant) {
            setSelectedAuditId(lastAssistant.id);
          }
          return;
        }
      }

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.assistantOutput || "(no decoder output)",
        auditLog: data.auditLog,
        runAuditDir: data.runAuditDir,
        lucidityDecision: data.lucidityDecision,
        checkpointVersion: data.checkpointVersion || optimisticSession.checkpointVersion,
        turnIndex: data.turnIndex
      };
      updateSession({
        ...optimisticSession,
        messages: [...optimisticSession.messages, assistantMessage]
      });
      setSelectedAuditId(assistantMessage.id);
    } catch (error) {
      updateSession({
        ...optimisticSession,
        messages: [
          ...optimisticSession.messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              error instanceof Error
                ? `The chat runtime did not return a response: ${error.message}`
                : "The chat runtime did not return a response.",
            checkpointVersion: optimisticSession.checkpointVersion
          }
        ]
      });
    } finally {
      setIsSending(false);
    }
  }

  if (!loaded) {
    return null;
  }

  if (!activeSession) {
    return (
      <main className="etoise-app">
        <header className="top-header">
          <div className="brand">
            <span className="crescent" aria-hidden="true" />
            <span>Etoise</span>
          </div>
        </header>
        <div className="body">
          <aside className="sidebar">
            <button className="new-chat" type="button" onClick={() => void startNewChat()}>
              <Plus size={20} strokeWidth={2} />
              <span>New Session</span>
            </button>
          </aside>
          <section className="chat-panel">
            <div className="empty-state">
              <span className="assistant-mark" aria-hidden="true" />
              <div className="assistant-bubble">Start a session to begin chatting.</div>
            </div>
          </section>
        </div>
      </main>
    );
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
          <button className="new-chat" type="button" onClick={() => void startNewChat()}>
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
                      <small>
                        {session.checkpointVersion}
                        {formatUpdatedAt(session.updatedAt) ? ` · ${formatUpdatedAt(session.updatedAt)}` : ""}
                      </small>
                    </span>
                  </button>
                  <button
                    aria-label={`Delete ${session.title}`}
                    className="delete-session"
                    type="button"
                    onClick={() => void deleteSession(session.id)}
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
                        {(message.auditLog ||
                          message.runAuditDir ||
                          message.lucidityDecision ||
                          message.checkpointVersion) && (
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
          <pre>
            {auditLoading
              ? "Loading audit log..."
              : auditContent || selectedAudit?.auditLog || "Run a turn, then open its audit here."}
          </pre>
        </aside>
      </div>
    </main>
  );
}
