export type Role = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  auditLog?: string;
  runAuditDir?: string;
  lucidityDecision?: string;
  checkpointVersion?: string;
  turnIndex?: number;
};

export type ChatSession = {
  id: string;
  title: string;
  checkpointVersion: string;
  messages: ChatMessage[];
};

export type ChatTurnRequest = {
  message: string;
  sessionId: string;
  checkpointVersion: string;
};

export type ChatTurnResponse = {
  assistantOutput: string;
  sessionId: string;
  auditLog: string;
  checkpointVersion: string;
  lucidityDecision: string;
  runAuditDir: string;
  turnIndex?: number;
  sessionAuditPath?: string;
};

export type ServerChatSession = {
  id: string;
  title: string;
  checkpointVersion: string;
  turnCount: number;
  messages: ChatMessage[];
};

export type CheckpointVersion = {
  id: string;
  label: string;
  detail: string;
  isDefault?: boolean;
};
