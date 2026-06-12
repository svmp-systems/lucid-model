export type Role = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  auditLog?: string;
  runAuditDir?: string;
  lucidityDecision?: string;
  checkpointVersion?: string;
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
  history: ChatMessage[];
};

export type ChatTurnResponse = {
  assistantOutput: string;
  sessionId: string;
  auditLog: string;
  checkpointVersion: string;
  lucidityDecision: string;
  runAuditDir: string;
};

export type CheckpointVersion = {
  id: string;
  label: string;
  detail: string;
  isDefault?: boolean;
};
