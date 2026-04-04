export interface Citation {
  document: string;
  chunk_id: string;
  text: string;
  score: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  metadata?: Record<string, unknown>;
  isStreaming?: boolean;
  error?: string;
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AskRequest {
  query: string;
  top_k?: number;
  use_rerank?: boolean;
  use_multi_query?: boolean;
  history?: HistoryMessage[];
}

export type Pipeline = "/ask" | "/ask-graph";

export interface SSEEvent {
  event: "token" | "citations" | "metadata" | "error" | "done";
  data: unknown;
}
