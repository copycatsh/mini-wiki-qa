import { AskRequest, Pipeline, SSEEvent } from "@/types/chat";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
// Exposed via NEXT_PUBLIC_ intentionally — this is a localhost-only dev tool,
// not a secret. For network-exposed deployments, add a server-side proxy.
const API_KEY =
  process.env.NEXT_PUBLIC_API_KEY ?? "change-me-in-production";

function headers(): HeadersInit {
  return {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  };
}

export async function fetchHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_URL}/health`, { headers: headers() });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function* streamAsk(
  pipeline: Pipeline,
  body: AskRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${API_URL}${pipeline}/stream`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Stream request failed (${res.status}): ${text}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Response body is not readable");

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last (possibly incomplete) line in the buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const json = trimmed.slice("data:".length).trim();
        if (!json) continue;

        try {
          yield JSON.parse(json) as SSEEvent;
        } catch (e) {
          console.warn("[SSE] Malformed JSON line dropped:", json.slice(0, 200), e);
        }
      }
    }

    // Process any remaining data in the buffer
    if (buffer.trim().startsWith("data:")) {
      const json = buffer.trim().slice("data:".length).trim();
      if (json) {
        try {
          yield JSON.parse(json) as SSEEvent;
        } catch (e) {
          console.warn("[SSE] Malformed JSON in buffer remainder:", json.slice(0, 200), e);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
