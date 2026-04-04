"use client";

import { useCallback, useRef, useState } from "react";
import { streamAsk } from "@/lib/api";
import { ChatMessage, Citation, Pipeline } from "@/types/chat";

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (query: string, pipeline: Pipeline, useRerank?: boolean) => {
      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: query,
      };
      const assistantId = generateId();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const stream = streamAsk(
          pipeline,
          { query, use_rerank: useRerank },
          controller.signal,
        );

        for await (const event of stream) {
          switch (event.event) {
            case "token":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + (event.data as string) }
                    : m,
                ),
              );
              break;
            case "citations":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, citations: event.data as Citation[] }
                    : m,
                ),
              );
              break;
            case "metadata":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, metadata: event.data as Record<string, unknown> }
                    : m,
                ),
              );
              break;
            case "error":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, error: event.data as string, isStreaming: false }
                    : m,
                ),
              );
              setIsStreaming(false);
              return;
            case "done":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m,
                ),
              );
              setIsStreaming(false);
              return;
          }
        }

        // Stream ended without a done event
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        );
        setIsStreaming(false);
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, isStreaming: false } : m,
            ),
          );
          return;
        }

        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, error: errorMessage, isStreaming: false }
              : m,
          ),
        );
        setIsStreaming(false);
      } finally {
        abortRef.current = null;
      }
    },
    [],
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendMessage, stopStreaming, clearMessages };
}
