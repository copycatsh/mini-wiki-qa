"use client";

import { useState, useRef, useEffect } from "react";
import { useChat } from "@/hooks/use-chat";
import { Pipeline } from "@/types/chat";
import { Sidebar } from "@/components/sidebar/sidebar";
import { ChatMessage } from "@/components/chat/chat-message";
import { ChatInput } from "@/components/chat/chat-input";
import { EmptyState } from "@/components/chat/empty-state";
import { MobileSettings } from "@/components/sidebar/mobile-settings";

export default function Home() {
  const [pipeline, setPipeline] = useState<Pipeline>("/ask");
  const [useRerank, setUseRerank] = useState(false);
  const { messages, isStreaming, sendMessage, stopStreaming } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSend = (query: string) => {
    sendMessage(query, pipeline, useRerank);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-screen">
      <Sidebar
        pipeline={pipeline}
        onPipelineChange={setPipeline}
        useRerank={useRerank}
        onRerankChange={setUseRerank}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <MobileSettings
          pipeline={pipeline}
          onPipelineChange={setPipeline}
          useRerank={useRerank}
          onRerankChange={setUseRerank}
        />
        {messages.length === 0 ? (
          <EmptyState onSuggestionClick={handleSend} />
        ) : (
          <div
            className="flex-1 overflow-y-auto px-8 py-6"
            role="log"
            aria-live="polite"
          >
            <div className="flex flex-col gap-4">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        <ChatInput
          onSend={handleSend}
          onStop={stopStreaming}
          isStreaming={isStreaming}
        />
      </main>
    </div>
  );
}
