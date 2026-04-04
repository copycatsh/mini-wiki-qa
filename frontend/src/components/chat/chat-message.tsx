"use client";

import { ChatMessage as ChatMessageType } from "@/types/chat";
import { CitationList } from "./citation-list";
import { MetadataTags } from "./metadata-tags";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className="w-full max-w-chat mx-auto">
      <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-1.5">
        {isUser ? "You" : "Mini Wiki"}
      </div>

      {isUser ? (
        <div className="font-sans text-sm font-medium text-text">
          {message.content}
        </div>
      ) : (
        <div>
          <div className="font-body text-[15px] leading-[1.7] text-text">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-0.5 h-4 bg-accent animate-pulse ml-0.5 align-text-bottom" />
            )}
          </div>

          {message.error && (
            <div className="mt-2 px-3 py-2 rounded-md border-l-[3px] border-error bg-[#fef2f2] text-sm text-[#9b2c2c] dark:bg-[#2d0f0f] dark:text-[#fc8181]">
              {message.error}
            </div>
          )}

          {message.citations && message.citations.length > 0 && !message.isStreaming && (
            <CitationList citations={message.citations} />
          )}

          {message.metadata && !message.isStreaming && (
            <MetadataTags metadata={message.metadata} />
          )}
        </div>
      )}
    </div>
  );
}
