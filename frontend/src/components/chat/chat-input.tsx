"use client";
import { useState, useRef, useCallback, KeyboardEvent } from "react";

interface ChatInputProps {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, onStop, isStreaming, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
  }, [value, isStreaming, onSend]);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background px-8 py-4">
      <div className="max-w-chat mx-auto flex gap-2 items-center">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your documents..."
          disabled={disabled}
          className="flex-1 font-sans text-sm px-3.5 py-2.5 border border-border rounded-md bg-surface text-text outline-none transition-colors duration-150 focus:border-accent placeholder:text-text-muted disabled:opacity-50"
          aria-label="Ask a question"
        />
        {isStreaming ? (
          <button
            onClick={onStop}
            className="w-10 h-10 rounded-md bg-error flex items-center justify-center shrink-0 transition-colors hover:opacity-90"
            aria-label="Stop streaming"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="white">
              <rect width="14" height="14" rx="2" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!value.trim() || disabled}
            className="w-10 h-10 rounded-md bg-accent flex items-center justify-center shrink-0 transition-colors hover:bg-accent-hover disabled:opacity-50"
            aria-label="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 2L11 13" />
              <path d="M22 2L15 22L11 13L2 9L22 2Z" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
