"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";

interface EmptyStateProps {
  onSuggestionClick: (query: string) => void;
}

const SUGGESTIONS = [
  "What is retrieval-augmented generation?",
  "How does the reranking pipeline work?",
  "What safety mechanisms are in place?",
  "How are documents chunked and indexed?",
];

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
  });

  return (
    <div className="flex-1 flex items-center justify-center px-8 pb-16">
      <div className="text-center max-w-md">
        <h1 className="font-serif text-[32px] text-text leading-tight">
          Ask your documents anything
        </h1>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => onSuggestionClick(suggestion)}
              className="px-4 py-2.5 text-[13px] font-sans border border-border rounded-sm text-text-muted hover:border-accent hover:text-accent transition-colors duration-150"
            >
              {suggestion}
            </button>
          ))}
        </div>
        {health && (
          <p className="mt-4 text-[11px] font-mono text-text-muted">
            {health.status === "healthy" ? "Ready" : "Service unavailable"}
          </p>
        )}
      </div>
    </div>
  );
}
