"use client";
import { useState } from "react";
import { Citation } from "@/types/chat";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());

  const toggleChunk = (idx: number) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  return (
    <div className="mt-2.5">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 bg-code-bg rounded-sm text-xs font-sans text-text-muted hover:text-accent transition-colors duration-150"
        aria-expanded={isOpen}
      >
        <span
          className="text-[10px] transition-transform duration-150"
          style={{ transform: isOpen ? "rotate(90deg)" : "rotate(0deg)" }}
        >
          &#9654;
        </span>
        {citations.length} source{citations.length !== 1 ? "s" : ""}
      </button>

      {isOpen && (
        <div className="mt-2 p-2.5 bg-surface border border-border rounded-sm">
          {citations.map((citation, idx) => (
            <div key={idx} className="py-1 flex gap-2 items-baseline">
              <span className="font-mono text-[10px] bg-code-bg px-1.5 py-0.5 rounded-sm text-accent font-medium shrink-0">
                {idx + 1}
              </span>
              <div className="min-w-0">
                <span className="text-xs font-sans text-accent">
                  {citation.document}
                </span>
                <span className="text-[11px] font-mono text-text-muted ml-2">
                  {(citation.score * 100).toFixed(0)}%
                </span>
                <p className="text-[13px] font-body text-text-muted mt-0.5 leading-relaxed">
                  {expandedChunks.has(idx)
                    ? citation.text
                    : citation.text.length > 200
                      ? citation.text.slice(0, 200) + "..."
                      : citation.text}
                  {citation.text.length > 200 && (
                    <button
                      onClick={() => toggleChunk(idx)}
                      className="ml-1 text-accent text-xs hover:underline"
                    >
                      {expandedChunks.has(idx) ? "Show less" : "Show full text"}
                    </button>
                  )}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
