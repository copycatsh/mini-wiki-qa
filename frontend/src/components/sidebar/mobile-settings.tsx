"use client";
import { useState } from "react";
import { Pipeline } from "@/types/chat";

interface MobileSettingsProps {
  pipeline: Pipeline;
  onPipelineChange: (pipeline: Pipeline) => void;
  useRerank: boolean;
  onRerankChange: (useRerank: boolean) => void;
  useMultiQuery: boolean;
  onMultiQueryChange: (useMultiQuery: boolean) => void;
}

export function MobileSettings({ pipeline, onPipelineChange, useRerank, onRerankChange, useMultiQuery, onMultiQueryChange }: MobileSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Mobile header - only visible below lg breakpoint */}
      <div className="lg:hidden flex items-center justify-between px-4 py-3 border-b border-border bg-background">
        <span className="font-serif text-lg text-text">Mini Wiki Q&A</span>
        <button
          onClick={() => setIsOpen(true)}
          className="w-8 h-8 flex items-center justify-center text-text-muted hover:text-accent transition-colors"
          aria-label="Open settings"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
          </svg>
        </button>
      </div>

      {/* Backdrop + Bottom Sheet */}
      {isOpen && (
        <>
          <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40" onClick={() => setIsOpen(false)} />
          <div className="fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border rounded-t-lg max-h-[280px] p-4" role="dialog" aria-label="Settings">
            <div className="w-10 h-1 bg-border rounded-full mx-auto mb-4" />
            <div className="mb-4">
              <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2">Pipeline</div>
              <select
                value={pipeline}
                onChange={(e) => onPipelineChange(e.target.value as Pipeline)}
                className="w-full font-sans text-sm px-3 py-2 border border-border rounded-sm bg-background text-text outline-none"
                aria-label="Select pipeline"
              >
                <option value="/ask">/ask</option>
                <option value="/ask-graph">/ask-graph</option>
              </select>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="font-sans text-sm text-text">Reranking</span>
              <button
                role="switch"
                aria-checked={useRerank}
                aria-label="Enable reranking"
                onClick={() => onRerankChange(!useRerank)}
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${useRerank ? "bg-accent" : "bg-border"}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${useRerank ? "translate-x-5" : ""}`} />
              </button>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="font-sans text-sm text-text">Multi-query</span>
              <button
                role="switch"
                aria-checked={useMultiQuery}
                aria-label="Enable multi-query expansion"
                onClick={() => onMultiQueryChange(!useMultiQuery)}
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${useMultiQuery ? "bg-accent" : "bg-border"}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${useMultiQuery ? "translate-x-5" : ""}`} />
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
}
