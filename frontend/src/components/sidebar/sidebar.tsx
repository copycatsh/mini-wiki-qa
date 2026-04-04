"use client";
import { Pipeline } from "@/types/chat";
import { ThemeToggle } from "./theme-toggle";

interface SidebarProps {
  pipeline: Pipeline;
  onPipelineChange: (pipeline: Pipeline) => void;
  useRerank: boolean;
  onRerankChange: (useRerank: boolean) => void;
}

export function Sidebar({ pipeline, onPipelineChange, useRerank, onRerankChange }: SidebarProps) {
  return (
    <aside className="w-sidebar min-w-[240px] bg-sidebar border-r border-border px-4 py-5 flex flex-col gap-6 max-lg:hidden">
      <h1 className="font-serif text-xl text-text">Mini Wiki Q&A</h1>

      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2.5">Pipeline</div>
        <select
          value={pipeline}
          onChange={(e) => onPipelineChange(e.target.value as Pipeline)}
          className="w-full font-sans text-[13px] px-2.5 py-1.5 border border-border rounded-sm bg-surface text-text outline-none appearance-none cursor-pointer focus:border-accent"
          aria-label="Select pipeline"
        >
          <option value="/ask">/ask</option>
          <option value="/ask-graph">/ask-graph</option>
        </select>
      </div>

      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2.5">Settings</div>
        <div className="flex items-center justify-between py-1.5">
          <span className="font-sans text-[13px] text-text">Reranking</span>
          <button
            role="switch"
            aria-checked={useRerank}
            aria-label="Enable reranking"
            onClick={() => onRerankChange(!useRerank)}
            className={`relative w-10 h-[22px] rounded-full transition-colors duration-200 ${useRerank ? "bg-accent" : "bg-border"}`}
          >
            <span className={`absolute top-0.5 left-0.5 w-[18px] h-[18px] bg-white rounded-full shadow-sm transition-transform duration-200 ${useRerank ? "translate-x-[18px]" : ""}`} />
          </button>
        </div>
      </div>

      <div className="mt-auto">
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2.5">Info</div>
        <div className="font-mono text-[11px] text-text-muted leading-relaxed">
          Pipeline: {pipeline}<br />
          Rerank: {useRerank ? "on" : "off"}
        </div>
      </div>

      <ThemeToggle />
    </aside>
  );
}
