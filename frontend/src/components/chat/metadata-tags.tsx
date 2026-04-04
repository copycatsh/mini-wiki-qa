interface MetadataTagsProps {
  metadata: Record<string, unknown>;
}

export function MetadataTags({ metadata }: MetadataTagsProps) {
  const pipeline = metadata.pipeline as string | undefined;
  const chunksRetrieved = metadata.chunks_retrieved as number | undefined;
  const useRerank = metadata.use_rerank as boolean | undefined;

  return (
    <div className="flex gap-2 mt-2 flex-wrap">
      {chunksRetrieved !== undefined && (
        <span className="font-mono text-[11px] px-2 py-0.5 bg-code-bg text-text-muted rounded-sm">
          {chunksRetrieved} chunks
        </span>
      )}
      {useRerank && (
        <span className="font-mono text-[11px] px-2 py-0.5 bg-success/10 text-success rounded-sm">
          reranked
        </span>
      )}
      {pipeline && (
        <span className="font-mono text-[11px] px-2 py-0.5 bg-accent/10 text-accent rounded-sm">
          {pipeline === "langgraph" ? "/ask-graph" : "/ask"}
        </span>
      )}
    </div>
  );
}
