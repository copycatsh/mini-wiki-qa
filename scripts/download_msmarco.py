"""Download MS MARCO dataset and convert to markdown files"""
from datasets import load_dataset
from pathlib import Path
import json

def main():
    print("📦 Downloading MS MARCO dataset...")

    dataset = load_dataset("ms_marco", "v1.1", split="train[:150]")

    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "data" / "documents" / "msmarco"
    golden_dir = project_root / "data" / "golden_set"

    docs_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"📝 Processing {len(dataset)} examples...")

    golden_set = []

    for idx, example in enumerate(dataset):
        doc_path = docs_dir / f"doc_{idx:03d}.md"

        passages = example["passages"]["passage_text"]
        context = "\n\n".join(passages)
        content = f"# Document {idx}\n\n{context}"
        doc_path.write_text(content, encoding='utf-8')

        answers = example["answers"]
        answer = answers[0] if answers else "No answer available"

        golden_set.append({
            "question": example["query"],
            "answer": answer,
            "query_type": example["query_type"],
            "document": f"doc_{idx:03d}.md",
            "context": context[:200] + "..."
        })

    golden_path = golden_dir / "msmarco_qa.json"
    with open(golden_path, 'w', encoding='utf-8') as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)

    print(f"✅ Created {len(dataset)} documents in {docs_dir}")
    print(f"✅ Created golden set with {len(golden_set)} Q&A pairs in {golden_path}")
    print("\n📊 Example Q&A:")
    print(f"Q: {golden_set[0]['question']}")
    print(f"A: {golden_set[0]['answer']}")

if __name__ == "__main__":
    main()
