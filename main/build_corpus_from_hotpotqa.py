"""
Extract all unique Wikipedia context passages from the HotpotQA dev set
and create a retrieval corpus in the format expected by RT-RAG's indexing pipeline.

Each output item has {"title": ..., "paragraph_text": ...} matching
what dense_build_index.py and the BM25 indexer expect.
"""

import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Build RT-RAG retrieval corpus from HotpotQA context passages."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="hotpotqa_dev_1k.json",
        help="Path to HotpotQA JSON file with context field.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="main/raw/hotpotqa_dev_1k.json",
        help="Output path for the corpus JSON (list of {title, paragraph_text}).",
    )
    args = parser.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen_titles = set()
    corpus = []

    for item in data:
        for title, sentences in item["context"]:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            full_text = " ".join(sentences).strip()
            if full_text:
                corpus.append({
                    "title": title,
                    "paragraph_text": f"{title}\n{full_text}",
                })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Built corpus with {len(corpus)} unique articles → {args.output}")


if __name__ == "__main__":
    main()
