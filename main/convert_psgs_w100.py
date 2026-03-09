"""
Convert DPR's psgs_w100.tsv (Wikipedia passages) to the JSON corpus format
expected by RT-RAG's indexing pipeline.

TSV columns: id \t text \t title

Usage:
    # Download first (if not already):
    #   wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
    #   gunzip psgs_w100.tsv.gz

    # Keep ~1 GB of passages (default):
    python main/convert_psgs_w100.py --input psgs_w100.tsv --output main/raw/wiki_psgs.json

    # Keep all passages (~14 GB JSON):
    python main/convert_psgs_w100.py --input psgs_w100.tsv --output main/raw/wiki_psgs.json --max_passages 0
"""

import argparse
import csv
import json
import os
import sys
from tqdm import tqdm


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Convert psgs_w100.tsv to RT-RAG corpus JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="psgs_w100.tsv",
        help="Path to DPR psgs_w100.tsv file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="main/raw/wiki_psgs.json",
        help="Output JSON path for the corpus.",
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=1_500_000,
        help="Max passages to keep (0 = all). Default 1.5M ≈ 1 GB of text.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: {args.input} not found.")
        print("Download it first:")
        print("  wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz")
        print("  gunzip psgs_w100.tsv.gz")
        return

    limit = args.max_passages if args.max_passages > 0 else float("inf")

    print(f"Reading {args.input} ...")
    if limit < float("inf"):
        print(f"Keeping first {limit:,} passages")

    corpus = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Converting passages"):
            title = row.get("title", "").strip()
            text = row.get("text", "").strip()
            if not text:
                continue
            corpus.append({
                "title": title,
                "paragraph_text": f"{title}\n{text}" if title else text,
            })
            if len(corpus) >= limit:
                break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Writing {len(corpus):,} passages to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Corpus: {len(corpus):,} passages, {size_mb:.1f} MB → {args.output}")


if __name__ == "__main__":
    main()
