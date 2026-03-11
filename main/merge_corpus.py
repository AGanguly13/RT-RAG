"""
Merge the DPR Wikipedia passages corpus with HotPotQA gold context articles.

This ensures the gold supporting passages for the 1k eval questions exist in the
search index while keeping a realistically large retrieval corpus (~1.5M passages).
Deduplicates by title so there are no exact duplicates.

Streams the wiki JSON with ijson to keep memory low.

Usage:
    python main/merge_corpus.py \
        --wiki main/raw/wiki_psgs.json \
        --hotpot hotpotqa_dev_1k.json \
        --output main/raw/wiki_psgs_merged.json
"""

import argparse
import json
import os
import ijson
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Merge Wikipedia passages with HotPotQA gold context articles."
    )
    parser.add_argument(
        "--wiki",
        type=str,
        default="main/raw/wiki_psgs.json",
        help="Path to the Wikipedia passages corpus JSON.",
    )
    parser.add_argument(
        "--hotpot",
        type=str,
        default="hotpotqa_dev_1k.json",
        help="Path to original HotPotQA JSON with context field.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="main/raw/wiki_psgs_merged.json",
        help="Output path for the merged corpus.",
    )
    args = parser.parse_args()

    if args.output == args.wiki:
        print("ERROR: --output must differ from --wiki (streaming can't overwrite in place).")
        print("       After verifying, you can: mv <output> <wiki>")
        return

    # --- Extract HotPotQA gold context articles first (small, fits in memory) ---
    print(f"Loading HotPotQA from {args.hotpot} ...")
    with open(args.hotpot, "r", encoding="utf-8") as f:
        hotpot_data = json.load(f)

    gold_by_title = {}
    for ex in hotpot_data:
        for title, sentences in ex.get("context", []):
            if title not in gold_by_title:
                full_text = " ".join(sentences).strip()
                if full_text:
                    gold_by_title[title] = {
                        "title": title,
                        "paragraph_text": f"{title}\n{full_text}",
                    }
    del hotpot_data

    print(f"  Gold articles: {len(gold_by_title):,}")

    # --- Stream wiki corpus, collect titles, write to output ---
    print(f"Streaming wiki corpus from {args.wiki} ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    seen_titles = set()
    count = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write("[")
        first = True

        with open(args.wiki, "rb") as fin:
            for item in tqdm(ijson.items(fin, "item"), desc="Streaming wiki"):
                title = (item.get("title") or "").strip()
                if title:
                    seen_titles.add(title)

                if not first:
                    fout.write(",")
                json.dump(item, fout, ensure_ascii=False)
                first = False
                count += 1

        # Append gold articles not already in wiki
        added = 0
        for title, article in gold_by_title.items():
            if title not in seen_titles:
                fout.write(",")
                json.dump(article, fout, ensure_ascii=False)
                added += 1
                count += 1

        fout.write("]")

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Wiki passages kept: {count - added:,}")
    print(f"  Gold articles added: {added:,}")
    print(f"  Total: {count:,} passages, {size_mb:.1f} MB → {args.output}")


if __name__ == "__main__":
    main()
