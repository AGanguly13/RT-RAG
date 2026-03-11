import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import load_dataset

import config
from retrieve import search_with_bm25


def load_hotpotqa_with_supports(path: str) -> Dict[str, Dict]:
    """Load original HotpotQA-style JSON and index by _id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {}
    for ex in data:
        qid = ex.get("_id")
        if qid is not None:
            by_id[qid] = ex
    return by_id


def get_gold_support_titles(ex: Dict) -> List[str]:
    """Extract unique supporting article titles from a HotpotQA example."""
    sfs = ex.get("supporting_facts", [])
    titles = {sf[0] for sf in sfs if isinstance(sf, list) and len(sf) >= 1}
    return list(titles)


# ---------------- Retrieval metrics (Recall@k, supporting-fact retrieval) ----------------

def eval_retrieval(
    hotpot_path: str,
    raw_corpus_path: str,
    k: int = 10,
) -> None:
    """
    Compute Retrieval Recall@k and supporting-fact retrieval accuracy using BM25.

    - hotpot_path: original HotpotQA dev JSON with `supporting_facts`.
    - raw_corpus_path: the RT-RAG corpus JSON (e.g. main/raw/wiki_psgs.json).
    """
    print(f"Loading HotpotQA questions from: {hotpot_path}")
    hotpot_by_id = load_hotpotqa_with_supports(hotpot_path)
    all_ids = list(hotpot_by_id.keys())
    print(f"Total questions: {len(all_ids)}")

    # Load mapping from chunk index -> raw article index, and raw titles
    base_path = os.path.join(
        config.EMBEDDING_DATA,
        config.DATASET,
        f"{config.CHUNK_SIZE}_{config.MIN_SENTENCE}_{config.OVERLAP}",
    )
    id_to_rawid_path = os.path.join(base_path, "id_to_rawid.json")
    with open(id_to_rawid_path, "r", encoding="utf-8") as f:
        id_to_rawid = {int(k): v for k, v in json.load(f).items()}

    with open(raw_corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    def chunk_idx_to_title(idx: int) -> str:
        raw_idx = id_to_rawid.get(idx)
        if raw_idx is None or raw_idx < 0 or raw_idx >= len(corpus):
            return ""
        return corpus[raw_idx].get("title", "") or ""

    total_recall = 0.0
    total_full_hit = 0
    n_questions = 0

    for qid, ex in hotpot_by_id.items():
        question = ex.get("question", "")
        gold_titles = get_gold_support_titles(ex)
        if not gold_titles:
            continue

        # Use same BM25 index & params as RT-RAG
        results = search_with_bm25(
            query=question,
            dataset=config.DATASET,
            chunk_size=config.CHUNK_SIZE,
            min_sentence=config.MIN_SENTENCE,
            overlap=config.OVERLAP,
            top_k=k,
        )

        retrieved_titles = set()
        for r in results:
            try:
                idx = int(r.get("id", "-1"))
            except Exception:
                continue
            title = chunk_idx_to_title(idx)
            if title:
                retrieved_titles.add(title)

        hits = sum(1 for t in gold_titles if t in retrieved_titles)
        recall = hits / max(1, len(gold_titles))

        total_recall += recall
        if hits == len(gold_titles):
            total_full_hit += 1
        n_questions += 1

    if n_questions == 0:
        print("No questions with supporting_facts found; cannot compute retrieval metrics.")
        return

    avg_recall = total_recall / n_questions
    full_hit_acc = total_full_hit / n_questions

    print("\n=== Retrieval Metrics (BM25) ===")
    print(f"Recall@{k}: {avg_recall:.4f} ({avg_recall * 100:.2f}%)")
    print(f"Supporting-fact retrieval accuracy (all gold titles in top-{k}): "
          f"{full_hit_acc:.4f} ({full_hit_acc * 100:.2f}%)")


# ---------------- Reasoning metrics (supporting-fact prediction, chain completeness) ----------------

def parse_qa_results(file_path: str) -> List[Dict]:
    """Reuse the same parsing logic as evaulate.py to read QA blocks."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("---\n")
    qa_pairs = []
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        qa = {}
        for line in lines:
            if line.startswith("qid:"):
                qa["qid"] = line.replace("qid:", "").strip()
            elif line.startswith("question:"):
                qa["question"] = line.replace("question:", "").strip()
            elif line.startswith("predicted_answer:"):
                qa["predicted_answer"] = line.replace("predicted_answer:", "").strip()
        if "qid" in qa and "predicted_answer" in qa:
            qa_pairs.append(qa)
    return qa_pairs


def normalize_text(s: str) -> str:
    """Simple normalization for substring checks."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def eval_reasoning_supports(
    hotpot_path: str,
    result_file: str,
) -> None:
    """
    Approximate reasoning metrics from existing outputs:

    - supporting-fact prediction accuracy (titles mentioned in prediction)
    - evidence-chain completeness (average title recall)
    """
    print(f"Loading HotpotQA questions from: {hotpot_path}")
    hotpot_by_id = load_hotpotqa_with_supports(hotpot_path)

    print(f"Loading QA results from: {result_file}")
    qa_pairs = parse_qa_results(result_file)
    print(f"Total QA pairs in results: {len(qa_pairs)}")

    n_eval = 0
    n_full_hit = 0
    sum_title_recall = 0.0

    for qa in qa_pairs:
        qid = qa["qid"]
        ex = hotpot_by_id.get(qid)
        if ex is None:
            continue
        gold_titles = get_gold_support_titles(ex)
        if not gold_titles:
            continue

        pred = normalize_text(qa["predicted_answer"])
        hits = 0
        for t in gold_titles:
            t_norm = normalize_text(t)
            if t_norm and t_norm in pred:
                hits += 1

        recall = hits / max(1, len(gold_titles))
        sum_title_recall += recall
        if hits == len(gold_titles):
            n_full_hit += 1
        n_eval += 1

    if n_eval == 0:
        print("No overlapping questions between results and HotpotQA; cannot compute reasoning metrics.")
        return

    avg_title_recall = sum_title_recall / n_eval
    full_hit_acc = n_full_hit / n_eval

    print("\n=== Reasoning Metrics (approximate, title-based) ===")
    print("We treat a gold supporting title as 'predicted' if it appears (normalized) in the predicted_answer text.")
    print(f"Supporting-fact prediction accuracy (all gold titles mentioned): "
          f"{full_hit_acc:.4f} ({full_hit_acc * 100:.2f}%)")
    print(f"Evidence-chain completeness (average title recall): "
          f"{avg_title_recall:.4f} ({avg_title_recall * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and reasoning metrics for RT-RAG on HotpotQA."
    )
    parser.add_argument(
        "--hotpot_path",
        type=str,
        default="hotpotqa_dev_1k.json",
        help="Path to original HotpotQA dev JSON with supporting_facts.",
    )
    parser.add_argument(
        "--raw_corpus_path",
        type=str,
        default="main/raw/wiki_psgs.json",
        help="Path to RT-RAG corpus JSON used for indexing.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="RT-RAG result file (outputs/.../1.txt) to evaluate reasoning metrics.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k for Recall@k (retrieval).",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval eval (avoids loading the large corpus JSON into RAM).",
    )
    args = parser.parse_args()

    if not args.skip_retrieval:
        eval_retrieval(
            hotpot_path=args.hotpot_path,
            raw_corpus_path=args.raw_corpus_path,
            k=args.k,
        )
    else:
        print("Skipping retrieval evaluation (--skip-retrieval).")

    eval_reasoning_supports(
        hotpot_path=args.hotpot_path,
        result_file=args.results_file,
    )


if __name__ == "__main__":
    main()

