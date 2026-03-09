import argparse
import json
import os
from typing import List, Dict, Any


def load_hotpotqa(path: str) -> List[Dict[str, Any]]:
    """Load original HotpotQA JSON (list of dicts)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at top level in {path}, got {type(data)}")
    return data


def convert_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw HotpotQA example to the RT-RAG format expected by load_data.py:
      - _id: original id
      - input: question text
      - answers: list of acceptable answers
    """
    qid = ex.get("_id")
    question = ex.get("question")
    answer = ex.get("answer")

    if qid is None or question is None or answer is None:
        raise ValueError(f"Missing required keys in example: {ex.keys()}")

    # HotpotQA's 'answer' is usually a single string; wrap it in a list.
    if isinstance(answer, list):
        answers = answer
    else:
        answers = [str(answer)]

    return {
        "_id": qid,
        "input": question,
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a HotpotQA dev_1k JSONL subset in the format expected by RT-RAG."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="main/raw/hotpotqa.json",
        help="Path to original HotpotQA JSON file (list of dicts).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="main/data/hotpotqa_dev_1k.jsonl",
        help="Output JSONL path for the 1k subset.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of examples to keep (default: 1000).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index in the source list (default: 0).",
    )

    args = parser.parse_args()

    data = load_hotpotqa(args.source)
    if args.offset < 0 or args.offset >= len(data):
        raise ValueError(f"Offset {args.offset} is out of range for dataset of size {len(data)}")

    end = min(args.offset + args.num_samples, len(data))
    subset = data[args.offset:end]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    converted = [convert_example(ex) for ex in subset]

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in converted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(converted)} examples to {args.output} "
        f"(source {args.source}, indices [{args.offset}, {end}))"
    )


if __name__ == "__main__":
    main()

