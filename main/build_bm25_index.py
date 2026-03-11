"""
Build a local BM25 index from the corpus JSON.
Reuses the same chunking logic as the dense index builder.

Outputs (saved to embedding_data/{dataset}/{chunk_size}_{min_sentence}_{overlap}/):
  - chunks.json        : list of text chunks
  - bm25_index.pkl     : serialized BM25Okapi object
  - id_to_rawid.json   : chunk-id → source-article-id mapping
"""

import json
import os
import re
import pickle
import gc
import ijson

from tqdm import tqdm
from rank_bm25 import BM25Okapi


def get_word_count(text):
    regex = re.compile(r'[\W]')
    words = regex.split(text.lower())
    return len([w for w in words if len(w.strip()) > 0])


def split_sentences(content, chunk_size, min_sentence, overlap):
    """Split content into chunks based on sentence delimiters and constraints.
    (Copied from build_dense_index/dense_build_index.py to avoid faiss import.)
    """
    stop_list = ['!', '。', '，', '！', '?', '？', ',', '.', ';']
    split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
    sentences = re.split(split_pattern, content)

    if len(sentences) == 1:
        return sentences

    sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]
    chunks = []
    temp_text = ''
    sentence_overlap_len = 0
    start_index = 0

    for i, sentence in enumerate(sentences):
        temp_text += sentence
        if get_word_count(temp_text) >= chunk_size - sentence_overlap_len or i == len(sentences) - 1:
            if i + 1 > overlap:
                sentence_overlap_len = sum(get_word_count(sentences[j]) for j in range(i + 1 - overlap, i + 1))
            if chunks:
                if start_index > overlap:
                    start_index -= overlap
            chunk_text = ''.join(sentences[start_index:i + 1])
            if not chunks:
                chunks.append(chunk_text)
            elif i == len(sentences) - 1 and (i - start_index + 1) < min_sentence:
                chunks[-1] += chunk_text
            else:
                chunks.append(chunk_text)
            temp_text = ''
            start_index = i + 1

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a local BM25 index from corpus JSON.")
    parser.add_argument("--raw_path", default="main/raw", help="Directory containing corpus JSON files.")
    parser.add_argument("--dataset", default="wiki_psgs", help="Dataset name (filename without .json).")
    parser.add_argument("--save_dir", default=None, help="Override output directory.")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--min_sentence", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--stream", action="store_true", default=True,
                        help="Stream JSON with ijson to reduce peak memory (default: True).")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Load entire JSON into memory (faster for small corpora).")
    args = parser.parse_args()

    file_path = os.path.join(args.raw_path, f"{args.dataset}.json")
    save_dir = args.save_dir or os.path.join(
        "embedding_data", args.dataset, f"{args.chunk_size}_{args.min_sentence}_{args.overlap}"
    )

    print(f"Corpus  : {file_path}")
    print(f"Output  : {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # --- Phase 1: chunk and write chunks.json + id_to_rawid.json incrementally ---
    id_to_rawid = {}
    chunk_count = 0
    chunks_path = os.path.join(save_dir, "chunks.json")

    print("Phase 1: Chunking corpus and writing chunks to disk ...")
    with open(chunks_path, "w", encoding="utf-8") as fout:
        fout.write("[")
        first = True

        if args.stream:
            with open(file_path, "rb") as fin:
                items = ijson.items(fin, "item")
                for idx, item in tqdm(enumerate(items), desc="Chunking (streaming)"):
                    content = item.get("paragraph_text") or item.get("ch_content") or ""
                    if not content.strip():
                        continue
                    item_chunks = split_sentences(content, args.chunk_size, args.min_sentence, args.overlap)
                    for i, chunk in enumerate(item_chunks):
                        id_to_rawid[chunk_count + i] = idx
                        if not first:
                            fout.write(",")
                        json.dump(chunk, fout, ensure_ascii=False)
                        first = False
                    chunk_count += len(item_chunks)
        else:
            with open(file_path, encoding="utf-8") as fin:
                data = json.load(fin)
            for idx, item in tqdm(enumerate(data), total=len(data), desc="Chunking"):
                content = item.get("paragraph_text") or item.get("ch_content") or ""
                if not content.strip():
                    continue
                item_chunks = split_sentences(content, args.chunk_size, args.min_sentence, args.overlap)
                for i, chunk in enumerate(item_chunks):
                    id_to_rawid[chunk_count + i] = idx
                    if not first:
                        fout.write(",")
                    json.dump(chunk, fout, ensure_ascii=False)
                    first = False
                chunk_count += len(item_chunks)
            del data
            gc.collect()

        fout.write("]")

    print(f"Total chunks: {chunk_count:,}")

    with open(os.path.join(save_dir, "id_to_rawid.json"), "w", encoding="utf-8") as f:
        json.dump(id_to_rawid, f, ensure_ascii=False)
    del id_to_rawid
    gc.collect()

    # --- Phase 2: read chunks back, tokenize, and build BM25 ---
    print("Phase 2: Tokenizing and building BM25 index ...")
    tokenized = []
    with open(chunks_path, "rb") as f:
        for chunk in tqdm(ijson.items(f, "item"), total=chunk_count, desc="Tokenizing"):
            tokenized.append(tokenize(chunk))

    print("Building BM25Okapi index ...")
    bm25 = BM25Okapi(tokenized)
    del tokenized
    gc.collect()

    with open(os.path.join(save_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print(f"Saved chunks.json, id_to_rawid.json, bm25_index.pkl → {save_dir}")


if __name__ == "__main__":
    main()
