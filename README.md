

# 🧠🌳Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering




## 🚀 Quick Start: End-to-End (HotPotQA 1k + Wikipedia BM25)

This section describes how to run the full pipeline: build a retrieval corpus from DPR Wikipedia passages, serve the LLM, run RT-RAG on the 1k HotPotQA dev set, and evaluate.

### 1. Environment

```bash
cd RT-RAG
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Retrieval corpus (Wikipedia ~1 GB)

Use scratch (or another large filesystem) to avoid home-directory quota limits.

```bash
mkdir -p /scratch/$USER/wiki_corpus
cd /scratch/$USER/wiki_corpus
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz
```

Convert to corpus JSON (keeps first ~1.5M passages ≈ 1 GB):

```bash
cd /path/to/RT-RAG
ln -s /scratch/$USER/wiki_corpus/psgs_w100.tsv psgs_w100.tsv   # if not in repo

python main/convert_psgs_w100.py \
  --input psgs_w100.tsv \
  --output /scratch/$USER/wiki_corpus/wiki_psgs.json \
  --max_passages 1500000
```

Point the repo at the corpus and index on scratch:

```bash
ln -sf /scratch/$USER/wiki_corpus/wiki_psgs.json main/raw/wiki_psgs.json
mkdir -p /scratch/$USER/embedding_data
ln -sf /scratch/$USER/embedding_data embedding_data   # if not already
```

Build the BM25 index:

```bash
python main/build_bm25_index.py --dataset wiki_psgs
```

### 3. LLM server (vLLM)

Use scratch for the Hugging Face cache to avoid quota:

```bash
export HF_HOME=/scratch/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/hf_cache
mkdir -p /scratch/$USER/hf_cache
```

Start vLLM (from the repo root or anywhere):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

Optional: higher throughput with `--max-num-seqs 32 --gpu-memory-utilization 0.95`.

### 4. Run evaluation (1k HotPotQA questions)

In another terminal, with the vLLM server running:

```bash
cd /path/to/RT-RAG
python main/load_data.py
```

Predictions are appended to:

`outputs/wiki_psgs/bm25_chunk200_topk1_45_topk2_10/1.txt`

(Config in `main/config.py`: `DATA_PATH`, `DATASET`, `METHOD`, etc.)

### 5. Evaluate results

**EM & F1 (answer accuracy):**

```bash
python main/evaulate.py outputs/wiki_psgs/bm25_chunk200_topk1_45_topk2_10/1.txt
```

**Retrieval & reasoning metrics (Recall@k, supporting-fact accuracy):**

```bash
python main/eval_retrieval_and_reasoning.py \
  --hotpot_path hotpotqa_dev_1k.json \
  --raw_corpus_path main/raw/wiki_psgs.json \
  --results_file outputs/wiki_psgs/bm25_chunk200_topk1_45_topk2_10/1.txt \
  --k 10
```

`hotpotqa_dev_1k.json` is the original HotPotQA dev JSON with `supporting_facts` (for gold titles). If it's not in the repo root, pass the full path.

### Summary

| Step | Command / output |
|------|-------------------|
| Corpus | `convert_psgs_w100.py` → `main/raw/wiki_psgs.json` |
| Index | `build_bm25_index.py --dataset wiki_psgs` → `embedding_data/wiki_psgs/200_2_2/` |
| LLM | `vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000` |
| Inference | `load_data.py` → `outputs/wiki_psgs/.../1.txt` |
| EM/F1 | `evaulate.py <path-to-1.txt>` |
| Retrieval/reasoning | `eval_retrieval_and_reasoning.py --results_file <path-to-1.txt>` |

---

## 📋 Important changes from the original RT-RAG codebase

| Area | Original behavior | Change made |
|------|-------------------|-------------|
| **Retrieval corpus** | Corpus was built from HotPotQA dev *context passages* (`build_corpus_from_hotpotqa.py` → `main/raw/hotpotqa_dev_1k.json`). The same 1k examples were both searched and evaluated, so retrieval was artificially easy. | **Separated corpus from eval:** retrieval now uses DPR Wikipedia passages (`psgs_w100.tsv` → `wiki_psgs.json`). The 1k HotPotQA questions are **evaluation-only**; they are not part of the search index. |
| **Corpus pipeline** | Only script to build a corpus was from HotPotQA contexts. | **New `main/convert_psgs_w100.py`** converts DPR's `psgs_w100.tsv` to the JSON format expected by the indexers. Optional `--max_passages` (default 1.5M) limits corpus size to ~1 GB for memory and disk. |
| **Config** | `DATASET = "hotpotqa_dev_1k"`; index/output paths referred to that corpus. | **`DATASET = "wiki_psgs"`** everywhere (main `config.py`, `build_bm25_index.py`, `build_dense_index/config.py`). `DATA_PATH` still points to `main/data/hotpotqa_dev_1k.jsonl` for the 1k eval questions. |
| **BM25 indexer** | Loaded the full corpus JSON into memory (`json.load()`), then built the index. Large corpora (e.g. full Wikipedia) caused OOM. | **Streaming build:** `main/build_bm25_index.py` uses **ijson** to stream the JSON and writes chunks to disk in phase 1, then builds BM25 in phase 2. Enables indexing ~1 GB+ corpora on limited RAM. Added **ijson** to `requirements.txt`. |
| **Eval script** | `eval_retrieval_and_reasoning.py` default `--raw_corpus_path` was `main/raw/hotpotqa_dev_1k.json`. | Default set to **`main/raw/wiki_psgs.json`** so retrieval metrics use the same corpus as the index. |
| **Speed / throughput** | Defaults: 5 trees/question, height 4, 5 sampling iterations, 4 query-rewrite iterations, 8 concurrent jobs. | **Tuned for faster runs:** fewer trees (3), lower max height (3), fewer sampling (3) and rewrite (2) iterations, higher concurrency (16). All in `main/config.py`; revert if you need original quality/settings. |
| **Documentation** | README focused on LongBench, dense index, 14B model. | **Quick Start** added for HotPotQA 1k + Wikipedia BM25, scratch storage, vLLM (1.5B), and the full eval commands. |

---

## 🔍 What is RT-RAG? 
![RT-RAG Overview](assets/overview.png)
**RT-RAG** systematically decomposes complex multi-hop questions into explicit **binary reasoning trees**. It leverages structured entity analysis and **consensus-based tree selection** to ensure e decomposition, clearly separating core queries, known entities, and unknown targets.

Once the tree is built, a **bottom-up traversal strategy** is used to iteratively rewrite and refine sub-questions. This process efficiently collects high-quality evidence while mitigating error propagation through recursive reasoning.



---

## ⚙️ 1. Environment Setup

### ✅ Install Dependencies

```bash
pip install -r requirements.txt
```

### ⚡️ (Optional) Serve Qwen2.5-14B-Instruct via vLLM

To serve Qwen2.5-14B-Instruct locally using [vLLM](https://github.com/vllm-project/vllm) with OpenAI-compatible API:

First, install vLLM:

```bash
pip install vllm
```

Then, start the server:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --dtype auto \
  --api-key your-api-key
```

> Replace `your-api-key` with a secure token. This key must match what you configure in `config.py`.

📝 **Tip:** For more details, see [vLLM OpenAI-Compatible Server Docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)


---

## 📦 2. Model Downloads

You can download models manually or use Hugging Face CLI:

### 🔍 Reranker Model

- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)

```bash
huggingface-cli download BAAI/bge-reranker-base
```

### 🧠 Language Model (Qwen2.5-14B-Instruct)

- [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```

> Make sure to login if authentication is required:

```bash
huggingface-cli login
```

---

## 🛠️ 3. Data Preparation

The preprocessed corpus is already in the `raw` folder.  
Evaluation and retrieval data are from [LongBench](https://github.com/THUDM/LongBench).

---

## ✏️ 4. Configure `main/build_dense_index/config.py`

Update your configuration for embedding/index building:

| Parameter       | Description |
|----------------|-------------|
| `raw_path`     | Path to folder containing preprocessed JSON |
| `save_path`    | Where to store FAISS index & metadata |
| `dataset_name` | Filename without `.json` |
| `chunk_size`   | Max words per chunk (e.g., 200) |
| `min_sentence` | Min sentences per chunk (e.g., 2) |
| `overlap`      | Overlapping sentences between chunks (e.g., 2) |
| `base_url`     | API endpoint (e.g., `http://localhost:8000/v1`) |
| `api_key`      | Your API key used with the embedding service |

---

## 🧱 5. Build the Dense Index

Once `main/build_dense_index/config.py` is ready, build your FAISS index with:

```bash
python build_dense_index/dense_build_index.py
```

---

---

## 🧪 6. Run on the Full Dataset

After the dense index is successfully built:

1. Configure runtime parameters in:

    ```text
    main/config.py
    ```

    Make sure the dataset path, retrieval settings, API credentials, and output paths are correct and aligned with the built index.

2. Run the full dataset through the system:

    ```bash
    python main/load_data.py
    ```

> This step runs the entire dataset through the RT-RAG pipeline: it performs retrieval, reranking, tree generation, and LLM querying.



---

---

## 📊 7. Evaluate the Results

Once inference on the full dataset is complete, you can evaluate the generated answers using:

```bash
python main/evaulate.py /path/to/result.txt
```
> Replace `/path/to/result.txt` with the actual path to the output file generated by `main/load_data.py`.

This script will compute metrics on the dataset.

## 📈 RT-RAG Performance

The table below summarizes RT-RAG's performance across three benchmark datasets using two different backbone models:

| Model           | Dataset     | F1     | EM     |
|----------------|-------------|--------|--------|
| **GPT-4o-mini** | MuSiQue     | 54.42  | 41.50  |
|                | 2WikiMQA    | 75.08  | 63.00  |
|                | HotpotQA    | 65.26  | 52.50  |
|                | **Average** | **64.92** | **52.33** |
| **Qwen2.5-14B** | MuSiQue     | 50.04  | 39.00  |
|                | 2WikiMQA    | 73.69  | 64.00  |
|                | HotpotQA    | 66.24  | 51.00  |
|                | **Average** | **63.32** | **51.33** |

> RT-RAG consistently outperforms all baselines across diverse multi-hop QA datasets.




