# Tree generation parameters for hierarchical QA
TREES_PER_QUESTION = 3           # Number of trees to generate per question (for consensus-based QA)
MAX_TOKENS = 2000                # Maximum number of tokens allowed per tree
DECOMPOSE_TEMPERATURE = 0.8
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
NUM_EXAMPLES = 25                # Number of few-shot examples
MAX_HEIGHT = 3                   # Maximum depth of the generated tree (was 4)
ENHANCED_RIGHT_SUBTREE = True
RIGHT_SUBTREE_VARIANTS = 1
RIGHT_SUBTREE_TREES_PER_VARIANT = 3
MAX_VARIANTS = 2

# Final output control (for fair comparisons vs vanilla baselines)
# Caps only the *final predicted answer string* written to outputs,
# not the internal RT-RAG decomposition/reasoning calls.
MAX_NEW_TOKENS_FINAL_ANSWER = 32

# Path to save run-time statistics and logs
STATS_FILE_PATH = "outputs/tree_stats.txt"

# LLM model name (used for all generate_response calls)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# OpenAI-compatible language model API settings
BASE_URL = "http://localhost:8000/v1"
API_KEY = "token-placeholder"

# Path to save generated dense embeddings
EMBEDDING_DATA = "embedding_data"

# External reranker service settings (for dense embedding API)
RANKER_URL = "http://localhost:8001/v1"
RANKER_KEY = "token-placeholder"

# Retrieval configuration
RETRIEVE_TEMPERATURE = 0.3
DATASET = "wiki_psgs"           # Retrieval corpus name (wiki_psgs = DPR Wikipedia passages)
METHOD = "bm25"                 # Retrieval method: "dense" or "bm25"
CHUNK_SIZE = 200                # Max number of words per chunk
MIN_SENTENCE = 2                # Minimum number of sentences per chunk
OVERLAP = 2                     # Number of overlapping sentences between chunks
TOPK1 = 45                      # Top-K candidates from initial retrieval
TOPK2 = 10                      # Top-K reranked candidates (match vanilla top-k=10)
SAMPLING_ITERATIONS = 3         # Number of sampling iterations for consensus
MAX_ITERATIONS = 3              # Maximum number of iterations for query rewriting

# Root output directory for saving predictions/results
OUTPUT_DIR_ROOT = "outputs"

# Concurrency control
MAX_CONCURRENT = 16             # Maximum number of concurrent QA jobs (bump if vLLM handles more)

# Path to evaluation dataset (in .jsonl format)
DATA_PATH = "main/data/hotpotqa_dev_1k.jsonl"
