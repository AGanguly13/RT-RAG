# config.py
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

raw_path = os.path.join(_project_root, "main", "raw")
save_path = os.path.join(_project_root, "embedding_data", "wiki_psgs", "200_2_2")

# OpenAI-compatible embedding API (e.g. vLLM with text-embedding-3-small or local endpoint)
base_url = "http://localhost:8001/v1"
api_key = "token-placeholder"

dataset_name = "wiki_psgs"

chunk_size = 200
min_sentence = 2
overlap = 2
