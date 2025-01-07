import os

IP_WHITELIST = os.environ.get("IP_WHITELIST") or ["127.0.0.1"]
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH") or "./vector_stores/"

#NVIDIA
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

PORT = os.getenv("PORT") or 8000
RETRIEVE_K = os.getenv("RETRIEVE_K") or 15
RERANK_TOP_N = os.getenv("RERANK_TOP_N") or 3
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL") or "mistralai/mixtral-8x22b-instruct-v0.1"
