"""Central configuration file for the SciPathBench project."""

import logging
import os
from dotenv import load_dotenv

# --- API Configurations ---
OPENALEX_API_BASE_URL = "https://api.openalex.org"
OPENALEX_USER_EMAIL = "mateoamadoares@gmail.com"  # OpenAlex kindly requests an email for high-volume users
OPENCITATIONS_API_KEY = os.getenv("OPENCITATIONS_API_KEY")  # Default key for OpenCitations

# --- HTTP Caching (persistent) ---
# Persistent cache for HTTP GET responses (OpenAlex, OpenCitations, etc.)
# The SQLite file will be stored at: output/openalex_http_cache.sqlite
OPENALEX_CACHE_BACKEND = "sqlite"
OPENALEX_CACHE_NAME = "output/openalex_http_cache"  # no extension; .sqlite will be appended by requests-cache
OPENALEX_CACHE_EXPIRE_SECONDS = None  # never expire; keep responses indefinitely
OPENALEX_MAX_RETRIES = 5
OPENALEX_RETRY_BACKOFF_SECONDS = 2.0
OPENALEX_MAX_WORKERS = 8

# --- OpenRouter Configuration ---
load_dotenv()
# IMPORTANT: Set your OpenRouter API key from .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"

# --- LLM Agent Configuration ---
# Recommended models: google/gemini-flash-1.5, cohere/command-r, mistralai/mistral-7b-instruct-v0.2
LLM_PROVIDER_MODEL = "mistralai/ministral-8b"
AGENT_MAX_TURNS = 10 # Max number of decisions the agent can make

# --- BFS Ground Truth Configuration ---
BFS_MAX_DEPTH = 10  # Search depth limit to prevent excessive runtimes (max path length of 2*BFS_MAX_DEPTH)

# --- Logging and Results ---
LOG_FILE = "output/scipathbench_run.log"
RESULTS_FILE = "output/scipathbench_results.json"
LANDMARK_DATA_FILE = "output/landmark_papers.json"
LANDMARK_ID_PREFERENCE = "openalex"  # "openalex" or "doi"

# --- Benchmark Execution Configuration ---
# 'precalculated': Use a pair from BENCHMARK_DATA_FILE.
# 'runtime': Generate a random pair from the dataset.py list and find the path at runtime.
BENCHMARK_MODE = "precalculated"
BENCHMARK_DATA_FILE = "output/benchmark_pairs.json"
NUMBER_OF_BENCHMARK_TASKS = 1  # Number of tasks to run in benchmark mode

logging.info(f"Configuration loaded: {LLM_PROVIDER_MODEL}, {AGENT_MAX_TURNS}")
