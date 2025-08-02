# config.py
# Central configuration file for the SciPathBench project.

import os
import logging

# --- API Configurations ---
OPENALEX_API_BASE_URL = "https://api.openalex.org"
OPENALEX_USER_EMAIL = "mateoamadoares@gmail.com"  # OpenAlex kindly requests an email for high-volume users

# --- OpenRouter Configuration ---
from dotenv import load_dotenv

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

# --- Benchmark Execution Configuration ---
# 'precalculated': Use a pair from BENCHMARK_DATA_FILE.
# 'runtime': Generate a random pair from the dataset.py list and find the path at runtime.
BENCHMARK_MODE = "precalculated"
BENCHMARK_DATA_FILE = "output/benchmark_pairs.json"
NUMBER_OF_BENCHMARK_TASKS = 1  # Number of tasks to run in benchmark mode

logging.info(f"Configuration loaded: {LLM_PROVIDER_MODEL}, {AGENT_MAX_TURNS}")
