# main.py
# Main execution script for the SciPathBench benchmark.

import logging
import json
import time
import random

# Import configurations and utility functions
import config
from src.utils import setup_logging
from src.dataset import LANDMARK_PAPERS

# Import core logic classes
from src.visualization import create_vosviewer_files
from src.openalex_client import OpenAlexClient
from src.graph_search import GraphSearch
from src.llm_agent import LLMAgent
from src.eval import EvaluationHarness


def get_benchmark_task():
    """
    Selects a benchmark task based on the BENCHMARK_MODE in config.
    Returns a dictionary with 'start_id', 'end_id', and 'ground_truth_path'.
    """
    if config.BENCHMARK_MODE == "precalculated":
        logging.info(
            f"Using pre-calculated benchmark mode from '{config.BENCHMARK_DATA_FILE}'"
        )
        try:
            with open(config.BENCHMARK_DATA_FILE, "r") as f:
                all_pairs = json.load(f)

            if not all_pairs:
                logging.error(
                    f"{config.BENCHMARK_DATA_FILE} is empty. Falling back to runtime mode."
                )
                return get_runtime_task()

            selected_task = random.choice(all_pairs)
            logging.info(
                f"Selected pre-calculated task with difficulty {selected_task['difficulty']}."
            )

            return {
                "start_id": selected_task["start_id"],
                "end_id": selected_task["end_id"],
                "ground_truth_path": selected_task[
                    "path_ids"
                ],  # Use path_ids for the ground truth
            }
        except (FileNotFoundError, IndexError):
            logging.error(
                f"Could not load from {config.BENCHMARK_DATA_FILE}. Falling back to runtime mode."
            )
            return get_runtime_task()

    elif config.BENCHMARK_MODE == "runtime":
        logging.info("Using runtime benchmark generation mode.")
        return get_runtime_task()
    else:
        raise ValueError(f"Unknown BENCHMARK_MODE: {config.BENCHMARK_MODE}")


def get_runtime_task():
    """Generates a task by finding a path at runtime. Returns a dictionary."""
    start_id, end_id = random.sample(LANDMARK_PAPERS, 2)
    logging.info(f"Randomly selected runtime pair: {start_id} -> {end_id}")

    bfs_client = OpenAlexClient()
    bfs_search = GraphSearch(api_client=bfs_client)
    ground_truth, _ = bfs_search.find_shortest_path_bfs(start_id, end_id)

    if not ground_truth:
        logging.warning(
            "Failed to find a path for the random pair at runtime. Trying again."
        )
        return get_runtime_task()

    return {"start_id": start_id, "end_id": end_id, "ground_truth_path": ground_truth}


def main():
    """Main function to run the entire benchmark process."""
    setup_logging(config.LOG_FILE)
    logging.info("==================================================")
    logging.info("Starting New SciPathBench Run")
    logging.info("==================================================")

    # 1. Get Benchmark Task
    task = get_benchmark_task()

    if not task or not task.get("ground_truth_path"):
        logging.error("Failed to obtain a valid benchmark task. Exiting.")
        return

    start_paper_id = task["start_id"]
    end_paper_id = task["end_id"]
    ground_truth = task["ground_truth_path"]

    logging.info(
        f"Objective: Find shortest path between {start_paper_id} and {end_paper_id}"
    )
    logging.info(f"Ground Truth Path: {ground_truth} (Length: {len(ground_truth)-1})")

    # 2. Run LLM Agent
    agent_client = OpenAlexClient()
    agent = LLMAgent(api_client=agent_client, llm_provider=config.LLM_PROVIDER_MODEL)
    agent_found_path, agent_api_calls = agent.find_path(
        start_paper_id, end_paper_id, max_turns=config.AGENT_MAX_TURNS
    )

    if agent_found_path:
        logging.info(
            f"Agent Path: {agent_found_path} (Length: {len(agent_found_path)-1})"
        )
    else:
        logging.warning("Agent did not find a path.")
    logging.info(f"Agent used {agent_api_calls} API calls (steps).")

    # 3. Evaluate Performance
    evaluator = EvaluationHarness(
        ground_truth_path=ground_truth,
        agent_path=agent_found_path,
        agent_steps=agent_api_calls,
    )
    final_scorecard = evaluator.run_evaluation()

    # 4. Save Results
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_pair": {
            "start_paper_id": start_paper_id,
            "end_paper_id": end_paper_id,
        },
        "ground_truth": {
            "path": ground_truth,
            "path_length": len(ground_truth) - 1 if ground_truth else 0,
        },
        "agent_run": {
            "model": config.LLM_PROVIDER_MODEL,
            "path": agent_found_path,
            "path_length": len(agent_found_path) - 1 if agent_found_path else 0,
            "api_calls": agent_api_calls,
        },
        "scorecard": final_scorecard,
    }

    with open(config.RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=4)

    logging.info(f"--- Final Scorecard written to {config.RESULTS_FILE} ---")
    logging.info(json.dumps(final_scorecard, indent=4))

    # 5. Generate VOSviewer Visualization Files
    create_vosviewer_files(
        ground_truth_path=ground_truth,
        agent_path=agent_found_path,
        output_prefix="visualization",
    )

    logging.info("Visualization files created for VOSviewer.")
    logging.info("SciPathBench Run Finished.")


if __name__ == "__main__":
    main()
