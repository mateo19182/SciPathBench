# main.py
# Main execution script for the SciPathBench benchmark.

import logging
import json
import time

# Import configurations and utility functions
import config
from src.utils import setup_logging

# Import core logic classes
from src.openalex_client import OpenAlexClient
from src.graph_search import GraphSearch
from src.llm_agent import LLMAgent
from src.eval import EvaluationHarness


def main():
    """Main function to run the entire benchmark process."""
    setup_logging(config.LOG_FILE)
    logging.info("==================================================")
    logging.info("Starting New SciPathBench Run")
    logging.info("==================================================")

    # 1. Setup
    # TODO: Load a curated list of (start, end) pairs from a file.
    # Using a known 2-hop path for demonstration.
    start_paper_id = "W2059020082" 
    end_paper_id = "W1995017064" 

    logging.info(
        f"Objective: Find shortest path between {start_paper_id} and {end_paper_id}"
    )

    # 2. Get Ground Truth
    bfs_client = OpenAlexClient()
    bfs_search = GraphSearch(api_client=bfs_client)
    ground_truth, bfs_steps = bfs_search.find_shortest_path_bfs(
        start_paper_id, end_paper_id
    )

    if ground_truth:
        logging.info(
            f"Ground Truth Path (BFS): {ground_truth} (Length: {len(ground_truth)-1}, Cost: {bfs_steps} API calls)"
        )
    else:
        logging.error(
            "Could not find a ground truth path. Papers may not be connected or BFS depth limit reached."
        )
        return  # Exit if no path exists

    # 3. Run LLM Agent
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

    # 4. Evaluate Performance
    evaluator = EvaluationHarness(
        ground_truth_path=ground_truth,
        agent_path=agent_found_path,
        agent_steps=agent_api_calls,
    )
    final_scorecard = evaluator.run_evaluation()

    # 5. Save Results
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_pair": {
            "start_paper_id": start_paper_id,
            "end_paper_id": end_paper_id,
        },
        "ground_truth": {
            "path": ground_truth,
            "path_length": len(ground_truth) - 1 if ground_truth else 0,
            "api_calls": bfs_steps,
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
    logging.info("SciPathBench Run Finished.")


if __name__ == "__main__":
    main()
