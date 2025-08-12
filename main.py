# main.py
# Main execution script for the SciPathBench benchmark.

import logging
import json
import time
import random
import statistics
import argparse

# Import configurations and utility functions
from src import config
from src.utils import setup_logging
from src.data.dataset import LANDMARK_PAPERS
from src.data.generate_data import get_path_from_inciteful

# Import core logic classes
from src.visualization.visualization import create_vosviewer_files
from src.services.openalex_client import OpenAlexClient
from src.agents.llm_agent import LLMAgent
from src.agents.human_agent import HumanAgent
from src.core.eval import EvaluationHarness

def get_benchmark_tasks():
    """
    Selects benchmark tasks based on the BENCHMARK_MODE in the config.
    
    Returns:
        list: A list of task dictionaries, where each dictionary contains
              'start_id', 'end_id', and 'ground_truth_path'.
              Returns an empty list if no tasks can be generated.
    """
    if config.BENCHMARK_MODE == "runtime":
        logging.info("Using runtime benchmark generation mode.")
        task = get_runtime_task()
        return [task] if task else []

    if config.BENCHMARK_MODE == "precalculated":
        logging.info(
            f"Using pre-calculated benchmark mode with {config.NUMBER_OF_BENCHMARK_TASKS} "
            f"task(s) from '{config.BENCHMARK_DATA_FILE}'"
        )
        try:
            with open(config.BENCHMARK_DATA_FILE, "r") as f:
                all_pairs = json.load(f)

            if not all_pairs:
                logging.error(f"'{config.BENCHMARK_DATA_FILE}' is empty. Cannot select tasks.")
                return []
            
            # Ensure we don't request more tasks than available
            num_to_sample = min(config.NUMBER_OF_BENCHMARK_TASKS, len(all_pairs))
            if num_to_sample < config.NUMBER_OF_BENCHMARK_TASKS:
                logging.warning(f"Requested {config.NUMBER_OF_BENCHMARK_TASKS} tasks, but only {len(all_pairs)} are available. Using {num_to_sample}.")

            selected_tasks = random.sample(all_pairs, k=num_to_sample)
            logging.info(f"Selected {len(selected_tasks)} tasks.")

            return [
                {
                    "start_id": task["start_id"],
                    "end_id": task["end_id"],
                    "ground_truth_path": task["path_ids"],
                }
                for task in selected_tasks
            ]
        except (FileNotFoundError, IndexError, ValueError) as e:
            logging.error(f"Could not load or sample from {config.BENCHMARK_DATA_FILE}: {e}")
            return []
    
    raise ValueError(f"Unknown BENCHMARK_MODE: {config.BENCHMARK_MODE}")

def get_runtime_task():
    """Generates a single task by finding a path at runtime."""
    for _ in range(config.MAX_RUNTIME_RETRIES): # Retry loop to avoid getting stuck
        start_id, end_id = random.sample(LANDMARK_PAPERS, 2)
        logging.info(f"Attempting to generate runtime task: {start_id} -> {end_id}")

        bfs_client = OpenAlexClient()

        # Resolve DOIs or URLs to OpenAlex IDs for BFS
        start_work = bfs_client.get_paper_by_id(start_id)
        end_work = bfs_client.get_paper_by_id(end_id)
        if not start_work or not end_work:
            logging.warning("Failed to resolve start or end paper. Retrying...")
            continue
        start_id_norm = (start_work.get("id") or "").split("/")[-1]
        end_id_norm = (end_work.get("id") or "").split("/")[-1]
        if not start_id_norm or not end_id_norm:
            logging.warning("Could not normalize OpenAlex IDs from works. Retrying...")
            continue

        # Use Inciteful connector API to fetch the shortest path
        path_ids, _ = get_path_from_inciteful(start_id_norm, end_id_norm)
        ground_truth = path_ids

        if ground_truth:
            logging.info("Successfully found a path for the runtime task.")
            return {"start_id": start_id_norm, "end_id": end_id_norm, "ground_truth_path": ground_truth}
        else:
            logging.warning("Failed to find a path for the random pair. Retrying...")
    
    logging.error("Failed to generate a valid runtime task after multiple retries.")
    return None

def run_single_task(task, task_index=1, interactive_mode=False):
    """
    Executes the agent, evaluation, and visualization for a single benchmark task.

    Args:
        task (dict): A dictionary containing 'start_id', 'end_id', and 'ground_truth_path'.
        task_index (int): The index of the current task for logging and file naming.
        interactive_mode (bool): Whether to use human interactive mode instead of LLM agent.

    Returns:
        dict: A dictionary containing the comprehensive results for this task.
    """
    start_paper_id = task["start_id"]
    end_paper_id = task["end_id"]
    ground_truth = task["ground_truth_path"]
    
    # 2. Run Agent (LLM or Human)
    agent_client = OpenAlexClient()
    
    if interactive_mode:
        agent = HumanAgent(api_client=agent_client)
    else:
        agent = LLMAgent(api_client=agent_client, llm_provider=config.LLM_PROVIDER_MODEL)
    
    # Get paper titles for logging
    start_paper = agent.api_client.get_paper_by_id(start_paper_id)
    end_paper = agent.api_client.get_paper_by_id(end_paper_id)
    start_paper_title = start_paper.get("title", "Unknown") if start_paper else "Unknown"
    end_paper_title = end_paper.get("title", "Unknown") if end_paper else "Unknown"
    
    if not interactive_mode:
        logging.info(f"Objective: Find shortest path between \"{start_paper_title}\" and \"{end_paper_title}\"")
        
        # Get ground truth titles for logging
        ground_truth_titles = []
        for paper_id in ground_truth:
            paper = agent.api_client.get_paper_by_id(paper_id)
            title = paper.get("title", "Unknown") if paper else "Unknown"
            ground_truth_titles.append(title)
            
        logging.info(f"Ground Truth Path: {ground_truth_titles} (Length: {len(ground_truth)-1})")

    
    agent_found_path, path = agent.find_path(
        start_paper_id, end_paper_id, max_turns=config.AGENT_MAX_TURNS, ground_truth_path=ground_truth
    )

    if agent_found_path:
        logging.info(f"Agent Path: {agent_found_path} (Length: {len(agent_found_path)-1})")

    # 3. Evaluate Performance
    evaluator = EvaluationHarness(
        ground_truth_path=ground_truth,
        agent_path=agent_found_path,
    )
    final_scorecard = evaluator.run_evaluation()
    logging.info(f"Scorecard for Task {task_index}:\n{json.dumps(final_scorecard, indent=4)}")

    # 4. Generate VOSviewer Visualization Files
    create_vosviewer_files(
        ground_truth_path=ground_truth,
        agent_path=agent_found_path if agent_found_path is not None else path,
        output_prefix=f"visualization_task_{task_index}",
    )

    # logging.info(f"VOSviewer files created for task {task_index}.")

    # 5. Collate results for this single task
    results_data = {
        "task_index": task_index,
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
        },
        "scorecard": final_scorecard,
    }
    return results_data

def log_summary(all_results):
    """Calculates and logs a summary of metrics from all task results."""
    if not all_results:
        logging.info("No results to summarize.")
        return

    num_tasks = len(all_results)
    
    # A run is successful if the agent found a path.
    # We create a list of successful runs to easily calculate stats on them.
    successful_runs = [r for r in all_results if r["agent_run"]["path"]]
    success_count = len(successful_runs)
    success_rate = (success_count / num_tasks) * 100 if num_tasks > 0 else 0

    # These metrics should only be calculated on successful runs
    precisions = [r["scorecard"]["precision"] for r in successful_runs]
    recalls = [r["scorecard"]["recall"] for r in successful_runs]
    
    summary = {
        "Total Tasks": num_tasks,
        "Success Rate": f"{success_rate:.2f}% ({success_count}/{num_tasks})",
        "Average Precision (Successful Runs)": f"{statistics.mean(precisions):.2f}" if precisions else "N/A",
        "Average Recall (Successful Runs)": f"{statistics.mean(recalls):.2f}" if recalls else "N/A",
    }
    
    logging.info("==================================================")
    logging.info("Benchmark Run Summary")
    logging.info("==================================================")
    logging.info(json.dumps(summary, indent=4))

def run_interactive_mode():
    """Run the interactive mode for human players."""
    print("🎮 Welcome to SciPathBench Interactive Mode!")
    print("You'll be challenged to find the shortest path between two academic papers.")
    
    # Get a random task
    tasks = get_benchmark_tasks()
    if not tasks:
        print("❌ Failed to load any benchmark tasks.")
        return
        
    task = tasks[0]  # Use the first task
    
    try:
        result = run_single_task(task, task_index=1, interactive_mode=True)
        
        if result:
            print("\n📊 Your performance has been recorded!")
        else:
            print("\n🎯 Thanks for playing!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

def main():
    """Main function to run the entire benchmark process for one or more tasks."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SciPathBench - Academic Paper Pathfinding Benchmark")
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode for human players"
    )
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_mode()
        return
        
    # Standard benchmark mode
    setup_logging(config.LOG_FILE)
    logging.info("==================================================")
    logging.info(f"Starting New SciPathBench Run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("==================================================")

    # 1. Get a list of all benchmark tasks
    tasks = get_benchmark_tasks()

    if not tasks:
        logging.error("Failed to obtain any valid benchmark tasks. Exiting.")
        return

    all_results = []
    logging.info(f"Beginning benchmark run with {len(tasks)} task(s).")

    # Process each task in a loop
    for i, task in enumerate(tasks):
        task_num = i + 1
        logging.info(f"----------------- Running Task {task_num}/{len(tasks)} -----------------")
        try:
            result = run_single_task(task, task_index=task_num, interactive_mode=False)
            if result:
                all_results.append(result)
        except Exception as e:
            logging.error(f"An unexpected error occurred during task {task_num}: {e}", exc_info=True)
        logging.info(f"----------------- Finished Task {task_num}/{len(tasks)} -----------------\n")

    # Save all collected results to a single file
    if all_results:
        with open(config.RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=4)
        logging.info(f"--- All results saved to {config.RESULTS_FILE} ---")
    else:
        logging.warning("No results were generated to save.")

    # Log a final summary
    log_summary(all_results)
    
    logging.info("==================================================")
    logging.info("SciPathBench Run Finished.")
    logging.info("==================================================")

if __name__ == "__main__":
    main()
