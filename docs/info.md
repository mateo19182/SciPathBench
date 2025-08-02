The generate_benchmark_data.py script creates a set of pre-calculated problems using papers from dataset.py. Using the inciteful.xyz API the shortest citation paths between pairs of papers is calculated. Start and end papers are not reused once a path is found for them. The resulting problems, sorted by difficulty, are saved to benchmark_pairs.json.

The main.py script orchestrates. It selects problems from a file and extracts ground truth (min number of hops) if not available. a task contains a start_paper_id, end_paper_id and ground_truth_path.

Each turn, it queries an LLM via OpenRouter to decide which paper to explore next. 

The agent's performance is scored on its ability to find the path (Success), the path's quality (Optimality), and the number of API calls used (Efficiency).Logging: Results are saved to scipathbench_results.json.4. Phase 3: VisualizationThe visualization.py script generates files for VOSviewer.It uses networkx to create a combined graph of both the ground truth and the agent's path.The nx2vos library converts this graph into VOSviewer's map and network file formats.The final files can be opened in VOSviewer to visually compare the two paths.