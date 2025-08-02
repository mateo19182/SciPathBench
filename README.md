# SciPathBench

SciPathBench is a benchmark for evaluating LLM agents' capabilities on citation graphs. The goal is to find the shortest path between two academic papers.

Inspired by [WikiBench: 76% of SOTA Models Fail - by hero thousandfaces](https://1thousandfaces.substack.com/p/wikibench-76-of-sota-models-fail)

Using [OpenAlex](https://openalex.org/), [OpenRouter](https://openrouter.ai/), [VOSviewer](https://www.vosviewer.com/) and [Inciteful](https://inciteful.xyz/).

## Usage

Create a `.env` file in the root directory with your OPENROUTER_API_KEY.
Modify config.py as needed.

```bash
uv sync
uv run main.py
```

## Evaluation

The agent's performance is evaluated based on wether the path it finds is correct and how many API requests it took to find (compared to the optimal path).

## Visualization

To visualize the citation graph, use the generated graph.json file with [VOSviewer](https://www.vosviewer.com/).

## 

The generate_benchmark_data.py script creates a set of pre-calculated problems using papers from dataset.py. Using the inciteful.xyz API the shortest citation paths between pairs of papers is calculated. Start and end papers are not reused once a path is found for them. The resulting problems, sorted by difficulty, are saved to benchmark_pairs.json.

The main.py script orchestrates. It selects problems from a file and extracts ground truth (min number of hops) if not available. a task contains a start_paper_id, end_paper_id and ground_truth_path.

Each turn, it queries an LLM via OpenRouter to decide which paper to explore next. 

The agent's performance is scored on its ability to find the path (Success) and the path's quality (Optimality). Final results are saved to scipathbench_results.json. The visualization.py script generates files for VOSviewer.It uses networkx to create a combined graph of both the ground truth and the agent's path.The nx2vos library converts this graph into VOSviewer's map and network file formats.The final files can be opened in VOSviewer to visually compare the two paths.

## TODO

- improve prompt, let them go abck to previous papers?

- get human baseline, ui?

- speed up api calls
- dont redo api calls, use cache!

- make resilient to error: "message":"Provider returned error","code":429,"

- better eval, steps are counted badly...

- improve visualization, weights, attributes...

- get better dataset

- cli interface?

- take a list of problems

- actually verify agent path is correct

- get openalex id from doi/name directly

- traverse by authors?

- manage W4285719527 (Deleted Work)

- leaderboard
