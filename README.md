# SciPathBench

A benchmark for evaluating LLM agents on finding shortest citation paths between academic papers.
Inspired by [WikiBench](https://1thousandfaces.substack.com/p/wikibench-76-of-sota-models-fail) | Uses [OpenAlex](OpenAlex), [OpenRouter](https://openrouter.ai/), [Inciteful](https://inciteful.xyz/).

## Quick Start

```bash
# Add OPENROUTER_API_KEY to .env file
# Modify config.py for LLM model and agent settings
uv sync
uv run main.py
```

## How It Works

Data Generation: generate_benchmark_data.py creates problems using Inciteful API to find shortest citation paths between paper found in dataset.py, with results saved to benchmark_pairs.json.
Evaluation: main.py runs the benchmark. LLM agents navigate citation graphs turn-by-turn, scored on success and optimality vs ground truth.
Visualization: visualization.py generates VOSviewer files and .html interactive visualizations.
Results saved to scipathbench_results.json