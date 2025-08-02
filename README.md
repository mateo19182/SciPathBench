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

## TODO

- improve prompt, only forward frontier!

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
