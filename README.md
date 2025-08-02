# SciPathBench

SciPathBench is a benchmark for evaluating LLM agents' capabilities on citation graphs. The goal is to find the shortest path between two academic papers.

Inspired by [WikiBench: 76% of SOTA Models Fail - by hero thousandfaces](https://1thousandfaces.substack.com/p/wikibench-76-of-sota-models-fail)

Using [OpenAlex](https://openalex.org/), [OpenRouter](https://openrouter.ai/), [VOSviewer](https://www.vosviewer.com/) and [Inciteful](https://inciteful.xyz/).

## Evaluation

The agent's performance is evaluated based on wether the path it finds is correct and how many API requests it took to find (compared to the optimal path).

## Usage

```bash
uv run main.py
```

## TODO

- improve prmpt, only forward frontier

- speed up api calls

- make resilient to error: "message":"Provider returned error","code":429,"

- add visualization with [VOSviewer](https://www.vosviewer.com/)

- use https://inciteful.xyz/c?to=W1995017064&from=W2059020082 for shortest path!
- get lists of papers for start and end, by difficulty? have optimal precalculated

- cli interface?

- actually verify agent path is correct

- get openalex id from doi/name

- traverse by authors?

- manage W4285719527 (Deleted Work)

- leaderboard
