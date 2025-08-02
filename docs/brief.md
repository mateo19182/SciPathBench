Subject: Implementation Objective and Requisites for SciPathBench

This document outlines the core objective and technical requisites for implementing the SciPathBench project. The goal is to create a benchmark to evaluate an LLM agent's ability to find the shortest citation path between academic papers.

1. Project Objective

The primary objective is to build a benchmark named SciPathBench that evaluates an LLM agent's procedural reasoning capabilities on a graph. The agent's task is to find the shortest citation path between two given academic papers, competing against the known optimal path found by a classical algorithm like Breadth-First Search (BFS). The agent's efficiency is measured by the number of API calls it makes.

2. Core Task Definition

    Input: A start_paper and an end_paper, identified by their titles, DOIs, or other metadata.

    Objective: Find a sequence of papers [start_paper, paper_1, paper_2, ..., end_paper] that constitutes a shortest path in the citation graph.

    Path Definition:

        An edge between two papers exists if one cites the other.

        For the purpose of pathfinding, all citation links are treated as undirected.

    Success Criterion: Find any valid shortest path while minimizing the number of "steps" taken.

3. System Requisites & Architecture

3.1. Core Technology Stack

    Knowledge Graph API: The system must use the OpenAlex API as its exclusive source for scholarly data. This is chosen for its free and high-rate-limit access, rich metadata (including abstracts), and built-in support for retrieving both incoming and outgoing citations.

    LLM Provider: The agent's reasoning component must be powered by models accessed via OpenRouter.

3.2. Knowledge Graph (KG) Model

    Nodes: Unique academic works, identified by their OpenAlex ID.

    Edges: Undirected citation links between works.

    Interaction Model: The KG is not pre-compiled. The agent must explore it dynamically by making live calls to the OpenAlex API.

3.3. LLM Agent Architecture (Hybrid Traversal)

The agent must be implemented as a GraphRAG-style system using a hybrid traversal strategy.

    Initialization: Use the OpenAlex API to find the unique IDs for the start_paper and end_paper.

    Bidirectional Setup: Maintain two "frontier" sets of nodes, one expanding from the start_paper (start_frontier) and one from the end_paper (end_frontier).

    LLM as Planner: The core of the agent's logic. At each turn, the LLM's role is to strategically select which paper(s) from the frontiers to expand next.

        The LLM will be prompted with the metadata (titles, abstracts, etc.) of candidate papers from both frontiers.

        Based on semantic analysis, the LLM must decide which candidates are most "promising" for connecting the start and end papers. Its output must be a structured command specifying the next action (e.g., a JSON object).

    State Management: The agent must maintain state, tracking all visited nodes to avoid loops and redundant work.

3.4. Definition of a "Step"

A "step" is the critical unit for measuring efficiency.

    1 Step = 1 API call to the OpenAlex API (e.g., get_references(paper_id), search_papers(query)).

    The agent's primary goal is to find the optimal path using the least number of steps.

4. Evaluation Framework

An automated evaluation harness must be built to run the agent against a benchmark dataset and score its performance.

4.1. Benchmark Dataset

    Content: A curated set of (start_paper, end_paper) tuples.

    Structure: The pairs must be categorized by difficulty, defined by the known shortest path length (e.g., 2, 3, 4, 5 hops).

    Ground Truth: For each pair, the true shortest path length and an example path must be pre-calculated and stored using a classical BFS algorithm. This is the standard against which the agent is measured.

4.2. Performance Metrics (The "Scorecard")

The evaluation harness must calculate the following four metrics for every run:

    Path Success (Binary): 1 if the agent found any valid path; 0 otherwise.

    Path Optimality (Ratio): Measures the quality of the found path.
    Length of Agent’s Found PathLength of True Shortest Path​


    A score of 1.0 is perfectly optimal.

    Step Efficiency (Integer): The total count of API calls made by the agent.

    Reasoning Faithfulness (Binary): 1 if the entire path consists of real papers and verified citation links from the API; 0 if any part is hallucinated.