# llm_agent.py
# The LLM-powered agent that attempts to find the shortest path using a forward-only search.

import requests
import json
import logging
import re
from src.openalex_client import OpenAlexClient
from src.paper_graph import PaperGraph
from config import OPENROUTER_API_KEY, OPENROUTER_API_BASE_URL

class LLMAgent:
    """The LLM-powered agent that finds a path using a forward-only search."""
    def __init__(self, api_client: OpenAlexClient, llm_provider: str):
        self.api_client = api_client
        self.llm_provider = llm_provider
        self.graph = PaperGraph()
        self.visited_nodes = set()
        self.frontier = {}  # paper_id -> metadata for LLM

    def find_path(self, start_id: str, end_id: str, max_turns: int, ground_truth_path: list = None):
        """Main execution loop for the agent."""
        logging.info("--- Starting LLM Agent Run ---")

        # Get start and end papers
        start_paper = self.api_client.get_paper_by_id(start_id)
        end_paper = self.api_client.get_paper_by_id(end_id)
        
        if not start_paper or not end_paper:
            logging.error("Could not retrieve start or end paper.")
            return None, None

        # Add start and end nodes to graph
        self.graph.add_node(start_id, start_paper, "start")
        self.graph.add_node(end_id, end_paper, "end")

        # Add ground truth path nodes and edges if provided
        if ground_truth_path:
            logging.info("Adding ground truth path references to graph")
            for i, paper_id in enumerate(ground_truth_path):
                paper_data = self.api_client.get_paper_by_id(paper_id)
                if paper_data:
                    node_type = "start" if i == 0 else "end" if i == len(ground_truth_path) - 1 else "ground_truth"
                    self.graph.add_node(paper_id, paper_data, node_type)
                
                # Add edges between consecutive papers in ground truth path
                if i > 0:
                    self.graph.add_edge(ground_truth_path[i-1], paper_id)
                
                # Add references from each ground truth paper to show the citation network
                if i < len(ground_truth_path) - 1:  # Don't expand the last paper (end node)
                    neighbors = self.api_client.get_neighbors(paper_id)
                    logging.info(f"Found {len(neighbors)} neighbors for ground truth paper {paper_id}")
                    for neighbor_id in neighbors[:10]:  # Limit to first 10 references to avoid clutter
                        self.graph.add_edge(paper_id, neighbor_id)
                        if neighbor_id not in self.graph.nodes:
                            neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                            if neighbor_paper:
                                self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                
        

        # Initialize with start node
        logging.info(f"Starting from: '{start_paper.get('title')}'")
        self.visited_nodes.add(start_id)
        self.graph.agent_path.append(start_id)
        
        # Expand start node automatically
        initial_neighbors = self.api_client.get_neighbors(start_id)
        for neighbor_id in initial_neighbors:
            self.graph.add_edge(start_id, neighbor_id)
            if neighbor_id not in self.visited_nodes:
                self.visited_nodes.add(neighbor_id)
                neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                if neighbor_paper:
                    self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                    self.frontier[neighbor_id] = self.graph.get_node_metadata_for_llm(neighbor_id)

        # Main search loop
        for turn in range(max_turns):
            logging.info(f"--- Agent's Turn {turn + 1}/{max_turns} ---")
            
            if not self.frontier:
                logging.warning("Frontier is empty. Agent cannot continue.")
                break

            # Get LLM decision
            prompt = self._build_prompt(start_paper, end_paper)
            llm_decision = self._get_llm_decision(prompt)

            if not llm_decision or "paper_id" not in llm_decision:
                logging.warning("Agent failed to make a valid decision. Stopping.")
                break

            paper_id_to_expand = llm_decision["paper_id"]
            
            if paper_id_to_expand not in self.frontier:
                logging.error(f"Paper {paper_id_to_expand} not in the frontier. Agent is confused.")
                continue
            
            # Agent expands this paper
            paper_title = self.frontier[paper_id_to_expand]['title']
            logging.info(f"Agent expanding: '{paper_title}'")
            
            # Add to agent path and update node type
            self.graph.agent_path.append(paper_id_to_expand)
            self.graph.nodes[paper_id_to_expand]["node_type"] = "agent_path"
            
            # Remove from frontier
            del self.frontier[paper_id_to_expand]

            # Get neighbors of expanded paper
            neighbors = self.api_client.get_neighbors(paper_id_to_expand)
            logging.info(f"Found {len(neighbors)} neighbors")
            for neighbor_id in neighbors:
                self.graph.add_edge(paper_id_to_expand, neighbor_id)
                
                # Check if we found the target
                if neighbor_id == end_id:
                    logging.info(f"Path found! Target paper reached.")
                    self.graph.agent_path.append(end_id)
                    self.graph.nodes[end_id]["node_type"] = "agent_path"
                    
                    # Save graph and return success
                    self.graph.save_to_file("output/reference_graph.json")
                    return self.graph.agent_path, None
                
                # Add new neighbors to frontier
                if neighbor_id not in self.visited_nodes:
                    self.visited_nodes.add(neighbor_id)
                    neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                    if neighbor_paper:
                        self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                        self.frontier[neighbor_id] = self.graph.get_node_metadata_for_llm(neighbor_id)
        
        # Agent failed to find path
        logging.info("Agent failed to find a path within the turn limit.")
        self.graph.save_to_file("output/reference_graph.json")
        return None, self.graph.agent_path

    def _build_prompt(self, start_paper, end_paper):
        """Constructs the detailed prompt for the LLM planner."""
        # Build current path string
        path_titles = []
        for paper_id in self.graph.agent_path:
            node = self.graph.nodes.get(paper_id, {})
            path_titles.append(node.get("title", f"Paper {paper_id}"))
        path_str = " -> ".join(path_titles)
        
        prompt_lines = [
            "You are a research assistant AI finding the shortest citation path from a START to an END paper.",
            "You can only expand one paper at a time from the frontier.",
            f"START: \"{start_paper['title']}\" ({start_paper['publication_year']})",
            f"END: \"{end_paper['title']}\" ({end_paper['publication_year']})",
            f"\nCURRENT PATH SO FAR: {path_str}",
            "\nAnalyze the papers in the frontier below and decide which SINGLE paper is most promising to expand next to reach the END paper.",
            "\nCURRENT FRONTIER (Papers to choose from):",
            json.dumps(self.frontier, indent=2),
            "\nYou MUST respond in a valid JSON format with ONE key: \"paper_id\".",
            "Example: {\"paper_id\": \"W12345\"}"
        ]
        return "\n".join(prompt_lines)
        
    def _get_llm_decision(self, prompt):
        """Makes the API call to OpenRouter to get the agent's next move."""
        if not OPENROUTER_API_KEY:
            logging.error("OPENROUTER_API_KEY not set. Using simple heuristic fallback.")
            if self.frontier:
                return {"paper_id": list(self.frontier.keys())[0]}
            return None

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {"model": self.llm_provider, "messages": [{"role": "user", "content": prompt}]}
        data_json = json.dumps(body)
        logging.debug(f"LLM Request Body: {data_json}")
        
        try:
            response = requests.post(f"{OPENROUTER_API_BASE_URL}/chat/completions", headers=headers, data=data_json)
            response.raise_for_status()
            response_text = response.json()['choices'][0]['message']['content']
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            logging.debug(f"LLM Response: {response_text}")

            if match:
                return json.loads(match.group(0))
            else:
                logging.error(f"Could not find a JSON object in the LLM response: {response_text}")
                return None
        except requests.exceptions.HTTPError as e:
            logging.error(f"OpenRouter API request failed: {e.response.text}")
            return None
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"An error occurred: {e}")
            return None
