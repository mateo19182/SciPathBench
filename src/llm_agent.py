# llm_agent.py
# The LLM-powered agent that attempts to find the shortest path using a forward-only search.

import requests
import json
import logging
import re
from src.openalex_client import OpenAlexClient
from config import OPENROUTER_API_KEY, OPENROUTER_API_BASE_URL

class LLMAgent:
    """The LLM-powered agent that finds a path using a forward-only search."""
    def __init__(self, api_client: OpenAlexClient, llm_provider: str):
        self.api_client = api_client
        self.llm_provider = llm_provider
        self.visited_nodes = set()
        self.frontier = {}
        self.parent_map = {}
        self.current_path_ids = []
        self.current_path_titles = []

    def find_path(self, start_id: str, end_id: str, max_turns: int):
        """Main execution loop for the agent."""
        logging.info("--- Starting LLM Agent Run ---")

        start_paper = self.api_client.get_paper_by_id(start_id)
        end_paper = self.api_client.get_paper_by_id(end_id)
        
        if not start_paper or not end_paper:
            logging.error("Could not retrieve start or end paper.")
            return None, None

        # --- Initial Step Optimization ---
        # Automatically expand the start node as the first step is always redundant.
        logging.info(f"Automatically expanding start node '{start_paper.get('title')}'")
        self.visited_nodes.add(start_id)
        self.parent_map = {start_id: None}
        self.current_path_ids = [start_id]
        self.current_path_titles = [start_paper.get('title')]
        
        initial_neighbors = self.api_client.get_neighbors(start_id)
        for neighbor_id in initial_neighbors:
            neighbor_id = neighbor_id.split('/')[-1]
            if neighbor_id not in self.visited_nodes:
                self.visited_nodes.add(neighbor_id)
                self.parent_map[neighbor_id] = start_id
                neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                if neighbor_paper:
                    self.frontier[neighbor_id] = self._extract_metadata(neighbor_paper)
        # --- End of Initial Step ---

        for turn in range(max_turns):
            logging.info(f"--- Agent's Turn {turn + 1}/{max_turns} ---")
            
            if not self.frontier:
                logging.warning("Frontier is empty. Agent cannot continue.")
                break

            prompt = self._build_prompt(start_paper, end_paper)
            llm_decision = self._get_llm_decision(prompt)

            if not llm_decision or "paper_id" not in llm_decision:
                logging.warning("Agent failed to make a valid decision. Stopping.")
                break

            paper_id_to_expand = llm_decision["paper_id"]
            
            if paper_id_to_expand not in self.frontier:
                logging.error(f"Paper {paper_id_to_expand} not in the frontier. Agent is confused.")
                continue
            
            # Update current path
            self.current_path_ids.append(paper_id_to_expand)
            self.current_path_titles.append(self.frontier[paper_id_to_expand]['title'])
            logging.info(f"Agent decided to expand '{self.frontier[paper_id_to_expand]['title']}'.")

            del self.frontier[paper_id_to_expand]

            neighbors = self.api_client.get_neighbors(paper_id_to_expand)
            for neighbor_id in neighbors:
                #TODO if neighbor_id == #Deleted Work
                neighbor_id = neighbor_id.split('/')[-1]
                if neighbor_id == end_id:
                    logging.info(f"Path found! Target paper {end_id} reached.")
                    # Reconstruct the final path using the parent map
                    final_path = self._reconstruct_final_path(paper_id_to_expand, end_id)
                    return final_path, None
                
                if neighbor_id not in self.visited_nodes:
                    self.visited_nodes.add(neighbor_id)
                    self.parent_map[neighbor_id] = paper_id_to_expand
                    neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                    if neighbor_paper:
                        self.frontier[neighbor_id] = self._extract_metadata(neighbor_paper)
        
        logging.info("Agent failed to find a path within the turn limit.")
        return None, self._reconstruct_failed_path(self.current_path_ids[-1])

    def _reconstruct_final_path(self, last_node, end_node):
        """Reconstructs the full path from the parent map after finding the end node."""
        self.parent_map[end_node] = last_node
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = self.parent_map.get(curr)
        return path[::-1] # Reverse to get start -> end order
    
    def _reconstruct_failed_path(self, last_node):
        """Reconstructs the path when the agent fails to find the end node."""
        path = []
        curr = last_node
        while curr is not None:
            path.append(curr)
            curr = self.parent_map.get(curr)
        return path[::-1]
    
    def _extract_metadata(self, paper_json: dict) -> dict:
        """Extracts key info for the agent's context."""
        return {
            "title": paper_json.get("title"),
            "publication_year": paper_json.get("publication_year"),
            "concepts": [c.get("display_name") for c in paper_json.get("concepts", [])[:3]] # Top 3 concepts
        }
        
    def _build_prompt(self, start_paper, end_paper):
        """Constructs the detailed prompt for the LLM planner."""
        path_str = " -> ".join(self.current_path_titles)
        prompt_lines = [
            "You are a research assistant AI finding the shortest citation path from a START to an END paper.",
            "You can only expand one paper at a time from the frontier.",
            f"START: \"{start_paper['title']}\" ({start_paper['publication_year']})",
            f"END: \"{end_paper['title']}\" ({end_paper['publication_year']})",
            f"\nCURRENT PATH SO FAR: {path_str}",
            "\nAnalyze the papers in the frontier below and decide which SINGLE paper is most promising to expand next to reach the END paper.",
            # "Consider semantic similarity of titles and concepts.",
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
