# llm_agent.py
# The LLM-powered agent that attempts to find the shortest path.

import requests
import json
import logging
import re
from src.openalex_client import OpenAlexClient
from src.utils import reconstruct_abstract
from config import OPENROUTER_API_KEY, OPENROUTER_API_BASE_URL, LLM_PROVIDER_MODEL


class LLMAgent:
    """The LLM-powered agent that attempts to find the shortest path."""

    def __init__(self, api_client: OpenAlexClient, llm_provider: str):
        self.api_client = api_client
        self.llm_provider = llm_provider
        self.visited_nodes = set()
        self.start_frontier = {}
        self.end_frontier = {}
        self.parent_map = {}

    def find_path(self, start_id: str, end_id: str, max_turns: int):
        """Main execution loop for the agent."""
        self.api_client.reset_api_call_count()
        logging.info("--- Starting LLM Agent Run ---")

        start_paper = self.api_client.get_paper_by_id(start_id)
        end_paper = self.api_client.get_paper_by_id(end_id)

        if not start_paper or not end_paper:
            logging.error("Could not retrieve start or end paper.")
            return None, self.api_client.get_api_call_count()

        self.start_frontier[start_id] = self._extract_metadata(start_paper)
        self.end_frontier[end_id] = self._extract_metadata(end_paper)
        self.visited_nodes.update([start_id, end_id])
        self.parent_map = {start_id: None, end_id: None}

        for turn in range(max_turns):
            logging.info(f"--- Agent's Turn {turn + 1}/{max_turns} ---")

            prompt = self._build_prompt(start_paper, end_paper)
            llm_decision = self._get_llm_decision(prompt)

            if not llm_decision or "command" not in llm_decision:
                logging.warning("Agent failed to make a valid decision. Stopping.")
                break

            if llm_decision["command"] == "expand":
                path = self._execute_expansion(llm_decision)
                if path:
                    return path, self.api_client.get_api_call_count()
            else:
                logging.warning(
                    f"Agent returned unknown command: {llm_decision['command']}"
                )

        logging.info("Agent failed to find a path within the turn limit.")
        return None, self.api_client.get_api_call_count()

    def _execute_expansion(self, decision):
        paper_id = decision.get("paper_id")
        direction = decision.get("direction")

        if not paper_id or direction not in ["forward", "backward"]:
            logging.warning(f"Agent provided invalid expansion command: {decision}")
            return None

        logging.info(
            f"Agent decided to expand '{paper_id}' from the '{direction}' frontier."
        )

        current_frontier = (
            self.start_frontier if direction == "forward" else self.end_frontier
        )
        other_frontier = (
            self.end_frontier if direction == "forward" else self.start_frontier
        )

        if paper_id not in current_frontier:
            logging.error(
                f"Paper {paper_id} not in the specified frontier. Agent is confused."
            )
            return None

        del current_frontier[paper_id]

        neighbors = self.api_client.get_neighbors(paper_id)
        for neighbor_id in neighbors:
            if neighbor_id in other_frontier:
                logging.info(f"Path found! Intersection at: {neighbor_id}")
                return self._reconstruct_path(paper_id, neighbor_id, direction)

            if neighbor_id not in self.visited_nodes:
                self.visited_nodes.add(neighbor_id)
                self.parent_map[neighbor_id] = paper_id
                neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                if neighbor_paper:
                    current_frontier[neighbor_id] = self._extract_metadata(
                        neighbor_paper
                    )
        return None

    def _reconstruct_path(self, meet_point1, meet_point2, direction):
        """Reconstructs the path after a bidirectional search meets."""
        path1, curr = [], meet_point1
        while curr is not None:
            path1.append(curr)
            curr = self.parent_map.get(curr)

        path2, curr = [], meet_point2
        while curr is not None:
            path2.append(curr)
            curr = self.parent_map.get(curr)

        return path1[::-1] + path2 if direction == "forward" else path2[::-1] + path1

    def _extract_metadata(self, paper_json: dict) -> dict:
        """Extracts key info from the OpenAlex work object."""
        return {
            "title": paper_json.get("title"),
            "abstract": reconstruct_abstract(paper_json.get("abstract_inverted_index")),
            "publication_year": paper_json.get("publication_year"),
        }

    def _build_prompt(self, start_paper, end_paper):
        """Constructs the detailed prompt for the LLM planner."""
        # Cleaned up prompt to avoid leading whitespace issues from multiline f-strings
        prompt_lines = [
            "You are a research assistant AI finding the shortest citation path between two papers using a bidirectional search.",
            f"START: \"{start_paper['title']}\" ({start_paper['publication_year']})",
            f"END: \"{end_paper['title']}\" ({end_paper['publication_year']})",
            "Analyze the frontiers and decide which SINGLE paper to expand next to connect them. Consider semantic similarity, topic overlap, and dates.",
            "\nFORWARD FRONTIER (from START):",
            json.dumps(self.start_frontier, indent=2),
            "\nBACKWARD FRONTIER (from END):",
            json.dumps(self.end_frontier, indent=2),
            '\nYou MUST respond in a valid JSON format. The JSON object must contain three fields: "command", "paper_id", and "direction".',
            '- "command": Must always be "expand".',
            '- "paper_id": The OpenAlex ID (e.g., "W12345") of the paper you want to expand.',
            '- "direction": Either "forward" or "backward", indicating which frontier the chosen paper_id is from.',
        ]
        return "\n".join(prompt_lines)

    def _get_llm_decision(self, prompt):
        """Makes the API call to OpenRouter to get the agent's next move."""
        if not OPENROUTER_API_KEY:
            logging.error(
                "OPENROUTER_API_KEY not set. Using simple heuristic fallback."
            )
            if self.start_frontier:
                return {
                    "command": "expand",
                    "paper_id": list(self.start_frontier.keys())[0],
                    "direction": "forward",
                }
            return None

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mateo-m/SciPathBench",  # Recommended header
            "X-Title": "SciPathBench",  # Recommended header
        }

        body = {
            "model": self.llm_provider,
            "messages": [{"role": "user", "content": prompt}],
        }

        data_json = json.dumps(body)

        try:
            response = requests.post(
                f"{OPENROUTER_API_BASE_URL}/chat/completions",
                headers=headers,
                data=data_json,
            )
            # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()

            response_text = response.json()["choices"][0]["message"]["content"]

            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                logging.error(
                    f"Could not find a JSON object in the LLM response: {response_text}"
                )
                return None

        except requests.exceptions.HTTPError as e:
            # This is the crucial change: log the server's response body on error.
            logging.error(
                f"OpenRouter API request failed with status {e.response.status_code}: {e}"
            )
            logging.error(
                f"Server Response: {e.response.text}"
            )  # This will show the detailed error from OpenRouter
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"A network-level error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse LLM JSON response: {e} - Response was: {response_text}"
            )
            return None
        return None
