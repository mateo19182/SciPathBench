# main.py
# Orchestrates the entire benchmark process.

import requests
import json
from collections import deque
import os

# --- Configuration ---
OPENALEX_API_BASE_URL = "https://api.openalex.org"
# IMPORTANT: Set your OpenRouter API key as an environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"


def reconstruct_abstract(inverted_index: dict) -> str:
    """
    Reconstructs the abstract text from OpenAlex's inverted index format.
    """
    if not inverted_index:
        return "Abstract not available."

    max_len = max(max(positions) for positions in inverted_index.values())
    abstract_list = [""] * (max_len + 1)

    for word, positions in inverted_index.items():
        for pos in positions:
            abstract_list[pos] = word

    return " ".join(filter(None, abstract_list))


class OpenAlexClient:
    """
    Handles all interactions with the OpenAlex API.
    This class is responsible for fetching paper data and tracking API calls.
    """

    def __init__(self, user_email="scipathbench@example.com"):
        self.api_call_count = 0
        # OpenAlex politely asks for an email for better service
        self.headers = {"User-Agent": f"SciPathBench/1.0 (mailto:{user_email})"}

    def _make_request(self, endpoint, params=None):
        """Internal method to handle API requests and count them."""
        self.api_call_count += 1
        try:
            # Adding a polite 'mailto' parameter
            if params:
                params["mailto"] = self.headers["User-Agent"].split("mailto:")[1][:-1]

            response = requests.get(
                f"{OPENALEX_API_BASE_URL}{endpoint}",
                params=params,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"API HTTP Error: {e} - URL: {e.response.url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            return None

    def get_paper_by_id(self, openalex_id: str):
        """
        Retrieves a single paper's full metadata by its OpenAlex ID.
        A "step" is counted here.
        """
        print(
            f"API CALL {self.api_call_count + 1}: Getting details for paper ID {openalex_id}"
        )
        return self._make_request(f"/works/{openalex_id}")

    def get_neighbors(self, openalex_id: str):
        """
        Gets all papers that a given paper cites (outgoing) and that cite it (incoming).
        This treats the graph as undirected. Returns a list of neighbor IDs.
        This costs 2 API calls.
        """
        print(
            f"API CALLS {self.api_call_count + 1}-{self.api_call_count + 2}: Getting all neighbors for {openalex_id}"
        )
        work = self.get_paper_by_id(openalex_id)
        if not work:
            return []

        # Get outgoing citations
        citations = work.get("referenced_works", [])

        # Get incoming citations
        references_data = self._make_request(
            "/works", params={"filter": f"cites:{openalex_id}", "select": "id"}
        )
        references = (
            [item["id"].split("/")[-1] for item in references_data.get("results", [])]
            if references_data
            else []
        )

        # Combine and remove duplicates
        return list(set(citations + references))

    def reset_api_call_count(self):
        """Resets the counter for a new run."""
        self.api_call_count = 0

    def get_api_call_count(self):
        """Returns the total number of API calls made."""
        return self.api_call_count


class GraphSearch:
    """
    Implements classical graph search algorithms to find the ground truth.
    """

    def __init__(self, api_client: OpenAlexClient):
        self.api_client = api_client

    def find_shortest_path_bfs(self, start_id: str, end_id: str):
        """
        Performs a bidirectional Breadth-First Search (BFS) to find the shortest path.
        This establishes the ground truth for path length.
        """
        if start_id == end_id:
            return [start_id], 0

        # Forward search setup
        q_fwd = deque([start_id])
        visited_fwd = {start_id: [start_id]}

        # Backward search setup
        q_bwd = deque([end_id])
        visited_bwd = {end_id: [end_id]}

        self.api_client.reset_api_call_count()
        print("\n--- Starting BFS Ground Truth Calculation ---")

        # Limit search depth to prevent excessive API calls
        for i in range(10):  # Max path length of 20
            # Forward search step
            path_found = self._bfs_step(q_fwd, visited_fwd, visited_bwd)
            if path_found:
                return path_found, self.api_client.get_api_call_count()

            # Backward search step
            path_found = self._bfs_step(q_bwd, visited_bwd, visited_fwd, backward=True)
            if path_found:
                return path_found, self.api_client.get_api_call_count()

        return None, self.api_client.get_api_call_count()

    def _bfs_step(self, queue, visited_self, visited_other, backward=False):
        """Helper for a single expansion step in BFS."""
        level_size = len(queue)
        if level_size == 0:
            return None

        for _ in range(level_size):
            current_id = queue.popleft()
            path = visited_self[current_id]

            # Get all neighbors (undirected)
            neighbors = self.api_client.get_neighbors(current_id)

            for neighbor_id in neighbors:
                if neighbor_id in visited_other:
                    # Intersection found!
                    path_other = visited_other[neighbor_id]
                    if backward:
                        return path_other + path[::-1]
                    else:
                        return path + path_other[::-1]

                if neighbor_id not in visited_self:
                    new_path = path + [neighbor_id]
                    visited_self[neighbor_id] = new_path
                    queue.append(neighbor_id)
        return None


class LLMAgent:
    """
    The LLM-powered agent that attempts to find the shortest path.
    """

    def __init__(self, api_client: OpenAlexClient, llm_provider: str):
        self.api_client = api_client
        self.llm_provider = llm_provider
        self.visited_nodes = set()
        self.start_frontier = {}
        self.end_frontier = {}
        self.parent_map = {}  # Tracks path for reconstruction

    def find_path(self, start_id: str, end_id: str, max_turns=15):
        """
        Main execution loop for the agent.
        """
        self.api_client.reset_api_call_count()
        print("\n--- Starting LLM Agent Run ---")

        # 1. Initialization
        start_paper = self.api_client.get_paper_by_id(start_id)
        end_paper = self.api_client.get_paper_by_id(end_id)

        if not start_paper or not end_paper:
            print("Could not retrieve start or end paper.")
            return None, self.api_client.get_api_call_count()

        self.start_frontier[start_id] = self._extract_metadata(start_paper)
        self.end_frontier[end_id] = self._extract_metadata(end_paper)
        self.visited_nodes.update([start_id, end_id])
        self.parent_map = {start_id: None, end_id: None}

        # 2. Main loop
        for turn in range(max_turns):
            print(f"\n--- Agent's Turn {turn + 1}/{max_turns} ---")

            # 3. LLM as Planner
            prompt = self._build_prompt(start_paper, end_paper)
            llm_decision = self._get_llm_decision(prompt)

            if not llm_decision or "command" not in llm_decision:
                print("Agent failed to make a valid decision. Stopping.")
                break

            # 4. Execute decision
            if llm_decision["command"] == "expand":
                paper_id = llm_decision.get("paper_id")
                direction = llm_decision.get("direction")

                if not paper_id or not direction:
                    print(f"Agent provided invalid expansion command: {llm_decision}")
                    continue

                print(
                    f"Agent decided to expand '{paper_id}' from the '{direction}' frontier."
                )

                # Select frontiers for expansion
                current_frontier = (
                    self.start_frontier if direction == "forward" else self.end_frontier
                )
                other_frontier = (
                    self.end_frontier if direction == "forward" else self.start_frontier
                )

                if paper_id not in current_frontier:
                    print(
                        f"Error: Paper {paper_id} not in the specified frontier. Agent is confused."
                    )
                    continue

                del current_frontier[paper_id]  # Remove from frontier

                # 5. Expand node and update state
                neighbors = self.api_client.get_neighbors(paper_id)
                for neighbor_id in neighbors:
                    if neighbor_id in other_frontier:
                        print("Path found! Intersection at:", neighbor_id)
                        return self._reconstruct_path(
                            paper_id, neighbor_id, direction
                        ), self.api_client.get_api_call_count()

                    if neighbor_id not in self.visited_nodes:
                        self.visited_nodes.add(neighbor_id)
                        self.parent_map[neighbor_id] = paper_id
                        neighbor_paper = self.api_client.get_paper_by_id(neighbor_id)
                        if neighbor_paper:
                            current_frontier[neighbor_id] = self._extract_metadata(
                                neighbor_paper
                            )
            else:
                print(f"Agent returned unknown command: {llm_decision['command']}")

        print("Agent failed to find a path within the turn limit.")
        return None, self.api_client.get_api_call_count()

    def _reconstruct_path(self, meet_point1, meet_point2, direction):
        """Reconstructs the path after a bidirectional search meets."""
        path1 = []
        curr = meet_point1
        while curr is not None:
            path1.append(curr)
            curr = self.parent_map.get(curr)

        path2 = []
        curr = meet_point2
        while curr is not None:
            path2.append(curr)
            curr = self.parent_map.get(curr)

        if direction == "forward":
            # path1 was from start, path2 was from end
            return path1[::-1] + path2
        else:
            # path1 was from end, path2 was from start
            return path2[::-1] + path1

    def _extract_metadata(self, paper_json: dict) -> dict:
        """Extracts key info from the OpenAlex work object."""
        return {
            "title": paper_json.get("title"),
            "abstract": reconstruct_abstract(paper_json.get("abstract_inverted_index")),
            "publication_year": paper_json.get("publication_year"),
        }

    def _build_prompt(self, start_paper, end_paper):
        """Constructs the detailed prompt for the LLM planner."""
        prompt = f"""
        You are a research assistant AI. Your goal is to find the shortest citation path between two academic papers.
        All citation links are undirected. You are conducting a bidirectional search.

        START PAPER: "{start_paper['title']}" (Year: {start_paper['publication_year']})
        END PAPER: "{end_paper['title']}" (Year: {end_paper['publication_year']})

        You have two frontiers of papers. One expanding from the START paper (forward) and one from the END paper (backward).
        Your task is to analyze the papers in both frontiers and decide which SINGLE paper is the most promising to expand next to connect the two.
        Consider semantic similarity, topic overlap, and publication dates to make your choice. A paper published between the start and end dates is often a good candidate.

        CURRENT FRONTIERS:

        FORWARD FRONTIER (from START paper):
        {json.dumps(self.start_frontier, indent=2)}

        BACKWARD FRONTIER (from END paper):
        {json.dumps(self.end_frontier, indent=2)}

        You must respond in JSON format with your decision. The JSON object must contain three fields:
        1. "command": Must always be "expand".
        2. "paper_id": The OpenAlex ID (e.g., "W12345") of the paper you want to expand.
        3. "direction": Either "forward" or "backward", indicating which frontier the chosen paper_id is from.

        Example Response:
        {{
          "command": "expand",
          "paper_id": "W2959555392",
          "direction": "forward"
        }}

        Analyze the frontiers and provide your JSON decision.
        """
        return prompt

    def _get_llm_decision(self, prompt):
        """Makes the API call to OpenRouter to get the agent's next move."""
        if not OPENROUTER_API_KEY:
            print("ERROR: OPENROUTER_API_KEY environment variable not set.")
            # Fallback to a simple heuristic if no key is provided
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
        }
        body = {
            "model": self.llm_provider,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                f"{OPENROUTER_API_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            decision_text = response.json()["choices"][0]["message"]["content"]
            return json.loads(decision_text)
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter API request failed: {e}")
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response as JSON: {decision_text}")

        return None


class EvaluationHarness:
    """
    Runs the agent against the benchmark and calculates performance metrics.
    """

    def __init__(self, ground_truth_path: list, agent_path: list, agent_steps: int):
        self.ground_truth_path = ground_truth_path
        self.agent_path = agent_path
        self.agent_steps = agent_steps
        self.scorecard = {}

    def run_evaluation(self):
        """Calculates all metrics and returns the scorecard."""
        print("\n--- Running Evaluation ---")
        self._calculate_path_success()
        self._calculate_path_optimality()
        self._calculate_step_efficiency()
        self._calculate_reasoning_faithfulness()
        return self.scorecard

    def _calculate_path_success(self):
        self.scorecard["path_success"] = 1 if self.agent_path else 0
        print(f"Path Success: {self.scorecard['path_success']}")

    def _calculate_path_optimality(self):
        if not self.scorecard["path_success"] or not self.ground_truth_path:
            self.scorecard["path_optimality"] = 0
        else:
            len_agent_path = len(self.agent_path) - 1
            len_true_path = len(self.ground_truth_path) - 1
            if len_agent_path == 0:  # Avoid division by zero for same start/end
                self.scorecard["path_optimality"] = 1.0 if len_true_path == 0 else 0
            else:
                self.scorecard["path_optimality"] = len_true_path / len_agent_path
        print(f"Path Optimality: {self.scorecard['path_optimality']:.2f}")

    def _calculate_step_efficiency(self):
        self.scorecard["step_efficiency"] = self.agent_steps
        print(f"Step Efficiency: {self.scorecard['step_efficiency']}")

    def _calculate_reasoning_faithfulness(self):
        # TODO: Implement a function to verify each link in the agent's path
        # by making API calls. For now, we assume it's faithful if a path was found.
        self.scorecard["reasoning_faithfulness"] = 1 if self.agent_path else 0
        print(f"Reasoning Faithfulness: {self.scorecard['reasoning_faithfulness']}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup
    # Using a known 2-hop path for demonstration.
    # Start: "Generative Adversarial Nets" (Goodfellow et al.)
    start_paper_id = "W2127774231"
    # End: "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix)
    end_paper_id = "W2595420983"

    print(
        f"Objective: Find shortest citation path between {start_paper_id} and {end_paper_id}"
    )

    # 2. Get Ground Truth
    bfs_client = OpenAlexClient()
    bfs_search = GraphSearch(api_client=bfs_client)
    ground_truth, bfs_steps = bfs_search.find_shortest_path_bfs(
        start_paper_id, end_paper_id
    )

    if ground_truth:
        print(
            f"\nGround Truth Path (BFS): {ground_truth} (Length: {len(ground_truth)-1}, Cost: {bfs_steps} API calls)"
        )
    else:
        print(
            "\nCould not find a ground truth path with BFS. The papers may not be connected."
        )
        # Exit if no path exists
        exit()

    # 3. Run LLM Agent
    # Recommended model: Google Gemini Flash, Cohere Command R, or Mistral 7B Instruct
    agent_client = OpenAlexClient()
    agent = LLMAgent(api_client=agent_client, llm_provider="google/gemini-flash-1.5")
    agent_found_path, agent_api_calls = agent.find_path(start_paper_id, end_paper_id)

    if agent_found_path:
        print(f"Agent Path: {agent_found_path} (Length: {len(agent_found_path)-1})")
    else:
        print("Agent did not find a path.")
    print(f"Agent used {agent_api_calls} API calls (steps).")

    # 4. Evaluate Performance
    evaluator = EvaluationHarness(
        ground_truth_path=ground_truth,
        agent_path=agent_found_path,
        agent_steps=agent_api_calls,
    )
    final_scorecard = evaluator.run_evaluation()

    print("\n--- Final Scorecard ---")
    print(json.dumps(final_scorecard, indent=2))
