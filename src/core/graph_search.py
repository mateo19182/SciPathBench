# graph_search.py
# Implements classical graph search algorithms to find the ground truth.

import logging
from collections import deque
from src.services.openalex_client import OpenAlexClient
from src.config import BFS_MAX_DEPTH


class GraphSearch:
    """Implements BFS to find the ground truth shortest path."""

    def __init__(self, api_client: OpenAlexClient):
        self.api_client = api_client

    def find_shortest_path_bfs(self, start_id: str, end_id: str):
        """Performs a bidirectional BFS to find the shortest path."""
        if start_id == end_id:
            return [start_id], 0

        q_fwd = deque([start_id])
        visited_fwd = {start_id: [start_id]}
        q_bwd = deque([end_id])
        visited_bwd = {end_id: [end_id]}

        logging.info("--- Starting BFS Ground Truth Calculation ---")

        for i in range(BFS_MAX_DEPTH):
            logging.info(f"BFS Depth: {i + 1}")
            path_found = self._bfs_step(q_fwd, visited_fwd, visited_bwd)
            if path_found:
                return path_found

            path_found = self._bfs_step(q_bwd, visited_bwd, visited_fwd, backward=True)
            if path_found:
                return path_found

        return None

    def _bfs_step(self, queue, visited_self, visited_other, backward=False):
        """Helper for a single expansion step in BFS."""
        level_size = len(queue)
        if level_size == 0:
            return None

        for _ in range(level_size):
            current_id = queue.popleft()
            path = visited_self[current_id]

            neighbors = self.api_client.get_neighbors(current_id)

            for neighbor_id in neighbors:
                if neighbor_id in visited_other:
                    path_other = visited_other[neighbor_id]
                    return (
                        path + path_other[::-1]
                        if not backward
                        else path_other + path[::-1]
                    )

                if neighbor_id not in visited_self:
                    new_path = path + [neighbor_id]
                    visited_self[neighbor_id] = new_path
                    queue.append(neighbor_id)
        return None
