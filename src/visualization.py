# visualization.py
# Generates network visualization files for VOSviewer using NetworkX and nx2vos.

import logging
import networkx as nx
import nx2vos
from src.openalex_client import OpenAlexClient


def create_vosviewer_files(
    ground_truth_path: list, agent_path: list, output_prefix: str
):
    """
    Creates a single NetworkX graph containing both paths and converts it
    to VOSviewer format using nx2vos.

    Args:
        ground_truth_path (list): List of paper IDs for the optimal path.
        agent_path (list): List of paper IDs for the agent's found path.
        output_prefix (str): The prefix for the output files (e.g., 'visualization').
    """
    if not ground_truth_path and not agent_path:
        logging.warning("No paths provided for visualization. Skipping.")
        return

    logging.info(f"Generating combined NetworkX graph for VOSviewer visualization...")

    G = nx.Graph()
    client = OpenAlexClient()

    # Collect all unique paper IDs from both paths
    all_path_ids = set(ground_truth_path or []) | set(agent_path or [])

    # Fetch details for all unique papers to add as nodes
    logging.info(f"Fetching details for {len(all_path_ids)} unique papers...")
    for paper_id in all_path_ids:
        paper_data = client.get_paper_by_id(paper_id)
        if paper_data:
            in_ground = paper_id in (ground_truth_path or [])
            in_agent = paper_id in (agent_path or [])
            path_membership = (
                "both"
                if in_ground and in_agent
                else "ground_truth"
                if in_ground
                else "agent_path"
            )

            G.add_node(
                paper_id,
                label=paper_data.get("title", "No Title"),
                year=paper_data.get("publication_year", "N/A"),
                path_membership=path_membership,
                # Add a score for potential clustering in VOSviewer
                # VOSviewer uses 'weight' for node size/color by default
                weight=1
                if path_membership == "ground_truth"
                else 2
                if path_membership == "agent_path"
                else 3,
                url=paper_data.get("doi", ""),
            )

    # Add edges for the ground truth path with a specific weight
    if ground_truth_path and len(ground_truth_path) > 1:
        for i in range(len(ground_truth_path) - 1):
            source, target = ground_truth_path[i], ground_truth_path[i + 1]
            if G.has_edge(source, target):
                G[source][target]["link_strength"] += 1
                G[source][target]["path_type"] = "both"
            else:
                G.add_edge(source, target, path_type="ground_truth", link_strength=1)

    # Add edges for the agent path with a specific weight
    if agent_path and len(agent_path) > 1:
        for i in range(len(agent_path) - 1):
            source, target = agent_path[i], agent_path[i + 1]
            if G.has_edge(source, target):
                G[source][target]["link_strength"] += 1
                G[source][target]["path_type"] = "both"
            else:
                G.add_edge(source, target, path_type="agent_path", link_strength=1)

    # map_file = f"{output_prefix}_map.txt"
    # network_file = f"{output_prefix}_network.txt"
    # nx2vos.write_vos_map(
    #     G,
    #     map_file,
    # )
    # nx2vos.write_vos_network(
    #     G,
    #     network_file,
    # )
    nx2vos.write_vos_json(G, "graph.json")

    logging.info(f"VOSviewer file created: graph.json")
