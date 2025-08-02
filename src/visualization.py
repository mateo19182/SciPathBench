import logging
import networkx as nx
import nx2vos
from src.openalex_client import OpenAlexClient
from pyvis.network import Network

def extract_metadata(client, paper_id):
    return client.get_paper_by_id(paper_id).get("title"), client.get_paper_by_id(paper_id).get("publication_year"), client.get_paper_by_id(paper_id).get("concepts", []), client.get_paper_by_id(paper_id).get("ids", {}).get("doi", "N/A")

def create_vosviewer_files(ground_truth_path: list, agent_path: list, output_prefix: str
):
    """
    Creates a NetworkX graph with rich metadata for visualization in VOSviewer.
    """
    if not ground_truth_path and not agent_path:
        logging.warning("No paths provided for visualization. Skipping.")
        return

    logging.info(f"{ground_truth_path}, {agent_path}, {output_prefix=}")

    G = nx.Graph()
    client = OpenAlexClient()

    # Collect all unique paper IDs from both paths
    all_path_ids = set(ground_truth_path or []) | set(agent_path or [])

    # --- Add Nodes with Metadata ---
    # concept_cluster_map = {}  # Map concept to cluster ID
    # cluster_counter = 1
    for paper_id in all_path_ids:
        title, year, concepts, doi = extract_metadata(client, paper_id)
        if not title or not year:
            logging.warning(f"Missing metadata for paper {paper_id}")
            continue

        in_ground = paper_id in (ground_truth_path or [])
        in_agent = paper_id in (agent_path or [])
        path_membership = (
            "both"
            if in_ground and in_agent
            else "ground_truth"
            if in_ground
            else "agent_path"
        )


        # # Assign a cluster based on primary concept
        # primary_concept = concepts[0] if concepts else "Unknown"
        # if primary_concept not in concept_cluster_map:
        #     concept_cluster_map[primary_concept] = cluster_counter
        #     cluster_counter += 1
        # cluster_id = concept_cluster_map[primary_concept]

        # Add node with rich attributes
        G.add_node(
            paper_id,
            label=title,
            year=year,
            path_membership=path_membership,
            weight=1 if path_membership == "ground_truth" else 2 if path_membership == "agent_path" else 3,
            url=doi,
            # cluster=cluster_id,
        )

    # --- Add Edges with Path Type and Strength ---
    def add_path_edges(path, path_type):
        if not path or len(path) < 2:
            return
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                G[u][v]["link_strength"] = G[u][v].get("link_strength", 1) + 1
                G[u][v]["path_type"] = "both"
            else:
                G.add_edge(u, v, path_type=path_type, link_strength=1)

    add_path_edges(ground_truth_path, "ground_truth")
    add_path_edges(agent_path, "agent_path")

    # --- Export to VOSviewer JSON ---
    nx2vos.write_vos_json(G, f"output/{output_prefix}.json")

    # --- Export to Interactive HTML using PyVis ---
    logging.info(f"Generating interactive HTML visualization...")
    # Create a PyVis network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=False) # Set directed=False for undirected graph like yours

    # Define colors for path membership
    color_map = {
        "ground_truth": "#1f77b4",  # Blue
        "agent_path": "#ff7f0e",    # Orange
        "both": "#2ca02c"           # Green
    }

    # Add nodes from NetworkX graph to PyVis network
    for node, data in G.nodes(data=True):
        # Get attributes, providing defaults if missing
        label = data.get('label', f'Paper {node}')
        title = f"<b>{data.get('label', 'N/A')}</b><br>" \
                f"ID: {node}<br>" \
                f"Year: {data.get('year', 'N/A')}<br>" \
                f"Path: {data.get('path_membership', 'N/A')}<br>" \
                f"DOI: <a href='{data.get('url', '#')}'>{data.get('url', 'N/A')}</a>" # Create tooltip with rich info
        color = color_map.get(data.get('path_membership'), "#999999") # Default grey

        # --- Determine if node is start/end ---
        is_start = False
        is_end = False
        is_ground_truth_end = False
        is_agent_end = False
        
        # Check for start nodes
        if ground_truth_path and node == ground_truth_path[0]:
            is_start = True
        if agent_path and node == agent_path[0] and node != ground_truth_path[0]:
             is_start = True
        
        # Check for end nodes (ground truth end is the "correct" end)
        if ground_truth_path and node == ground_truth_path[-1]:
            is_end = True
            is_ground_truth_end = True
        if agent_path and node == agent_path[-1]:
            is_agent_end = True
            
        # If agent end is different from ground truth end, show it
        if is_agent_end and not is_ground_truth_end:
            is_end = True

        # --- Customize appearance for start/end ---
        if is_start or is_end:
            # Override color for start/end nodes
            if is_start and is_end:
                color = "#ff0000"  # Red if both start and end
            elif is_start:
                color = "#00ff00"   # Green if start only
            elif is_end:
                if is_ground_truth_end and is_agent_end:
                    color = "#ff0000"  # Red if it's the correct end AND agent also ends here
                elif is_ground_truth_end:
                    color = "#2ca02c"  # Correct end (green-ish, different from start)
                elif is_agent_end:
                    color = "#ff7f0e"  # Agent end only (orange)
            size = 35  # Make them larger
            # Modify label to indicate start/end
            marker = ""
            if is_start and is_end:
                marker = " [S/E]"
            elif is_start:
                marker = " [Start]"
            elif is_end:
                if is_ground_truth_end and is_agent_end:
                    marker = " [End]"  # Correct end that agent also found
                elif is_ground_truth_end:
                    marker = " [End]"  # Correct end
                elif is_agent_end:
                    marker = " [Dead Run]"  # Agent's wrong end
            display_label = (label + marker)
        else:
            size = 20 + data.get('weight', 1) * 5
            display_label = label[:50] + "..." if len(label) > 50 else label

        # Add node to PyVis network
        net.add_node(node, label=display_label,
                           title=title, # Tooltip
                           color=color,
                           size=size)

    # Add edges from NetworkX graph to PyVis network
    for u, v, data in G.edges(data=True):
        width = data.get('link_strength', 1) * 2 # Adjust width based on strength
        edge_label = data.get('path_type', '')
        net.add_edge(u, v, width=width, title=edge_label) # Add edge with tooltip for type

    # Generate and save the HTML file
    output_file = f"output/{output_prefix}.html"
    net.write_html(output_file) # This creates the interactive HTML file

    logging.info(f"Interactive HTML visualization created: {output_file}")
    
    logging.info(f"VOSviewer file created: output/{output_prefix}.json")