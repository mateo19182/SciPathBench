import logging
import json
import networkx as nx
import nx2vos
from pyvis.network import Network

def create_vosviewer_files(ground_truth_path: list, agent_path: list, output_prefix: str, reference_graph_path: str = "output/reference_graph.json"):
    """
    Creates a NetworkX graph with rich metadata for visualization in VOSviewer.
    Uses the unified graph structure from the agent.
    """
    if not ground_truth_path and not agent_path:
        logging.warning("No paths provided for visualization. Skipping.")
        return

    logging.info(f"Creating visualization: ground_truth={len(ground_truth_path or [])} nodes, agent_path={len(agent_path or [])} nodes")

    G = nx.Graph()
    graph_data = None

    # Load the unified graph data from the agent
    try:
        with open(reference_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        logging.info(f"Loaded graph data from {reference_graph_path}")
    except FileNotFoundError:
        logging.warning(f"Graph file not found at {reference_graph_path}. Creating minimal visualization.")
    except Exception as e:
        logging.error(f"Failed to load graph data: {e}")

    # Get nodes and edges from graph data
    nodes = graph_data.get("nodes", {}) if graph_data else {}
    edges = graph_data.get("edges", []) if graph_data else []
    actual_agent_path = graph_data.get("agent_path", []) if graph_data else []

    # Use actual agent path if available, otherwise fall back to provided agent_path
    if actual_agent_path:
        agent_path = actual_agent_path
        logging.info(f"Using actual agent path from graph: {len(agent_path)} steps")

    # Collect all unique paper IDs
    all_paper_ids = set()
    if ground_truth_path:
        all_paper_ids.update(ground_truth_path)
    if agent_path:
        all_paper_ids.update(agent_path)
    all_paper_ids.update(nodes.keys())

    # Add nodes to NetworkX graph
    for paper_id in all_paper_ids:
        node_data = nodes.get(paper_id, {})
        
        # Determine node type based on paths
        in_ground_truth = paper_id in (ground_truth_path or [])
        in_agent_path = paper_id in (agent_path or [])
        
        if in_ground_truth and in_agent_path:
            path_membership = "both"
        elif in_ground_truth:
            path_membership = "ground_truth"
        elif in_agent_path:
            path_membership = "agent_path"
        else:
            path_membership = "referenced_only"

        # Get node attributes
        title = node_data.get("title", f"Paper {paper_id}")
        year = node_data.get("year", "Unknown")
        doi = node_data.get("doi", "N/A")
        node_type = node_data.get("node_type", "referenced")

        # Add node to graph
        G.add_node(
            paper_id,
            label=title,
            year=year,
            path_membership=path_membership,
            node_type=node_type,
            weight=_get_node_weight(path_membership),
            url=doi,
        )

    # Add edges to NetworkX graph
    def add_path_edges(path, path_type):
        """Add edges for a specific path."""
        if not path or len(path) < 2:
            return
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                # Upgrade existing edge
                G[u][v]["link_strength"] = G[u][v].get("link_strength", 1) + 1
                if G[u][v].get("path_type") != path_type:
                    G[u][v]["path_type"] = "both"
            else:
                G.add_edge(u, v, path_type=path_type, link_strength=1)

    # Add path edges
    add_path_edges(ground_truth_path, "ground_truth")
    add_path_edges(agent_path, "agent_path")

    # Add reference edges from the graph data
    for edge in edges:
        u = edge.get("source")
        v = edge.get("target")
        if not u or not v:
            continue

        # Ensure both nodes exist
        for node_id in [u, v]:
            if not G.has_node(node_id):
                node_data = nodes.get(node_id, {})
                G.add_node(
                    node_id,
                    label=node_data.get("title", f"Paper {node_id}"),
                    year=node_data.get("year", "Unknown"),
                    path_membership="referenced_only",
                    node_type=node_data.get("node_type", "referenced"),
                    weight=0,
                    url=node_data.get("doi", "N/A"),
                )

        # Add edge if not already present from paths
        if not G.has_edge(u, v):
            G.add_edge(u, v, path_type="referenced_only", link_strength=1)

    # Export to VOSviewer JSON
    nx2vos.write_vos_json(G, f"output/{output_prefix}.json")
    logging.info(f"VOSviewer file created: output/{output_prefix}.json")

    # Create interactive HTML visualization
    _create_html_visualization(G, ground_truth_path, agent_path, output_prefix)

def _get_node_weight(path_membership):
    """Get node weight based on path membership."""
    weights = {
        "ground_truth": 1,
        "agent_path": 2,
        "both": 3,
        "referenced_only": 0
    }
    return weights.get(path_membership, 0)

def _create_html_visualization(G, ground_truth_path, agent_path, output_prefix):
    """Create interactive HTML visualization using PyVis."""
    logging.info("Generating interactive HTML visualization...")
    
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=False)
    
    # Configure physics for better node separation
    net.set_options("""
    var options = {
    "physics": {
        "enabled": true,
        "stabilization": {
        "enabled": true,
        "iterations": 100,
        "updateInterval": 100
        },
        "barnesHut": {
        "gravitationalConstant": -2500,
        "centralGravity": 0.3,
        "springLength": 200,
        "springConstant": 0.05,
        "damping": 0.09,
        "avoidOverlap": 10
        },
        "minVelocity": 0.75,
        "maxVelocity": 10,
        "solver": "barnesHut",
        "adaptiveTimestep": true
    },
    "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
    }
    }
    """)

    # Define colors for different node types
    color_map = {
        "ground_truth": "#1f77b4",  # Blue
        "agent_path": "#ff7f0e",    # Orange
        "both": "#2ca02c",          # Green
        "referenced_only": "#999999" # Grey
    }

    # Add nodes to PyVis network
    for node, data in G.nodes(data=True):
        label = data.get('label', f'Paper {node}')
        
        # Create tooltip with rich info
        title = f"<b>{data.get('label', 'N/A')}</b><br>" \
                f"ID: {node}<br>" \
                f"Year: {data.get('year', 'N/A')}<br>" \
                f"Path: {data.get('path_membership', 'N/A')}<br>" \
                f"Type: {data.get('node_type', 'N/A')}<br>" \
                f"DOI: <a href='{data.get('url', '#')}'>{data.get('url', 'N/A')}</a>"
        
        # Determine node appearance
        path_membership = data.get('path_membership', 'referenced_only')
        color = color_map.get(path_membership, "#999999")
        
        # Special handling for start/end nodes
        is_start = _is_start_node(node, ground_truth_path, agent_path)
        is_end = _is_end_node(node, ground_truth_path, agent_path)
        is_gt_end = ground_truth_path and node == ground_truth_path[-1]
        is_agent_end = agent_path and node == agent_path[-1]
        
        if is_start or is_end:
            size = 35
            if is_start and is_end:
                color = "#ff0000"  # Red for start/end
                marker = " [S/E]"
            elif is_start:
                color = "#00ff00"  # Green for start
                marker = " [Start]"
            elif is_end:
                if is_gt_end and is_agent_end:
                    color = "#2ca02c"  # Correct end - agent found the right target
                    marker = " [Success]"
                elif is_gt_end:
                    color = "#1f77b4"  # Ground truth end (blue)
                    marker = " [Target]"
                elif is_agent_end:
                    color = "#ff7f0e"  # Agent's wrong end (orange)
                    marker = " [Failed]"
            display_label = label + marker
        else:
            # For referenced nodes, show title only on hover
            if path_membership == "referenced_only":
                size = 12
                display_label = "•"  # Just a dot for referenced nodes
            else:
                size = 20 + data.get('weight', 1) * 5
                display_label = label

        net.add_node(node, label=display_label, title=title, color=color, size=size)

    # Add edges to PyVis network
    for u, v, data in G.edges(data=True):
        width = data.get('link_strength', 1) * 2
        edge_type = data.get('path_type', 'referenced_only')
        
        edge_colors = {
            "referenced_only": "#cccccc",
            "ground_truth": "#1f77b4",
            "agent_path": "#ff7f0e",
            "both": "#2ca02c"
        }
        
        color = edge_colors.get(edge_type, "#cccccc")
        if edge_type == "referenced_only":
            width = 1.5
            
        net.add_edge(u, v, width=width, title=edge_type, color=color)

    # Save HTML file
    output_file = f"output/{output_prefix}.html"
    net.write_html(output_file)
    logging.info(f"Interactive HTML visualization created: {output_file}")

def _is_start_node(node, ground_truth_path, agent_path):
    """Check if node is a start node."""
    is_gt_start = ground_truth_path and node == ground_truth_path[0]
    is_agent_start = agent_path and node == agent_path[0]
    return is_gt_start or is_agent_start

def _is_end_node(node, ground_truth_path, agent_path):
    """Check if node is an end node."""
    is_gt_end = ground_truth_path and node == ground_truth_path[-1]
    is_agent_end = agent_path and node == agent_path[-1]
    return is_gt_end or is_agent_end

def _is_correct_end(node, ground_truth_path, agent_path):
    """Check if this is the correct end node (ground truth end)."""
    is_gt_end = ground_truth_path and node == ground_truth_path[-1]
    is_agent_end = agent_path and node == agent_path[-1]
    return is_gt_end and is_agent_end
