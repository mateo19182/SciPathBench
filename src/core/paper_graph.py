# paper_graph.py
import json
import logging

class PaperGraph:
    """Unified graph structure for papers and citations."""
    def __init__(self):
        self.nodes = {}  # paper_id -> {title, year, concepts, doi, node_type}
        self.edges = []  # [{source, target}]
        self.agent_path = []  # Actual sequence of papers agent expanded
        
    def add_node(self, paper_id: str, paper_data: dict, node_type: str):
        """Add or update a node in the graph."""
        if not paper_data:
            return
            
        self.nodes[paper_id] = {
            "title": paper_data.get("title", "Unknown Title"),
            "year": paper_data.get("publication_year", "Unknown"),
            "concepts": [c.get("display_name") for c in paper_data.get("concepts", [])],
            "doi": paper_data.get("ids", {}).get("doi", "N/A"),
            "node_type": node_type
        }
    
    def add_edge(self, source: str, target: str):
        """Add an edge to the graph."""
        edge = {"source": source, "target": target}
        if edge not in self.edges:
            self.edges.append(edge)
    
    def get_node_metadata_for_llm(self, paper_id: str) -> dict:
        """Get simplified metadata for LLM context."""
        node = self.nodes.get(paper_id, {})
        return {
            "title": node.get("title", "Unknown"),
            "publication_year": node.get("year", "Unknown"),
            "concepts": node.get("concepts", [])[:3]  # Top 3 concepts
        }
    
    def save_to_file(self, filepath: str):
        """Save graph to JSON file."""
        try:
            data = {
                "nodes": self.nodes,
                "edges": self.edges,
                "agent_path": self.agent_path
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"Graph saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save graph: {e}")
