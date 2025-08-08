# web_human_agent.py
# Web-based human agent that allows users to play the pathfinding game through a web interface.

import logging
import asyncio
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from src.services.openalex_client import OpenAlexClient
from src.core.paper_graph import PaperGraph


class WebHumanAgent:
    """Web-based human agent for the pathfinding game."""
    
    def __init__(self, api_client: OpenAlexClient, message_callback: Callable = None):
        self.api_client = api_client
        self.message_callback = message_callback
        self.graph = PaperGraph()
        self.visited_nodes = set()
        self.frontier = {}  # paper_id -> metadata for display
        self.start_id = None
        self.end_id = None
        self.max_turns = 0
        self.current_turn = 0
        self.ground_truth_path = None
        self.game_active = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def send_message(self, message_type: str, data: dict):
        """Send a message to the web client."""
        if self.message_callback:
            await self.message_callback({
                "type": message_type,
                "data": data
            })
    
    def _extract_paper_metadata(self, paper_data: dict) -> dict:
        """Extract and format paper metadata for web display."""
        if not paper_data:
            return {}
            
        # Extract authors properly from OpenAlex format
        authors = []
        if 'authorships' in paper_data:
            for authorship in paper_data['authorships'][:3]:  # First 3 authors
                author = authorship.get('author', {})
                display_name = author.get('display_name', 'Unknown Author')
                authors.append(display_name)
        
        # Extract concepts
        concepts = []
        if 'concepts' in paper_data:
            for concept in paper_data['concepts'][:3]:  # First 3 concepts
                concepts.append({
                    'display_name': concept.get('display_name', 'Unknown Concept'),
                    'score': concept.get('score', 0)
                })
        
        return {
            'id': paper_data.get('id', '').split('/')[-1],  # Extract just the ID part
            'title': paper_data.get('title', 'Unknown Title'),
            'year': paper_data.get('publication_year', 'Unknown'),
            'authors': authors,
            'concepts': concepts,
            'cited_by_count': paper_data.get('cited_by_count', 0),
            'doi': paper_data.get('ids', {}).get('doi', 'N/A')
        }
    
    async def initialize_game(self, start_id: str, end_id: str, max_turns: int, ground_truth_path: list = None) -> bool:
        """Initialize the game with start and end papers."""
        try:
            self.start_id = start_id
            self.end_id = end_id
            self.max_turns = max_turns
            self.current_turn = 0
            self.ground_truth_path = ground_truth_path
            self.game_active = True
            
            # Reset state
            self.visited_nodes = set()
            self.frontier = {}
            self.graph = PaperGraph()
            
            # Get start and end papers (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            start_paper, end_paper = await asyncio.gather(
                loop.run_in_executor(self.executor, self.api_client.get_paper_by_id, start_id),
                loop.run_in_executor(self.executor, self.api_client.get_paper_by_id, end_id)
            )
            
            if not start_paper or not end_paper:
                await self.send_message("error", {"message": "Could not retrieve start or end paper"})
                return False
                
            # Add start and end nodes to graph
            self.graph.add_node(start_id, start_paper, "start")
            self.graph.add_node(end_id, end_paper, "end")
            
            # Initialize with start node
            self.visited_nodes.add(start_id)
            self.graph.agent_path.append(start_id)
            
            # Expand start node automatically (run in executor to avoid blocking)
            initial_neighbors = await loop.run_in_executor(
                self.executor, self.api_client.get_neighbors, start_id
            )
            logging.info(f"Found {len(initial_neighbors)} neighbors for start paper")
            
            # Get all neighbor papers in parallel
            new_neighbor_ids = [n for n in initial_neighbors if n not in self.visited_nodes]
            if new_neighbor_ids:
                neighbor_papers = await loop.run_in_executor(
                    self.executor, self.api_client.get_many_papers, new_neighbor_ids
                )
                
                for neighbor_id in initial_neighbors:
                    self.graph.add_edge(start_id, neighbor_id)
                    if neighbor_id not in self.visited_nodes:
                        self.visited_nodes.add(neighbor_id)
                        neighbor_paper = neighbor_papers.get(neighbor_id)
                        if neighbor_paper:
                            self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                            self.frontier[neighbor_id] = self._extract_paper_metadata(neighbor_paper)
            
            # Prepare available papers for display
            available_papers = list(self.frontier.values())
            
            # Send game initialization message
            await self.send_message("game_initialized", {
                "start_paper": self._extract_paper_metadata(start_paper),
                "end_paper": self._extract_paper_metadata(end_paper),
                "optimal_length": len(ground_truth_path) - 1 if ground_truth_path else None,
                "current_turn": self.current_turn,
                "available_papers": available_papers,
                "current_path": [self._extract_paper_metadata(start_paper)]
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Error initializing game: {e}", exc_info=True)
            await self.send_message("error", {"message": f"Failed to initialize game: {str(e)}"})
            return False
    
    async def make_choice(self, paper_id: str) -> dict:
        """Process a player's paper choice."""
        if not self.game_active:
            return {"success": False, "error": "Game not active"}
            
        if paper_id not in self.frontier:
            return {"success": False, "error": "Invalid paper choice"}
        
        try:
            self.current_turn += 1
            
            # Add to agent path and update node type
            self.graph.agent_path.append(paper_id)
            if paper_id in self.graph.nodes:
                self.graph.nodes[paper_id]["node_type"] = "agent_path"
            
            # Remove from frontier
            chosen_paper_metadata = self.frontier[paper_id]
            del self.frontier[paper_id]
            
            # Get neighbors of expanded paper (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            neighbors = await loop.run_in_executor(
                self.executor, self.api_client.get_neighbors, paper_id
            )
            logging.info(f"Found {len(neighbors)} neighbors for paper {paper_id}")
            
            # Check if we found the target
            if self.end_id in neighbors:
                # Game won!
                self.graph.agent_path.append(self.end_id)
                if self.end_id in self.graph.nodes:
                    self.graph.nodes[self.end_id]["node_type"] = "agent_path"
                
                self.game_active = False
                
                # Build final path for display
                final_path = []
                for pid in self.graph.agent_path:
                    if pid in self.graph.nodes:
                        paper_data = self.graph.nodes[pid]
                        final_path.append({
                            'id': pid,
                            'title': paper_data.get('title', 'Unknown Title'),
                            'year': paper_data.get('year', 'Unknown')
                        })
                
                return {
                    "success": True,
                    "game_complete": True,
                    "won": True,
                    "path": final_path,
                    "turns_used": self.current_turn,
                    "optimal_turns": len(self.ground_truth_path) - 1 if self.ground_truth_path else None
                }
            
            # Get all new neighbor papers in parallel
            new_neighbor_ids = [n for n in neighbors if n not in self.visited_nodes and n != self.end_id]
            if new_neighbor_ids:
                neighbor_papers = await loop.run_in_executor(
                    self.executor, self.api_client.get_many_papers, new_neighbor_ids
                )
            else:
                neighbor_papers = {}
            
            # Add new neighbors to frontier and graph
            for neighbor_id in neighbors:
                self.graph.add_edge(paper_id, neighbor_id)
                
                if neighbor_id not in self.visited_nodes and neighbor_id != self.end_id:
                    self.visited_nodes.add(neighbor_id)
                    neighbor_paper = neighbor_papers.get(neighbor_id)
                    if neighbor_paper:
                        self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                        self.frontier[neighbor_id] = self._extract_paper_metadata(neighbor_paper)
            
            # Check if we're out of turns
            if self.current_turn >= self.max_turns:
                self.game_active = False
                return {
                    "success": True,
                    "game_complete": True,
                    "won": False,
                    "reason": f"Turn limit reached ({self.max_turns} turns)",
                    "turns_used": self.current_turn,
                    "path": []
                }
            
            # Check if we're out of options
            if not self.frontier:
                self.game_active = False
                return {
                    "success": True,
                    "game_complete": True,
                    "won": False,
                    "reason": "No more papers to explore",
                    "turns_used": self.current_turn,
                    "path": []
                }
            
            # Continue game - prepare current path for display
            current_path = []
            for pid in self.graph.agent_path:
                if pid in self.graph.nodes:
                    paper_data = self.graph.nodes[pid]
                    current_path.append({
                        'id': pid,
                        'title': paper_data.get('title', 'Unknown Title'),
                        'year': paper_data.get('year', 'Unknown')
                    })
            
            # Prepare available papers for display
            available_papers = list(self.frontier.values())
            
            return {
                "success": True,
                "game_complete": False,
                "current_turn": self.current_turn,
                "available_papers": available_papers,
                "current_path": current_path,
                "chosen_paper": chosen_paper_metadata
            }
            
        except Exception as e:
            logging.error(f"Error processing choice: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to process choice: {str(e)}"}
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)