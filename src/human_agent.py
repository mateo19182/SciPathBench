# human_agent.py
# Interactive human agent that allows users to play the pathfinding game manually.

import logging
from src.openalex_client import OpenAlexClient
from src.paper_graph import PaperGraph


class HumanAgent:
    """Interactive human agent for the pathfinding game."""
    
    def __init__(self, api_client: OpenAlexClient):
        self.api_client = api_client
        self.graph = PaperGraph()
        self.visited_nodes = set()
        self.frontier = {}  # paper_id -> metadata for display
        
    def find_path(self, start_id: str, end_id: str, max_turns: int, ground_truth_path: list = None):
        """Main interactive game loop for human player."""
        print("\n" + "="*80)
        print("🎯 SCIPATHBENCH INTERACTIVE MODE")
        print("="*80)
        
        # Get start and end papers
        start_paper = self.api_client.get_paper_by_id(start_id)
        end_paper = self.api_client.get_paper_by_id(end_id)
        
        if not start_paper or not end_paper:
            print("❌ Could not retrieve start or end paper.")
            return None, None
            
        # Add start and end nodes to graph
        self.graph.add_node(start_id, start_paper, "start")
        self.graph.add_node(end_id, end_paper, "end")
        
        # Show game objective
        print(f"\n🎯 OBJECTIVE:")
        print(f"   START: '{start_paper.get('title')}' ({start_paper.get('publication_year')})")
        print(f"   END:   '{end_paper.get('title')}' ({end_paper.get('publication_year')})")
        
        if ground_truth_path:
            optimal_length = len(ground_truth_path) - 1
            print(f"   OPTIMAL PATH LENGTH: {optimal_length} steps")
            
        # Initialize with start node
        print(f"\n🚀 Starting from: '{start_paper.get('title')}'")
        self.visited_nodes.add(start_id)
        self.graph.agent_path.append(start_id)
        
        # Expand start node automatically
        print("   Expanding start paper...")
        initial_neighbors = self.api_client.get_neighbors(start_id)
        print(f"   Found {len(initial_neighbors)} connected papers")
        
        # Get all neighbor papers in parallel
        new_neighbor_ids = [n for n in initial_neighbors if n not in self.visited_nodes]
        if new_neighbor_ids:
            print(f"   Loading {len(new_neighbor_ids)} new papers...")
            neighbor_papers = self.api_client.get_many_papers(new_neighbor_ids)
            
            for neighbor_id in initial_neighbors:
                self.graph.add_edge(start_id, neighbor_id)
                if neighbor_id not in self.visited_nodes:
                    self.visited_nodes.add(neighbor_id)
                    neighbor_paper = neighbor_papers.get(neighbor_id)
                    if neighbor_paper:
                        self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                        self.frontier[neighbor_id] = self.graph.get_node_metadata_for_llm(neighbor_id)
        
        # Main game loop
        for turn in range(max_turns):
            print(f"\n" + "-"*60)
            print(f"🎮 TURN {turn + 1}/{max_turns}")
            print("-"*60)
            
            if not self.frontier:
                print("❌ No more papers to explore. Game over!")
                break
                
            # Show current path
            self._display_current_path()
            
            # Show frontier options
            paper_choice = self._display_frontier_and_get_choice()
            
            if paper_choice is None:
                print("👋 Game ended by player.")
                break
                
            if paper_choice not in self.frontier:
                print("❌ Invalid choice. Please try again.")
                continue
                
            # Player expands this paper
            paper_title = self.frontier[paper_choice]['title']
            print(f"\n📖 Expanding: '{paper_title}'")
            
            # Add to agent path and update node type
            self.graph.agent_path.append(paper_choice)
            self.graph.nodes[paper_choice]["node_type"] = "agent_path"
            
            # Remove from frontier
            del self.frontier[paper_choice]
            
            # Get neighbors of expanded paper
            neighbors = self.api_client.get_neighbors(paper_choice)
            print(f"   Found {len(neighbors)} connected papers")
            
            # Get all new neighbor papers in parallel
            new_neighbor_ids = [n for n in neighbors if n not in self.visited_nodes and n != end_id]
            if new_neighbor_ids:
                print(f"   Loading {len(new_neighbor_ids)} new papers...")
                neighbor_papers = self.api_client.get_many_papers(new_neighbor_ids)
            else:
                neighbor_papers = {}
            
            for neighbor_id in neighbors:
                self.graph.add_edge(paper_choice, neighbor_id)
                
                # Check if we found the target
                if neighbor_id == end_id:
                    print(f"\n🎉 SUCCESS! You found the target paper!")
                    self.graph.agent_path.append(end_id)
                    self.graph.nodes[end_id]["node_type"] = "agent_path"
                    
                    # Show final path
                    self._display_final_results(ground_truth_path)
                    
                    # Save graph and return success
                    self.graph.save_to_file("output/reference_graph.json")
                    return self.graph.agent_path, None
                    
                # Add new neighbors to frontier
                if neighbor_id not in self.visited_nodes:
                    self.visited_nodes.add(neighbor_id)
                    neighbor_paper = neighbor_papers.get(neighbor_id)
                    if neighbor_paper:
                        self.graph.add_node(neighbor_id, neighbor_paper, "referenced")
                        self.frontier[neighbor_id] = self.graph.get_node_metadata_for_llm(neighbor_id)
        
        # Player failed to find path within turn limit
        print(f"\n⏰ Turn limit reached! You didn't find the path in {max_turns} turns.")
        self._display_final_results(ground_truth_path)
        self.graph.save_to_file("output/reference_graph.json")
        return None, self.graph.agent_path
    
    def _display_current_path(self):
        """Display the current path taken by the player."""
        print(f"\n📍 CURRENT PATH ({len(self.graph.agent_path)} papers):")
        for i, paper_id in enumerate(self.graph.agent_path):
            node = self.graph.nodes.get(paper_id, {})
            title = node.get("title", f"Paper {paper_id}")
            year = node.get("year", "Unknown")
            arrow = " -> " if i < len(self.graph.agent_path) - 1 else ""
            print(f"   {i+1}. {title} ({year}){arrow}")
    
    def _display_frontier_and_get_choice(self):
        """Display frontier options and get player's choice."""
        print(f"\n🔍 FRONTIER ({len(self.frontier)} papers to choose from):")
        
        if len(self.frontier) == 0:
            return None
            
        # Sort frontier by relevance (you could implement scoring here)
        frontier_items = list(self.frontier.items())
        
        # Display options with numbers
        for i, (paper_id, metadata) in enumerate(frontier_items, 1):
            title = metadata.get('title', 'Unknown Title')
            year = metadata.get('publication_year', 'Unknown')
            concepts = metadata.get('concepts', [])
            concept_str = ', '.join(concepts[:3]) if concepts else 'No concepts'
            
            print(f"   {i:2d}. {title}")
            print(f"       Year: {year} | Concepts: {concept_str}")
            print(f"       ID: {paper_id}")
            print()
        
        # Get player choice
        while True:
            try:
                print("Choose a paper to expand:")
                print("  - Enter number (1-{})".format(len(frontier_items)))
                print("  - Enter 'q' to quit")
                print("  - Enter 'h' for help")
                
                choice = input("\n> ").strip().lower()
                
                if choice == 'q':
                    return None
                elif choice == 'h':
                    self._display_help()
                    continue
                else:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(frontier_items):
                        return frontier_items[choice_num - 1][0]  # Return paper_id
                    else:
                        print(f"❌ Please enter a number between 1 and {len(frontier_items)}")
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
    
    def _display_help(self):
        """Display help information for the player."""
        print("\n" + "="*60)
        print("📚 HELP - HOW TO PLAY")
        print("="*60)
        print("GOAL: Find the shortest path from START to END paper")
        print()
        print("RULES:")
        print("• You can expand one paper at a time")
        print("• Expanding a paper shows all papers it cites or is cited by")
        print("• If the END paper appears in the citations, you win!")
        print("• Try to find the shortest path possible")
        print()
        print("STRATEGY TIPS:")
        print("• Look for papers with relevant concepts/topics")
        print("• Consider publication years (newer papers cite older ones)")
        print("• Papers in similar research areas are more likely to be connected")
        print("="*60)
    
    def _display_final_results(self, ground_truth_path=None):
        """Display final game results."""
        print("\n" + "="*60)
        print("🏁 FINAL RESULTS")
        print("="*60)
        
        if self.graph.agent_path and len(self.graph.agent_path) > 1:
            path_length = len(self.graph.agent_path) - 1
            print(f"✅ Your path length: {path_length} steps")
            
            if ground_truth_path:
                optimal_length = len(ground_truth_path) - 1
                print(f"🎯 Optimal path length: {optimal_length} steps")
                
                if path_length == optimal_length:
                    print("🏆 PERFECT! You found the optimal path!")
                elif path_length <= optimal_length * 1.5:
                    print("🥈 Great job! Very close to optimal.")
                else:
                    print("🥉 Good effort! There's room for improvement.")
                    
                efficiency = optimal_length / path_length if path_length > 0 else 0
                print(f"📊 Efficiency: {efficiency:.2%}")
        else:
            print("❌ No path found")
        
        print("\n📁 Game data saved to: output/reference_graph.json")
        print("🎨 You can visualize your path using the visualization tools")