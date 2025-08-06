# persistence.py
# Simple file-based persistence for SciPathBench web interface

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock

class RunStorage:
    """Simple file-based storage for run data and leaderboard."""
    
    def __init__(self, storage_file: str = "output/web_runs.json"):
        self.storage_file = Path(storage_file)
        self.lock = Lock()
        self._ensure_storage_file()
    
    def _ensure_storage_file(self):
        """Ensure the storage file and directory exist."""
        self.storage_file.parent.mkdir(exist_ok=True)
        
        if not self.storage_file.exists():
            self._write_data({
                "runs": [],
                "metadata": {
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0"
                }
            })
    
    def _read_data(self) -> Dict:
        """Read data from storage file."""
        try:
            with open(self.storage_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Error reading storage file: {e}")
            return {"runs": [], "metadata": {}}
    
    def _write_data(self, data: Dict):
        """Write data to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error writing to storage file: {e}")
    
    def add_run(self, run_data: Dict) -> bool:
        """Add a new run to storage."""
        with self.lock:
            try:
                data = self._read_data()
                
                # Add timestamp if not present
                if "timestamp" not in run_data:
                    run_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Add unique ID
                run_data["id"] = f"run_{int(time.time() * 1000)}"
                
                data["runs"].append(run_data)
                
                # Keep only last 1000 runs to prevent file from growing too large
                if len(data["runs"]) > 1000:
                    data["runs"] = data["runs"][-1000:]
                
                self._write_data(data)
                logging.info(f"Added run to storage: {run_data.get('id')}")
                return True
                
            except Exception as e:
                logging.error(f"Error adding run to storage: {e}")
                return False
    
    def get_all_runs(self) -> List[Dict]:
        """Get all runs from storage."""
        with self.lock:
            data = self._read_data()
            return data.get("runs", [])
    
    def get_leaderboard_data(self, limit: Optional[int] = None) -> List[Dict]:
        """Get runs formatted for leaderboard display."""
        runs = self.get_all_runs()
        
        # Sort by timestamp (most recent first)
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if limit:
            runs = runs[:limit]
        
        return runs
    
    def get_runs_by_type(self, run_type: str) -> List[Dict]:
        """Get runs filtered by type (e.g., 'llm', 'human')."""
        runs = self.get_all_runs()
        return [run for run in runs if run.get("type") == run_type]
    
    def get_runs_by_model(self, model: str) -> List[Dict]:
        """Get runs filtered by model name."""
        runs = self.get_all_runs()
        return [run for run in runs if run.get("model") == model]
    
    def get_statistics(self) -> Dict:
        """Get overall statistics from stored runs."""
        runs = self.get_all_runs()
        
        if not runs:
            return {
                "total_runs": 0,
                "success_rate": 0,
                "average_optimality": 0,
                "average_runtime": 0,
                "models": [],
                "run_types": []
            }
        
        successful_runs = [r for r in runs if r.get("success", 0) == 1]
        success_rate = len(successful_runs) / len(runs) if runs else 0
        
        # Calculate average optimality for successful runs only
        optimalities = [r.get("optimality", 0) for r in successful_runs if r.get("optimality")]
        avg_optimality = sum(optimalities) / len(optimalities) if optimalities else 0
        
        # Calculate average runtime
        runtimes = [r.get("runtime", 0) for r in runs if r.get("runtime")]
        avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
        
        # Get unique models and run types
        models = list(set(r.get("model", "Unknown") for r in runs))
        run_types = list(set(r.get("type", "Unknown") for r in runs))
        
        return {
            "total_runs": len(runs),
            "success_rate": success_rate,
            "average_optimality": avg_optimality,
            "average_runtime": avg_runtime,
            "models": sorted(models),
            "run_types": sorted(run_types),
            "successful_runs": len(successful_runs)
        }
    
    def cleanup_old_runs(self, days: int = 30):
        """Remove runs older than specified days."""
        with self.lock:
            try:
                data = self._read_data()
                current_time = time.time()
                cutoff_time = current_time - (days * 24 * 60 * 60)
                
                # Filter out old runs
                filtered_runs = []
                for run in data.get("runs", []):
                    run_timestamp = run.get("timestamp", "")
                    try:
                        run_time = time.mktime(time.strptime(run_timestamp, "%Y-%m-%d %H:%M:%S"))
                        if run_time >= cutoff_time:
                            filtered_runs.append(run)
                    except (ValueError, TypeError):
                        # Keep runs with invalid timestamps
                        filtered_runs.append(run)
                
                removed_count = len(data.get("runs", [])) - len(filtered_runs)
                data["runs"] = filtered_runs
                
                self._write_data(data)
                logging.info(f"Cleaned up {removed_count} old runs")
                return removed_count
                
            except Exception as e:
                logging.error(f"Error cleaning up old runs: {e}")
                return 0

# Global storage instance
storage = RunStorage()

def format_run_for_storage(run_type: str, model: str, success: bool, 
                          path_length: int, optimal_length: int, 
                          runtime: float, **kwargs) -> Dict:
    """Format run data for storage."""
    optimality = optimal_length / path_length if path_length > 0 and optimal_length > 0 else 0
    
    return {
        "type": run_type,
        "model": model,
        "success": 1 if success else 0,
        "path_length": path_length,
        "optimal_length": optimal_length,
        "optimality": optimality,
        "runtime": runtime,
        **kwargs  # Additional fields like task_id, etc.
    }