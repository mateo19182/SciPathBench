# web_server.py
# FastAPI web server for SciPathBench interactive interface

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.openalex_client import OpenAlexClient
from src.llm_agent import LLMAgent
from src.web_human_agent import WebHumanAgent
from src.eval import EvaluationHarness
from src.utils import setup_logging
from src.persistence import storage, format_run_for_storage
import config

app = FastAPI(title="SciPathBench Web Interface")

# Setup static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
static_path.mkdir(exist_ok=True)
templates_path.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# Global state
active_connections: Dict[str, WebSocket] = {}
current_runs: Dict[str, Dict] = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/interactive", response_class=HTMLResponse)
async def interactive_mode(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})

@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_view(request: Request):
    return templates.TemplateResponse("leaderboard.html", {"request": request})

@app.get("/live", response_class=HTMLResponse)
async def live_runs(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await handle_websocket_message(client_id, message)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        # Clean up any active session for this client
        if client_id in current_runs:
            session = current_runs[client_id]
            if hasattr(session.get("agent"), "cleanup"):
                session["agent"].cleanup()
            del current_runs[client_id]
            logging.info(f"Cleaned up session for disconnected client {client_id}")

async def handle_websocket_message(client_id: str, message: dict):
    message_type = message.get("type")
    
    if message_type == "start_interactive":
        await start_interactive_session(client_id, message.get("data", {}))
    elif message_type == "interactive_choice":
        await handle_interactive_choice(client_id, message.get("data", {}))
    elif message_type == "start_llm_run":
        await start_llm_run(client_id, message.get("data", {}))
    elif message_type == "get_leaderboard":
        await send_leaderboard(client_id)

async def start_interactive_session(client_id: str, data: dict):
    try:
        logging.info(f"Starting interactive session for client {client_id}")
        
        # Get a random task from benchmark data
        with open(config.BENCHMARK_DATA_FILE, "r") as f:
            all_pairs = json.load(f)
        
        import random
        task = random.choice(all_pairs)
        logging.info(f"Selected task: {task['start_id']} -> {task['end_id']}")
        
        # Initialize web human agent with message callback
        api_client = OpenAlexClient()
        
        async def message_callback(message):
            try:
                await manager.send_message(client_id, message)
            except Exception as e:
                logging.error(f"Error sending message to client {client_id}: {e}")
        
        agent = WebHumanAgent(api_client=api_client, message_callback=message_callback)
        
        # Store session data
        current_runs[client_id] = {
            "type": "interactive",
            "agent": agent,
            "task": task,
            "start_time": time.time(),
            "status": "active"
        }
        
        # Initialize the game
        logging.info(f"Initializing game for client {client_id}")
        success = await agent.initialize_game(
            task["start_id"], 
            task["end_id"], 
            max_turns=config.AGENT_MAX_TURNS,
            ground_truth_path=task["path_ids"]
        )
        
        if not success:
            logging.error(f"Failed to initialize game for client {client_id}")
            await manager.send_message(client_id, {
                "type": "error",
                "data": {"message": "Failed to initialize game"}
            })
        else:
            logging.info(f"Game successfully initialized for client {client_id}")
        
    except Exception as e:
        logging.error(f"Error starting interactive session for client {client_id}: {e}", exc_info=True)
        await manager.send_message(client_id, {
            "type": "error",
            "data": {"message": f"Failed to start session: {str(e)}"}
        })

async def handle_interactive_choice(client_id: str, data: dict):
    if client_id not in current_runs:
        await manager.send_message(client_id, {
            "type": "error",
            "data": {"message": "No active session found"}
        })
        return
        
    session = current_runs[client_id]
    agent = session["agent"]
    
    try:
        # Debug logging
        logging.info(f"Handling interactive choice for client {client_id}, data: {data}")
        
        paper_id = data.get("paper_id")
        if not paper_id:
            await manager.send_message(client_id, {
                "type": "error",
                "data": {"message": "No paper ID provided"}
            })
            return
        
        # Process the choice through the web agent
        result = await agent.make_choice(paper_id)
        
        # Debug logging
        logging.info(f"Agent choice result: {result}")
        
        if not result.get("success", False):
            # Send error message
            await manager.send_message(client_id, {
                "type": "error",
                "data": {"message": result.get("error", "Unknown error")}
            })
            return
        
        # If game is complete, update session status and add to leaderboard
        if result.get("game_complete"):
            session["status"] = "completed"
            session["result"] = result
            
            # Clean up agent resources
            if hasattr(session["agent"], "cleanup"):
                session["agent"].cleanup()
            
            # Save run to persistent storage
            task = session["task"]
            path_length = 0
            if result.get("path") and isinstance(result["path"], list):
                path_length = len(result["path"]) - 1
            
            run_data = format_run_for_storage(
                run_type="human",
                model="Human Player",
                success=result.get("won", False),
                path_length=path_length,
                optimal_length=len(task["path_ids"]) - 1,
                runtime=time.time() - session["start_time"],
                turns_used=result.get("turns_used", 0),
                task_start=task["start_id"],
                task_end=task["end_id"]
            )
            storage.add_run(run_data)
            
            # Send game complete message
            await manager.send_message(client_id, {
                "type": "game_complete",
                "data": result
            })
        else:
            # Send turn complete message
            await manager.send_message(client_id, {
                "type": "turn_complete",
                "data": result
            })
        
    except Exception as e:
        logging.error(f"Error handling interactive choice: {e}", exc_info=True)
        await manager.send_message(client_id, {
            "type": "error", 
            "data": {"message": str(e)}
        })

async def start_llm_run(client_id: str, data: dict):
    try:
        # Get a random task
        with open(config.BENCHMARK_DATA_FILE, "r") as f:
            all_pairs = json.load(f)
        
        import random
        task = random.choice(all_pairs)
        
        # Initialize LLM agent
        api_client = OpenAlexClient()
        agent = LLMAgent(api_client=api_client, llm_provider=config.LLM_PROVIDER_MODEL)
        
        # Store run data
        run_id = f"llm_{int(time.time())}"
        current_runs[run_id] = {
            "type": "llm",
            "agent": agent,
            "task": task,
            "start_time": time.time(),
            "status": "running",
            "client_id": client_id
        }
        
        # Broadcast that a new run started
        await manager.broadcast({
            "type": "llm_run_started",
            "data": {
                "run_id": run_id,
                "model": config.LLM_PROVIDER_MODEL,
                "start_paper": task["start_id"],
                "end_paper": task["end_id"]
            }
        })
        
        # Run the agent in background
        asyncio.create_task(run_llm_agent_background(run_id))
        
    except Exception as e:
        logging.error(f"Error starting LLM run: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "data": {"message": str(e)}
        })

async def run_llm_agent_background(run_id: str):
    try:
        session = current_runs[run_id]
        agent = session["agent"]
        task = session["task"]
        
        # Run the agent
        agent_path, full_path = agent.find_path(
            task["start_id"], 
            task["end_id"], 
            max_turns=config.AGENT_MAX_TURNS, 
            ground_truth_path=task["path_ids"]
        )
        
        # Evaluate results
        evaluator = EvaluationHarness(
            ground_truth_path=task["path_ids"],
            agent_path=agent_path
        )
        scorecard = evaluator.run_evaluation()
        
        # Update session
        session["status"] = "completed"
        session["result"] = {
            "agent_path": agent_path,
            "scorecard": scorecard,
            "end_time": time.time()
        }
        
        # Save run to persistent storage
        run_data = format_run_for_storage(
            run_type="llm",
            model=config.LLM_PROVIDER_MODEL,
            success=bool(agent_path),
            path_length=len(agent_path) - 1 if agent_path else 0,
            optimal_length=len(task["path_ids"]) - 1,
            runtime=time.time() - session["start_time"],
            precision=scorecard.get("precision", 0),
            recall=scorecard.get("recall", 0),
            reasoning_faithfulness=scorecard.get("reasoning_faithfulness", 0),
            task_start=task["start_id"],
            task_end=task["end_id"]
        )
        storage.add_run(run_data)
        
        # Broadcast completion
        await manager.broadcast({
            "type": "llm_run_completed",
            "data": {
                "run_id": run_id,
                "result": session["result"],
                "run_data": run_data
            }
        })
        
    except Exception as e:
        logging.error(f"Error in LLM run background: {e}")
        session["status"] = "error"
        session["error"] = str(e)

async def send_leaderboard(client_id: str):
    leaderboard_data = storage.get_leaderboard_data(limit=100)  # Last 100 runs
    await manager.send_message(client_id, {
        "type": "leaderboard_data",
        "data": leaderboard_data
    })

@app.get("/api/leaderboard")
async def get_leaderboard():
    return {"data": storage.get_leaderboard_data(limit=100)}

@app.get("/api/statistics")
async def get_statistics():
    return storage.get_statistics()

@app.get("/api/runs")
async def get_runs(run_type: Optional[str] = None, model: Optional[str] = None, limit: int = 50):
    if run_type:
        runs = storage.get_runs_by_type(run_type)
    elif model:
        runs = storage.get_runs_by_model(model)
    else:
        runs = storage.get_all_runs()
    
    # Sort by timestamp and limit
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"data": runs[:limit]}

@app.get("/api/status")
async def get_status():
    active_runs = [
        {
            "run_id": run_id,
            "type": run_data["type"], 
            "status": run_data["status"],
            "start_time": run_data["start_time"]
        }
        for run_id, run_data in current_runs.items()
        if run_data["status"] in ["running", "active"]
    ]
    
    stats = storage.get_statistics()
    
    return {
        "active_connections": len(manager.active_connections),
        "active_runs": active_runs,
        "total_completed_runs": stats["total_runs"],
        "success_rate": f"{stats['success_rate'] * 100:.1f}%",
        "models": stats["models"]
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logging.info("SciPathBench Web Interface starting up...")
    
    # Load existing runs and display stats
    stats = storage.get_statistics()
    logging.info(f"Loaded {stats['total_runs']} existing runs from storage")
    logging.info(f"Success rate: {stats['success_rate']*100:.1f}%")
    logging.info(f"Available models: {', '.join(stats['models'])}")

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up on shutdown."""
    logging.info("SciPathBench Web Interface shutting down...")

if __name__ == "__main__":
    setup_logging(config.LOG_FILE)
    uvicorn.run(app, host="0.0.0.0", port=8001)