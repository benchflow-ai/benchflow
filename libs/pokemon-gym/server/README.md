# Pokemon Evaluator Server

This module provides a FastAPI server that exposes the Pokemon emulator functionality through a RESTful API. The server allows agents to interact with the Pokemon Red game remotely and provides evaluation and state management capabilities.

## Overview

The server provides the following functionality:
- Initialize a Pokemon game environment
- Take actions (press buttons, wait)
- Retrieve game state information
- Check environment status
- Stop the environment
- Save and load game states
- Manage and continue gameplay sessions
- Evaluate gameplay progress and performance

## Usage

### Starting the Server

Start the server using the provided script:

```bash
python -m server.evaluator_server --host 0.0.0.0 --port 8080 --rom Pokemon_Red.gb
```

Command-line arguments:
- `--host`: Host address to bind the server to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8080)
- `--rom`: Path to the Pokemon ROM file (default: Pokemon_Red.gb)
- `--log-file`: Custom CSV filename for logging (optional)

## API Endpoints

### Initialize Environment

```http
POST /initialize
```

Request body:
```json
{
  "headless": true,
  "sound": false,
  "load_state_file": "path/to/state.state",
  "load_autosave": false,
  "session_id": "session_20250404_180209"
}
```

All parameters except `headless` are optional:
- `headless`: Run without display (default: true)
- `sound`: Enable sound (requires non-headless mode, default: false)
- `load_state_file`: Path to a saved state file to load
- `load_autosave`: Whether to load the latest autosave
- `session_id`: Session ID to continue a previous session

Response:
```json
{
  "memory_info": { /* Memory data */ },
  "screenshot_base64": "base64_encoded_image",
  "location": "PALLET TOWN",
  "coordinates": [5, 6],
  "party": [ /* Pokemon party info */ ],
  "score": 25.5,
  "collision_map": "text representation of collision map",
  "valid_moves": ["up", "down", "left", "right"],
  "step_number": 0,
  "execution_time": 0.0
}
```

### Take Action

```http
POST /action
```

Request body (press buttons):
```json
{
  "action_type": "press_key",
  "keys": ["a"],
  "wait": true
}
```

Request body (wait):
```json
{
  "action_type": "wait",
  "frames": 60
}
```

Response: Same format as the initialize endpoint.

### Get Status

```http
GET /status
```

Response:
```json
{
  "status": "running",
  "steps_taken": 10,
  "session_id": "session_20250404_180209",
  "session_runtime": 125.4
}
```

### Stop Environment

```http
POST /stop
```

The stop endpoint will save the final state before stopping.

Response:
```json
{
  "status": "stopped",
  "session_id": "session_20250404_180209",
  "final_state_path": "gameplay_sessions/session_20250404_180209/final_state.state",
  "session_duration": 256.7
}
```

### Save State

```http
POST /save_state
```

Request body:
```json
{
  "filename": "custom_save.state"
}
```

If no filename is provided, a default name will be used.

Response:
```json
{
  "status": "success",
  "state_path": "gameplay_sessions/session_20250404_180209/custom_save.state"
}
```

### Load State

```http
POST /load_state
```

Request body:
```json
{
  "filename": "custom_save.state"
}
```

Response: Same format as the initialize endpoint.

### Get Evaluation

```http
GET /evaluate
```

Response:
```json
{
  "total_score": 45.5,
  "pokemon_caught": ["BULBASAUR", "PIDGEY", "RATTATA"],
  "badges_obtained": ["BOULDER"],
  "locations_visited": ["PALLET TOWN", "VIRIDIAN CITY", "PEWTER CITY"],
  "milestone_events": [
    {"name": "Start game", "points": 5.0, "achieved": true},
    {"name": "Get first Pokemon", "points": 10.0, "achieved": true}
  ]
}
```

## Session Management

The server automatically creates a unique session directory for each gameplay session. These are stored in the `gameplay_sessions` directory with names like `session_20250404_180209`.

Each session directory contains:
- `gameplay_data.csv`: Detailed record of each step, including state information
- `evaluation_summary.txt`: Summary of scores and achievements
- `images/`: Screenshots captured at each step 
- `autosave.state`: Automatically saved state (updated every 50 steps)
- `final_state.state`: Final state when the session ends
- `timeout_state.state`: State saved if the session times out

## Automatic State Saving

The server implements several automatic state saving mechanisms:

1. **Regular Autosaves**: The state is saved every 50 steps to `autosave.state`
2. **Final State**: When the session is stopped normally, the state is saved to `final_state.state`
3. **Timeout State**: If a session times out (exceeds 4 hours), the state is saved to `timeout_state.state`

## Evaluation System

The server includes a built-in evaluation system that scores gameplay based on:

- **Pokemon Collection**: Points for each unique Pokemon caught
- **Badges Earned**: Points for each gym badge obtained 
- **Locations Visited**: Points for discovering new areas
- **Milestone Events**: Points for key game events (getting first Pokemon, defeating rivals, etc.)

The evaluation state persists across saved/loaded states and continued sessions, allowing progress tracking across multiple play sessions.

## Client Usage Example

Here's a simple example of how to interact with the server using Python requests:

```python
import requests
import json
import base64
import time
from PIL import Image
import io

class PokemonClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.session = requests.Session()
        self.current_state = None
        
    def initialize(self, headless=True, sound=False, load_state_file=None, 
                  load_autosave=False, session_id=None):
        """Initialize the environment"""
        params = {
            "headless": headless,
            "sound": sound
        }
        
        # Add optional parameters if provided
        if load_state_file:
            params["load_state_file"] = load_state_file
        if load_autosave:
            params["load_autosave"] = load_autosave 
        if session_id:
            params["session_id"] = session_id
            
        response = self.session.post(
            f"{self.server_url}/initialize",
            headers={"Content-Type": "application/json"},
            json=params
        )
        self.current_state = response.json()
        return self.current_state
        
    def press_key(self, keys, wait=True):
        """Press one or more keys"""
        response = self.session.post(
            f"{self.server_url}/action",
            headers={"Content-Type": "application/json"},
            json={
                "action_type": "press_key",
                "keys": keys,
                "wait": wait
            }
        )
        self.current_state = response.json()
        return self.current_state
        
    def wait(self, frames=30):
        """Wait for a number of frames"""
        response = self.session.post(
            f"{self.server_url}/action",
            headers={"Content-Type": "application/json"},
            json={
                "action_type": "wait",
                "frames": frames
            }
        )
        self.current_state = response.json()
        return self.current_state
        
    def save_state(self, filename=None):
        """Save the current state"""
        params = {}
        if filename:
            params["filename"] = filename
            
        response = self.session.post(
            f"{self.server_url}/save_state",
            headers={"Content-Type": "application/json"},
            json=params
        )
        return response.json()
        
    def load_state(self, filename):
        """Load a saved state"""
        response = self.session.post(
            f"{self.server_url}/load_state",
            headers={"Content-Type": "application/json"},
            json={"filename": filename}
        )
        self.current_state = response.json()
        return self.current_state
        
    def get_evaluation(self):
        """Get the current evaluation summary"""
        response = self.session.get(f"{self.server_url}/evaluate")
        return response.json()
        
    def stop(self):
        """Stop the environment"""
        response = self.session.post(f"{self.server_url}/stop")
        return response.json()
        
    def get_screen_image(self):
        """Convert the current screenshot to a PIL Image"""
        if not self.current_state or "screenshot_base64" not in self.current_state:
            return None
            
        img_data = base64.b64decode(self.current_state["screenshot_base64"])
        return Image.open(io.BytesIO(img_data))
        
    def display_info(self):
        """Display current game information"""
        if not self.current_state:
            return
            
        print(f"Location: {self.current_state.get('location', 'Unknown')}")
        print(f"Coordinates: {self.current_state.get('coordinates', [0, 0])}")
        print(f"Party size: {len(self.current_state.get('party', []))}")
        print(f"Score: {self.current_state.get('score', 0)}")

# Example usage
if __name__ == "__main__":
    client = PokemonClient()
    
    # Initialize with a previous session
    client.initialize(session_id="session_20250404_180209")
    
    # Or initialize with a specific state file
    # client.initialize(load_state_file="gameplay_sessions/my_save.state")
    
    # Take some actions
    client.display_info()
    client.press_key(["a"])
    client.wait(30)
    
    # Save the current state
    save_result = client.save_state("my_checkpoint.state")
    print(f"State saved to: {save_result['state_path']}")
    
    # Get evaluation
    eval_result = client.get_evaluation()
    print(f"Current score: {eval_result['total_score']}")
    
    # Stop when done
    client.stop() 