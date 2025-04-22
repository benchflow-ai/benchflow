# PokemonGym Agents

This directory contains agents for interacting with the PokemonGym framework:

- **Demo Agent**: An AI agent powered by Claude that can play Pokemon Red autonomously
- **Human Agent**: A pygame interface that allows humans to play Pokemon Red

## Installation

Make sure you have installed the PokemonGym project and its dependencies:
```bash
pip install -r requirements.txt
```

For the Human Agent, ensure pygame is installed:
```bash
pip install pygame
```

For the Demo Agent, set up your API keys:
```bash
export ANTHROPIC_API_KEY=your_anthropic_key_here  # For Claude
export OPENAI_API_KEY=your_openai_key_here        # For GPT-4o
export OPENROUTER_API_KEY=your_openrouter_key_here  # For Llama
export GOOGLE_API_KEY=your_google_key_here        # For Gemini
```

## Demo Agent

The Demo Agent is an AI agent that uses various LLMs (Claude, GPT-4o, Gemini, or Llama) to make decisions based on the game state, demonstrating how to build an agent that can:

1. Connect to the PokemonGym server
2. Observe the game state
3. Make decisions based on the observed state
4. Take actions to play the game
5. Track performance metrics and scores

### Running the Demo Agent

Start the Demo Agent with:

```bash
python agents/demo_agent.py
```

Command-line arguments:
- `--server`: URL of the PokemonGym server (default: http://localhost:8080)
- `--steps`: Number of steps to run (default: 1000000)
- `--headless`: Run without displaying the game (default: False)
- `--sound`: Enable sound (default: False)
- `--provider`: AI provider to use (claude, openai, gemini, openrouter)
- `--model`: Model to use (default depends on provider)
- `--temperature`: Temperature for model generation (default: 1.0)
- `--max-tokens`: Max tokens for response (default: 4000)
- `--log-file`: File to save agent logs (default: agent_log.jsonl)
- `--load-state`: Path to a saved state file to load
- `--load-autosave`: Load the latest autosave
- `--session`: Session ID to continue a previous session

### Demo Agent Architecture

The Demo Agent has the following components:

1. **AIServerAgent**: Main class that:
   - Communicates with the PokemonGym server
   - Manages the game state
   - Calls the language model for decision making
   - Executes actions based on the model's decisions

2. **Decision Loop**:
   - Observe the current state
   - Send state information to the language model
   - Parse the model's response
   - Execute the chosen action
   - Repeat

### Implementation Details

#### Initialization

```python
def initialize(self, headless: bool = False, sound: bool = False,
              load_state_file: str = None, load_autosave: bool = False,
              session_id: str = None):
    """Initialize the environment."""
    # Log when loading a state file or using session ID
    if load_state_file:
        logging.info(f"Loading state file: {load_state_file}")
    if session_id:
        logging.info(f"Using session ID: {session_id}")
        
    # Prepare initialization parameters
    init_params = {
        "headless": headless,
        "sound": sound,
        "load_state_file": load_state_file,
        "load_autosave": load_autosave,
        "session_id": session_id
    }
    
    response = self.session.post(
        f"{self.server_url}/initialize",
        headers={"Content-Type": "application/json"},
        json=init_params
    )
    self.current_state = response.json()
```

#### Execution Loop

```python
def run(self, max_steps=1000):
    """Run the agent for a specified number of steps."""
    self.initialize(headless=False, sound=True)
    
    for step in range(max_steps):
        # Get the current state
        state = self.current_state
        
        # Log information about the current state
        logging.info(f"Step {step}: Location: {state['location']}, "
                    f"Coords: {state['coordinates']}, "
                    f"Party size: {len(state.get('party', []))}, "
                    f"Score: {state.get('score', 0)}")
        
        # Decide what action to take
        action, params = self.decide_action(state)
        
        # Execute the action
        self.take_action(action, **params)
        
        # Check if we should continue
        if not self.running:
            break
```

## Human Agent

The Human Agent provides a pygame-based interface that allows humans to play Pokemon Red through the PokemonGym server, useful for:

1. Establishing human baseline performance for comparison with AI agents
2. Gathering human gameplay data for training imitation learning agents
3. Verifying the environment works correctly with human input
4. Tracking progress and scores across multiple sessions

### Running the Human Agent

Start the Human Agent with:

```bash
python agents/human_agent.py
```

Command-line arguments:
- `--server`: URL of the PokemonGym server (default: http://localhost:8080)
- `--sound`: Enable sound (optional)
- `--load-state`: Path to a saved state file to load
- `--load-autosave`: Load the latest autosave
- `--session`: Session ID to continue a previous session

### Human Agent Controls

The Human Agent maps keyboard keys to Game Boy buttons:

| Keyboard Key | Function               |
|--------------|------------------------|
| Arrow Up     | D-pad Up               |
| Arrow Down   | D-pad Down             |
| Arrow Left   | D-pad Left             |
| Arrow Right  | D-pad Right            |
| Z            | A button               |
| X            | B button               |
| Enter        | Start button           |
| Right Shift  | Select button          |
| Space        | Wait (30 frames)       |
| F5           | Save state             |
| F7           | Load saved state       |
| Escape       | Quit                   |

### Implementation Details

#### Initialization

```python
def initialize(self, headless: bool = False, sound: bool = False,
              load_state_file: str = None, load_autosave: bool = False,
              session_id: str = None):
    response = self.session.post(
        f"{self.server_url}/initialize",
        headers={"Content-Type": "application/json"},
        json={
            "headless": headless,
            "sound": sound,
            "load_state_file": load_state_file,
            "load_autosave": load_autosave,
            "session_id": session_id
        }
    )
    self.current_state = response.json()
    # Update display with initial state
    self.update_display(self.current_state)
```

#### Display Updates

```python
def update_display(self, state):
    # Decode base64 image
    image_data = base64.b64decode(state['screenshot_base64'])
    image = Image.open(io.BytesIO(image_data))
    
    # Resize and display image
    image = image.resize((self.screen_width, self.screen_height), Image.NEAREST)
    pygame_image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    self.screen.blit(pygame_image, (0, 0))
    
    # Display game information including score
    info_text = f"Location: {state['location']} | Coords: {state['coordinates']} | Step: {self.step_count} | Score: {self.score:.1f}"
    
    # Display controls with save/load options
    controls_text = "Controls: Arrows = Move | Z = A | X = B | Enter = Start | R-Shift = Select | Space = Wait | F5 = Save | F7 = Load"
    
    # Render and display text
    # ...
    
    pygame.display.flip()
```

## State Management

Both agents support several state management features:

### Saving Game States

The Demo Agent automatically saves states at regular intervals.
The Human Agent allows saving the current state by pressing F5.

### Loading Game States

You can load a previously saved state by:

```bash
# Using a specific state file
python agents/demo_agent.py --load-state gameplay_sessions/session_20250404_180209/final_state.state
python agents/human_agent.py --load-state gameplay_sessions/session_20250404_180209/final_state.state

# Loading the latest autosave
python agents/demo_agent.py --load-autosave
python agents/human_agent.py --load-autosave
```

### Session Continuation

To continue a previous gameplay session:

```bash
# Demo agent with session
python agents/demo_agent.py --session session_20250404_180209

# Human agent with session
python agents/human_agent.py --session session_20250404_180209
```

When continuing a session:
- The system will automatically load the final state
- All game progress and scores will be preserved
- New gameplay data will be appended to the existing record

## Example Usage

Here's how to create your own agent based on these examples:

```python
# Import the appropriate agent class
from agents.demo_agent import AIServerAgent
# or
from agents.human_agent import HumanAgent

# Create and initialize the agent
agent = AIServerAgent(server_url="http://localhost:8080")
agent.initialize(
    headless=False,
    sound=True,
    load_state_file="gameplay_sessions/session_20250404_180209/checkpoint_50.state"
)

# Run the agent
agent.run(max_steps=500)
```

## Customization

You can customize the agents in several ways:

1. **Modify the AI model**: Try different LLM providers and models
2. **Implement specialized behavior**: Add specific strategies for different game phases
3. **Add custom controls**: Modify the Human Agent controls for your preferences
4. **Extend with visualizations**: Add more detailed game state displays

## Troubleshooting

- If the AI doesn't make good decisions, try improving the state representation
- If the display is blank, ensure the server is returning valid screenshots
- If actions aren't being executed correctly, check the server connection
- If the game seems slow, reduce the server load or try running on a more powerful machine 