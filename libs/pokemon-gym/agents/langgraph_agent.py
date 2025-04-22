import argparse
import logging
import time
import os
import json
import datetime
import random
from typing import Dict, List, Any, Optional
from enum import Enum
import base64
import io
from PIL import Image

import requests

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# LLM provider imports
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.WARNING,  # Change to WARNING to reduce log volume
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)
# Add file handler
file_handler = logging.FileHandler(f"logs/pokemon_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setLevel(logging.WARNING)  # Set to WARNING to reduce log volume
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Define Action Types
class ActionType(str, Enum):
    PRESS_KEY = "press_key"
    WAIT = "wait"

# Define valid buttons
class Button(str, Enum):
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

# State model for the agent system
class PokemonAgentState(BaseModel):
    # Game state information
    game_state: Optional[Dict] = Field(default=None, description="Current game state information")
    
    # Memory components
    short_term_memory: List[Dict] = Field(default_factory=list, description="Recent events and observations (last ~20 steps)")
    
    # Task tracking
    current_task: Optional[str] = Field(default=None, description="The current task/goal the agent is working on")
    task_history: List[str] = Field(default_factory=list, description="History of completed tasks")
    
    # Map/navigation data
    known_locations: Dict[str, Any] = Field(default_factory=dict, description="Information about discovered locations")
    
    # Current action 
    action: Optional[Dict] = Field(default=None, description="Action to take")
    action_reasoning: Optional[str] = Field(default=None, description="Reasoning behind the action")
    
    # Current visual analysis
    current_visual_analysis: str = Field(default="", description="Visual analysis from the current step")
    
    # Execution tracking
    step_count: int = Field(default=0, description="Number of steps taken")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    consecutive_llm_errors: int = Field(default=0, description="Count of consecutive LLM call errors")
    
    # Trace for debugging
    trace: List[Dict] = Field(default_factory=list, description="Trace of agent actions and reasoning")
    
    def add_to_trace(self, action: str, content: Any) -> None:
        """Add an entry to the execution trace"""
        trace_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": self.step_count,
            "action": action,
            "content": content
        }
        self.trace.append(trace_entry)
        # Log trace entry
        logger.info(f"TRACE: Step {self.step_count}, Action: {action} - {content}")
    
    def add_to_short_term_memory(self, entry: Dict) -> None:
        """Add an entry to short-term memory with timestamp and complete game state information"""
        if not isinstance(entry, dict):
            entry = {"content": entry}
        
        entry["timestamp"] = datetime.datetime.now().isoformat()
        entry["step"] = self.step_count
        
        # Enrich with more detailed game state information if available
        if self.game_state:
            # Add complete game state snapshot
            entry["game_state"] = self.game_state.copy() if isinstance(self.game_state, dict) else self.game_state
            
            # Extract key game state elements for easier access
            entry["location"] = self.game_state.get("location", "Unknown")
            entry["coordinates"] = self.game_state.get("coordinates", [0, 0])
            
            # Add dialog information
            if "dialog" in self.game_state and self.game_state["dialog"]:
                entry["dialog"] = self.game_state["dialog"]
            
            # Add Pokemon party summary
            if "pokemons" in self.game_state and self.game_state["pokemons"]:
                party_summary = []
                for pokemon in self.game_state["pokemons"]:
                    hp = pokemon.get("hp", {})
                    hp_current = hp.get("current", 0)
                    hp_max = hp.get("max", 1)
                    
                    summary = {
                        "species": pokemon.get("species", "Unknown"),
                        "nickname": pokemon.get("nickname", ""),
                        "level": pokemon.get("level", 1),
                        "hp": f"{hp_current}/{hp_max}",
                        "status": pokemon.get("status", "Normal"),
                        "moves": pokemon.get("moves", [])
                    }
                    party_summary.append(summary)
                entry["party"] = party_summary
            
            # Add inventory summary
            if "inventory" in self.game_state and self.game_state["inventory"]:
                entry["inventory"] = self.game_state["inventory"]
                
            # Add player money and badges
            entry["money"] = self.game_state.get("money", 0)
            entry["badges"] = self.game_state.get("badges", [])
        
        # Add current thinking and decision if available
        if hasattr(self, "action_reasoning") and self.action_reasoning:
            entry["thinking"] = self.action_reasoning
            
        if hasattr(self, "action") and self.action:
            entry["decision"] = self.action
        
        # Add to short term memory, ensuring we keep the full 20 entries
        self.short_term_memory.append(entry)
        if len(self.short_term_memory) > 20:  # Keep last 20 entries
            self.short_term_memory = self.short_term_memory[-20:]
    
    def summarize_short_term_memory(self) -> str:
        """Create a summary of short-term memory"""
        if len(self.short_term_memory) == 0:
            return "No recent events."
            
        events = []
        for memory in self.short_term_memory[-10:]:  # Last 10 entries
            content = memory.get('content', '')
            if isinstance(content, str):
                events.append(content)
            elif isinstance(content, dict):
                events.append(str(content))
                
        return "Recent events: " + " ".join(events)
    
    def get_relevant_short_term_memories(self, query: str, max_results: int = 5) -> List[Dict]:
        """Get most relevant short-term memories for a query"""
        if not self.short_term_memory:
            return []
            
        # Simple keyword matching for relevance scoring
        relevant_memories = []
        query_terms = query.lower().split()
        
        for memory in self.short_term_memory:
            memory_text = str(memory.get('content', '')).lower()
            # Calculate relevance score
            score = sum(3 for term in query_terms if term in memory_text)
            
            if score > 0:
                relevant_memories.append((score, memory))
                
        # Sort by relevance score (descending) and take top results
        relevant_memories.sort(reverse=True, key=lambda x: x[0])
        return [memory for _, memory in relevant_memories[:max_results]]

# LLM Provider configuration and model selection
class LLMProvider:
    """Handles interaction with different LLM providers"""
    
    def __init__(
        self, 
        provider: str = "claude",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the appropriate client based on provider
        if self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
            self.model_name = model_name or "claude-3-7-sonnet-20250219"
            logger.info(f"Using Claude provider with model: {self.model_name}")
        
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name or "gpt-4o"
            logger.info(f"Using OpenAI provider with model: {self.model_name}")
        
        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.model_name = model_name or "meta-llama/llama-4-maverick"
            logger.info(f"Using OpenRouter provider with model: {self.model_name}")
        
        elif self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.model_name = model_name or "gemini-2.5-pro-preview-03-25"
            self.generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
            }
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            logger.info(f"Using Gemini provider with model: {self.model_name}")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Choose 'claude', 'openai', 'openrouter', or 'gemini'")
    
    def get_llm(self) -> Runnable:
        """Returns a LangChain-compatible LLM interface"""
        # Create the appropriate LLM based on provider
        if self.provider == "claude":
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        
        elif self.provider in ["openai", "openrouter"]:
            # For OpenRouter, we use OpenAI's client with a different base URL
            base_url = "https://openrouter.ai/api/v1" if self.provider == "openrouter" else None
            
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=os.getenv("OPENAI_API_KEY" if self.provider == "openai" else "OPENROUTER_API_KEY"),
                base_url=base_url
            )
        
        elif self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                convert_system_message_to_human=True  # Gemini doesn't natively support system messages
            )
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

# Server interface for Pokemon environment
class PokemonServerInterface:
    """Interface for communicating with the Pokemon Red environment server"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        """
        Initialize the server interface
        
        Args:
            server_url: URL of the evaluation server
        """
        self.server_url = server_url
        self.session = requests.Session()
        self.initialized = False
        self.current_state = None
        self.running = False
        self.step_count = 0
    
    def initialize(
        self, 
        headless: bool = True, 
        sound: bool = False,
        load_state_file: Optional[str] = None, 
        load_autosave: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initialize the game environment
        
        Args:
            headless: Whether to run without a GUI
            sound: Whether to enable sound
            load_state_file: Optional path to a saved state file to load
            load_autosave: Whether to load the latest autosave
            session_id: Optional session ID to continue an existing session
            
        Returns:
            Initial game state
        """
        try:
            # Prepare initialization parameters
            init_params = {
                "headless": headless,
                "sound": sound,
                "load_autosave": load_autosave
            }
            
            # Add load_state_file if provided
            if load_state_file:
                init_params["load_state_file"] = load_state_file
                logger.info(f"Will try to load state from {load_state_file}")
            
            # Add session_id if provided
            if session_id:
                init_params["session_id"] = session_id
                logger.info(f"Will continue existing session: {session_id}")
            
            # Send initialize request
            response = self.session.post(
                f"{self.server_url}/initialize",
                headers={"Content-Type": "application/json"},
                json=init_params
            )
            
            response.raise_for_status()
            self.current_state = response.json()
            
            # Set initialization flag
            self.initialized = True
            self.running = True
            
            logger.info(f"Initialization successful, location: {self.current_state['location']}")
            
            return self.current_state
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Initialization error: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Server response: {e.response.text}")
            raise
    
    def take_action(self, action_type: str, **kwargs) -> Dict[str, Any]:
        """
        Send an action request to the server
        
        Args:
            action_type: Action type ("press_key" or "wait")
            **kwargs: Action parameters
                press_key: keys
                wait: frames
        
        Returns:
            Game state after executing the action
        """
        if not self.initialized:
            raise RuntimeError("Environment not initialized, please call initialize() first")
        
        try:
            # Prepare request data
            request_data = {"action_type": action_type, **kwargs}
            
            # Send request
            response = self.session.post(
                f"{self.server_url}/action",
                headers={"Content-Type": "application/json"},
                json=request_data
            )
            
            response.raise_for_status()
            self.current_state = response.json()
            self.step_count += 1
            
            return self.current_state
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Action execution error: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Server response: {e.response.text}")
            raise
    
    def press_key(self, button: str) -> Dict[str, Any]:
        """
        Press a button on the Game Boy
        
        Args:
            button: The button to press (a, b, start, select, up, down, left, right)
            
        Returns:
            Game state after pressing the button
        """
        return self.take_action("press_key", keys=[button])
    
    def wait(self, frames: int) -> Dict[str, Any]:
        """
        Wait for a specified number of frames
        
        Args:
            frames: Number of frames to wait
            
        Returns:
            Game state after waiting
        """
        return self.take_action("wait", frames=frames)
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop the environment
        
        Returns:
            Response from the server
        """
        if not self.initialized:
            return {"status": "not_initialized"}
        
        try:
            response = self.session.post(f"{self.server_url}/stop")
            response.raise_for_status()
            self.initialized = False
            self.running = False
            logger.info("Environment stopped")
            return response.json()
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            self.running = False
            return {"status": "error", "message": str(e)} 

# Single Agent system using LangGraph
class PokemonSingleAgent:
    """
    Single agent system for playing Pokemon Red that combines observation, decision making, 
    and memory management into one unified agent.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        provider: str = "claude",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        log_dir: str = "logs",
        session_id: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the Pokemon Single Agent system
        
        Args:
            server_url: URL of the Pokemon game server
            provider: LLM provider ("claude", "openai", "gemini", "openrouter")
            model_name: Specific model name to use (if None, uses provider default)
            temperature: Temperature for LLM generation
            log_dir: Directory for logs
            session_id: Optional session ID for resuming
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Set up logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(numeric_level)
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_dir = log_dir
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Statistics tracking
        self.stats = {
            "llm_calls": 0,
            "llm_errors": 0,
            "action_counts": {"press_key": {}, "wait": 0},
            "locations_visited": {},
            "battles_fought": 0,
            "start_time": time.time()
        }
        
        # Initialize state
        self.state = PokemonAgentState()
        
        # Create LLM provider
        self.llm_provider = LLMProvider(
            provider=provider,
            model_name=model_name,
            temperature=temperature
        )
        
        # Create Pokemon server interface
        self.pokemon_server = PokemonServerInterface(server_url=server_url)
        
        # Log file for agent thinking
        self.log_filename = os.path.join(log_dir, f"agent_log_{self.session_id}.jsonl")
        
        # Setup the graph
        self.graph = self._setup_graph()
        
        logger.info(f"Pokemon Single Agent initialized with session ID: {self.session_id}, log level: {log_level}")
    
    def _get_agent_prompt(self) -> str:
        """
        Get the system prompt for the agent
        
        Returns:
            System prompt string
        """
        return """You are an expert Pokemon Red game-playing AI agent. You will analyze the game state, make decisions, and take actions to progress through the game.

Your goal is to explore the world, catch and train Pokemon, defeat gym leaders, and ultimately complete the game.

You will receive information about the current game state, including:
1. Current location and visual information
2. Dialog text if any is present
3. Your Pokemon team and their stats
4. Your inventory

You also have access to short-term memory containing recent observations and events (last ~20 steps).

Based on all available information, you will:
1. OBSERVE: Analyze the current game state to understand what's happening
2. THINK: Consider your options and strategy based on your memory and current situation
3. DECIDE: Choose a specific action to take (one of: pressing a button or waiting)
4. UPDATE MEMORY: Identify information worth remembering

For each cycle, provide your complete thinking process, then conclude with a specific action in this format:
ACTION: [press_key|wait] [button|frames]

Example actions:
ACTION: press_key a
ACTION: press_key up
ACTION: wait 30

Remember these important gameplay tips:
- The A button confirms selections and interacts with objects/NPCs
- The B button cancels or backs out of menus
- Use the D-pad (up/down/left/right) to navigate
- The START button opens the menu
- Navigate carefully and systematically to explore areas
- Talk to all NPCs for valuable information

Be strategic, observe carefully, and make decisions that will help you progress in the game!"""
    
    def _setup_graph(self):
        """
        Set up the LangGraph for the single agent system with LangGraph 0.3.28
        
        Returns:
            Configured LangGraph
        """
        import operator
        
        # Define the state schema
        class AgentState(BaseModel):
            """State tracked across graph execution"""
            pokemon_state: PokemonAgentState = Field(default_factory=PokemonAgentState)
            messages: List[Dict[str, str]] = Field(default_factory=list)
            recursion_depth: int = Field(default=0)  # Track recursion depth
            
        # Create the state graph
        graph = StateGraph(AgentState)
        
        # Define nodes
        def initialize_state(state: AgentState) -> AgentState:
            """Initialize the agent state if needed"""
            # If messages is empty, add system message
            if not state.messages:
                state.messages.append({
                    "role": "system",
                    "content": self._get_agent_prompt()
                })
            
            # Reset recursion depth to ensure fresh start
            state.recursion_depth = 0
            return state
        
        def observe(state: AgentState) -> AgentState:
            """Process game state observations"""
            # Extract game state
            game_state = state.pokemon_state.game_state
            if not game_state:
                return state
                
            # Add to trace
            state.pokemon_state.add_to_trace("observe", "Processing game observation")
            
            # Extract key information from game state
            key_info = self._extract_key_information(game_state)
            
            # Analyze screenshot if available
            screenshot_data = game_state.get("screenshot_base64", "")
            visual_analysis = ""
            if screenshot_data:
                try:
                    # Create human message with image
                    human_message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": f"""Analyze this Pokemon Red game screen. The current location is {key_info['location']}.
Current party: {[p['species'] for p in key_info['party']]}
Current dialog: {key_info['dialog']}
Available moves: {key_info['valid_moves']}

What can you observe in this image? Be specific about:
1. What's shown on the screen
2. Any text visible
3. The player's position and surroundings
4. Important game elements visible
5. Possible interactions available"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{screenshot_data}"}
                            }
                        ]
                    )
                    
                    # Get LLM analysis of the screenshot
                    response = self.llm_provider.get_llm().invoke([SystemMessage(content=self._get_agent_prompt()), human_message])
                    
                    # Add visual analysis to key info
                    key_info["visual_analysis"] = response.content
                    visual_analysis = response.content
                    
                    # Save screenshot for debugging if needed
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshots_dir = "logs/screenshots"
                    os.makedirs(screenshots_dir, exist_ok=True)
                    filename = f"{screenshots_dir}/step_{state.pokemon_state.step_count}_{timestamp}.png"
                    
                    try:
                        image_data = base64.b64decode(screenshot_data)
                        image = Image.open(io.BytesIO(image_data))
                        image.save(filename)
                        logger.debug(f"Screenshot saved to {filename}")
                    except Exception as e:
                        logger.error(f"Error saving screenshot: {e}")
                        
                except Exception as e:
                    logger.error(f"Error analyzing screenshot: {e}")
                    key_info["visual_analysis"] = "Error analyzing screenshot"
                    visual_analysis = "Error analyzing screenshot"
            
            # Update location knowledge
            location = key_info.get('location')
            if location and location != "Unknown":
                if location not in state.pokemon_state.known_locations:
                    state.pokemon_state.known_locations[location] = {
                        "first_visit": state.pokemon_state.step_count,
                        "visit_count": 1,
                        "coordinates": key_info.get('coordinates', [0, 0])
                    }
                else:
                    state.pokemon_state.known_locations[location]["visit_count"] += 1
                    state.pokemon_state.known_locations[location]["last_visit"] = state.pokemon_state.step_count
            
            # Don't create a separate memory entry for visual analysis
            # Instead, store it to be included in the consolidated memory
            state.pokemon_state.current_visual_analysis = visual_analysis
            
            return state
            
        def construct_prompt(state: AgentState) -> AgentState:
            """Create the prompt for the agent"""
            # Generate new human message
            human_message = self._create_thinking_prompt(state.pokemon_state)
            
            # Add to messages
            state.messages.append({
                "role": "human",
                "content": human_message
            })
            
            # Trim message history if too long (keep system message + recent exchanges)
            trimmed_messages = []
            system_message = None
            
            # Find and save system message
            for msg in state.messages:
                if msg.get("role") == "system":
                    system_message = msg
                    break
            
            # Always keep system message
            if system_message:
                trimmed_messages.append(system_message)
            
            # Add recent messages (last 4 exchanges = 8 messages)
            # Each exchange is human + AI
            recent_non_system = [msg for msg in state.messages if msg.get("role") != "system"]
            max_recent = min(8, len(recent_non_system))  # Keep at most 8 non-system messages
            trimmed_messages.extend(recent_non_system[-max_recent:])
            
            # Replace messages with trimmed version
            state.messages = trimmed_messages
            
            # Log message history size
            logger.debug(f"Message history size: {len(state.messages)} messages")
            if len(state.messages) > 10:
                logger.info(f"Trimmed message history to {len(state.messages)} messages")
            
            state.pokemon_state.add_to_trace("prompt", "Constructed prompt for LLM")
            return state
            
        def think(state: AgentState) -> AgentState:
            """Generate thinking and decisions using the LLM"""
            # Convert message objects for LLM
            messages = []
            for msg in state.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "human" or role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "ai" or role == "assistant":
                    messages.append(AIMessage(content=content))
            
            # Get the LLM
            llm = self.llm_provider.get_llm()
            
            # Log memory status
            logger.info("==== MEMORY STATUS ====")
            logger.info(f"  Short-term memories: {len(state.pokemon_state.short_term_memory)}")
            if state.pokemon_state.current_task:
                logger.info(f"  Current task: {state.pokemon_state.current_task}")
            
            # Add trace with synchronized step count
            state.pokemon_state.add_to_trace("think", "Calling LLM for thinking")
            logger.info(f"THINKING: Calling LLM to generate thoughts and decide action")
            
            # Set up retry parameters
            max_retries = 3
            retry_count = 0
            ai_message = None
            retry_backoff = [1, 3, 5]  # Wait times in seconds between retries
            
            while retry_count <= max_retries:
                try:
                    # Call LLM with timeout
                    if retry_count > 0:
                        logger.warning(f"Retry attempt {retry_count}/{max_retries} for LLM call")
                    
                    # Call LLM
                    response = llm.invoke(messages)
                    ai_message = response.content
                    
                    # If we got here, the call succeeded
                    if retry_count > 0:
                        logger.info(f"LLM call succeeded after {retry_count} retries")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    logger.error(f"Error in LLM call: {error_msg}")
                    
                    if retry_count <= max_retries:
                        # Wait before retrying
                        backoff_time = retry_backoff[min(retry_count-1, len(retry_backoff)-1)]
                        logger.info(f"Waiting {backoff_time}s before retry {retry_count}")
                        time.sleep(backoff_time)
                    else:
                        # We've exhausted our retries
                        logger.error(f"Failed to call LLM after {max_retries} attempts")
                        
                        # Fall back to a simple action based on the current state
                        state.pokemon_state.consecutive_llm_errors += 1
                        state.pokemon_state.last_error = f"LLM call failed after {max_retries} retries: {error_msg}"
                        
                        # Apply recovery strategy
                        if state.pokemon_state.consecutive_llm_errors >= 3:
                            # After 3 consecutive failures, try pressing random direction buttons to get unstuck
                            logger.warning("Multiple consecutive LLM failures - trying recovery action")
                            directions = ["up", "down", "left", "right"]
                            recovery_button = random.choice(directions)
                            state.pokemon_state.action = {"action_type": ActionType.PRESS_KEY, "button": recovery_button}
                            logger.info(f"RECOVERY ACTION: Press {recovery_button} button")
                        else:
                            # Default safe action
                            state.pokemon_state.action = {"action_type": ActionType.WAIT, "frames": 10}
                            logger.info("FALLBACK ACTION: Wait 10 frames")
                        
            # Reset consecutive error counter on success
            state.pokemon_state.consecutive_llm_errors = 0
            
            # Continue with normal processing if we have a valid response
            if ai_message:
                # Log the LLM response
                logger.info(f"LLM RESPONSE:\n{'-'*50}\n{ai_message}\n{'-'*50}")
                
                # Add AI response to messages
                state.messages.append({
                    "role": "ai",
                    "content": ai_message
                })
                
                # Parse the response
                result = self._parse_thinking_and_action(ai_message)
                
                # Update state with reasoning and action
                state.pokemon_state.action_reasoning = result.get("thinking", "")
                state.pokemon_state.action = result.get("action", {})
                
                # Log the parsed action
                action = result.get("action", {})
                action_type = action.get("action_type", "unknown")
                if action_type == ActionType.PRESS_KEY:
                    button = action.get("button", "unknown")
                    logger.info(f"DECISION: Press {button} button")
                elif action_type == ActionType.WAIT:
                    frames = action.get("frames", 0)
                    logger.info(f"DECISION: Wait {frames} frames")
                else:
                    logger.info(f"DECISION: {action}")
                
                # Process memory updates from explicit memory sections
                if result.get("memory_updates"):
                    logger.info(f"MEMORY UPDATES: {len(result.get('memory_updates', []))} new memories identified")
                    # We'll capture these in the consolidated memory later
                
                # Process task update
                if result.get("task_update"):
                    old_task = state.pokemon_state.current_task
                    new_task = result["task_update"]
                    
                    # If current task is completed, add to history
                    if old_task and old_task != new_task:
                        state.pokemon_state.task_history.append(old_task)
                        logger.info(f"TASK COMPLETED: {old_task}")
                    
                    # Update current task
                    state.pokemon_state.current_task = new_task
                    logger.info(f"NEW TASK: {new_task}")
                
                # Log the thinking
                self._log_thinking(
                    state.pokemon_state,
                    state.messages[-2]["content"],
                    ai_message,
                    result
                )
                
                # Add trace for decision
                state.pokemon_state.add_to_trace("decide", f"Decision: {result.get('action', {})}")
            else:
                # This should not happen, but just in case
                logger.error("Unexpected error: No AI message after successful LLM call")
                state.pokemon_state.consecutive_llm_errors += 1
                state.pokemon_state.last_error = "No AI message after successful LLM call"
                state.pokemon_state.action = {"action_type": ActionType.WAIT, "frames": 10}
                state.pokemon_state.add_to_trace("error", "No AI message after successful LLM call")
            
            return state
            
        def execute_action(state: AgentState) -> AgentState:
            """Execute the action decided by the agent"""
            action = state.pokemon_state.action
            if not action or "action_type" not in action:
                # No valid action, use default
                action = {"action_type": "wait", "frames": 10}
                state.pokemon_state.action = action
            
            # Add trace
            state.pokemon_state.add_to_trace("execute", f"Executing: {action}")
            
            try:
                # Execute action based on type
                if action["action_type"] == ActionType.PRESS_KEY:
                    button = action.get("button", Button.A)
                    logger.info(f"EXECUTING: Press {button} button")
                    game_state = self.pokemon_server.press_key(button)
                elif action["action_type"] == ActionType.WAIT:
                    frames = action.get("frames", 10)
                    logger.info(f"EXECUTING: Wait {frames} frames")
                    game_state = self.pokemon_server.wait(frames)
                else:
                    # Invalid action type, use default
                    logger.info(f"EXECUTING: Default wait (10 frames) due to invalid action type")
                    game_state = self.pokemon_server.wait(10)
                
                # Update game state
                state.pokemon_state.game_state = game_state
                
                # Increment step count ONLY here, after action is successfully executed
                state.pokemon_state.step_count += 1
                
                # Log location and dialog information if available
                location = game_state.get("location", "Unknown")
                dialog = game_state.get("dialog", "")
                if dialog:
                    logger.info(f"GAME STATE: At {location}, Dialog: \"{dialog[:50]}{'...' if len(dialog) > 50 else ''}\"")
                else:
                    logger.info(f"GAME STATE: At {location}, No dialog")
                
                # Log execution result
                self._log_execution(state.pokemon_state, action, game_state, None)
                
            except Exception as e:
                # Handle errors
                error_msg = str(e)
                state.pokemon_state.consecutive_llm_errors += 1
                state.pokemon_state.last_error = error_msg
                
                logger.error(f"Error executing action: {error_msg}")
                state.pokemon_state.add_to_trace("error", f"Error executing action: {error_msg}")
                
                # Log error information
                self._log_execution(state.pokemon_state, action, None, error_msg)
            
            return state
            
        def should_continue(state: AgentState) -> str:
            """Determine if the agent should continue or stop"""
            # Increment recursion depth counter
            state.recursion_depth += 1
            
            # Check if recursion depth has reached limit (one full cycle per step)
            if state.recursion_depth > 1:
                logger.info(f"Completed one full step cycle (depth={state.recursion_depth}), returning to main loop")
                # Reset recursion depth
                state.recursion_depth = 0
                return "stop"
                
            # Check if Pokemon server is still running
            if not self.pokemon_server.running:
                logger.warning("Stopping due to Pokemon server not running")
                return "stop"
                
            # Continue to next node in normal flow
            return "continue"
        
        # Add nodes to graph
        graph.add_node("initialize", initialize_state)
        graph.add_node("observe", observe)
        graph.add_node("construct_prompt", construct_prompt)
        graph.add_node("think", think)
        graph.add_node("execute", execute_action)
        
        # Define the edges
        graph.add_edge("initialize", "observe")
        graph.add_edge("observe", "construct_prompt")
        graph.add_edge("construct_prompt", "think")
        graph.add_edge("think", "execute")
        
        # Add conditional edge
        graph.add_conditional_edges(
            "execute",
            should_continue,
            {
                "continue": "observe",
                "stop": END
            }
        )
        
        # Set the entry point
        graph.set_entry_point("initialize")
        
        # Compile the graph without recursion_limit parameter
        return graph.compile()
    
    def _extract_key_information(self, game_state: Dict) -> Dict:
        """Extract key information from game state"""
        key_info = {
            "location": game_state.get("location", "Unknown"),
            "coordinates": game_state.get("coordinates", [0, 0]),
            "money": game_state.get("money", 0),
            "badges": game_state.get("badges", []),
            "valid_moves": game_state.get("valid_moves", []),
            "dialog": game_state.get("dialog", ""),
            "has_dialog": bool(game_state.get("dialog", "")),
            "score": game_state.get("score", 0.0),
            "step_number": game_state.get("step_number", 0),
            "session_id": game_state.get("session_id", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Process Pokemon party information
        pokemons = game_state.get("pokemons", [])
        party_info = []
        for p in pokemons:
            hp = p.get("hp", {})
            party_info.append({
                "nickname": p.get("nickname", ""),
                "species": p.get("species", ""),
                "level": p.get("level", 1),
                "hp": {
                    "current": hp.get("current", 0),
                    "max": hp.get("max", 0),
                    "percentage": int((hp.get("current", 0) / hp.get("max", 1)) * 100) if hp.get("max", 0) > 0 else 0
                },
                "status": p.get("status", ""),
                "moves": [m.get("name", "") for m in p.get("moves", [])]
            })
        key_info["party"] = party_info
        key_info["party_size"] = len(party_info)
        
        # Process inventory
        inventory = game_state.get("inventory", [])
        inv_list = []
        for item in inventory:
            inv_list.append({
                "item": item.get("item", ""),
                "quantity": item.get("quantity", 0)
            })
        key_info["inventory"] = inv_list
        
        # Add memory stats
        key_info["memory_stats"] = {
            "short_term_size": len(self.state.short_term_memory) if hasattr(self, "state") and hasattr(self.state, "short_term_memory") else 0
        }
        
        # Check if in battle
        key_info["in_battle"] = "battle" in game_state.get("location", "").lower() or bool(game_state.get("in_battle", False))
        
        return key_info
    
    def _create_thinking_prompt(self, state: PokemonAgentState) -> str:
        """Create the prompt for the agent's thinking process"""
        # Get game state info
        game_state = state.game_state or {}
        
        # Format current state information
        prompt_parts = [
            "# Current Game State",
            f"Location: {game_state.get('location', 'Unknown')}",
            f"Coordinates: {game_state.get('coordinates', [0, 0])}",
            f"Step: {state.step_count}",
            f"Score: {game_state.get('score', 0.0)}",
        ]
        
        # Add dialog if present
        dialog = game_state.get('dialog', '')
        if dialog:
            prompt_parts.extend([
                "",
                "# Dialog",
                dialog
            ])
        
        # Add Pokemon party
        pokemons = game_state.get('pokemons', [])
        if pokemons:
            prompt_parts.extend([
                "",
                "# Pokemon Party"
            ])
            
            for i, pokemon in enumerate(pokemons):
                hp = pokemon.get('hp', {})
                current_hp = hp.get('current', 0)
                max_hp = hp.get('max', 1)
                hp_percent = int((current_hp / max_hp) * 100) if max_hp > 0 else 0
                
                prompt_parts.append(
                    f"{i+1}. {pokemon.get('nickname', '')} ({pokemon.get('species', '')})" +
                    f" - Lv.{pokemon.get('level', 1)} - HP: {current_hp}/{max_hp} ({hp_percent}%)" +
                    f" - Status: {pokemon.get('status', 'Normal')}"
                )
                
                # Add moves
                moves = pokemon.get('moves', [])
                if moves:
                    move_texts = []
                    for move in moves:
                        pp = move.get('pp', {})
                        move_texts.append(f"{move.get('name', 'Unknown')} ({pp.get('current', 0)}/{pp.get('max', 0)})")
                    prompt_parts.append("   Moves: " + ", ".join(move_texts))
        
        # Add inventory
        inventory = game_state.get('inventory', [])
        if inventory:
            prompt_parts.extend([
                "",
                "# Inventory"
            ])
            
            for item in inventory:
                prompt_parts.append(f"- {item.get('item', 'Unknown')} x{item.get('quantity', 0)}")
        
        # Add current task
        if state.current_task:
            prompt_parts.extend([
                "",
                "# Current Task (IMPORTANT)",
                state.current_task
            ])
        
        # Add short-term memory (recent events)
        if state.short_term_memory:
            prompt_parts.extend([
                "",
                "# Recent History (Short-term Memory)"
            ])
            
            observations = []
            decisions = []
            outcomes = []
            
            for memory in reversed(state.short_term_memory):
                memory_type = memory.get('type', 'unknown')
                step = memory.get('step', 0)
                content = memory.get('content', '')
                location = memory.get('location', '')
                
                base_entry = f"Step {step}"
                if location:
                    base_entry += f" at {location}"
                base_entry += f": {content}"
                
                if memory_type == "observation":
                    observations.append(f"- {base_entry}")
                    if 'dialog' in memory and memory['dialog']:
                        dialog = memory['dialog']
                        if len(dialog) > 100:
                            dialog = dialog[:97] + "..."
                        observations.append(f"  Dialog: \"{dialog}\"")
                
                elif memory_type == "decision":
                    decisions.append(f"- {base_entry}")
                
                elif memory_type == "error" or "action_result" in memory:
                    outcomes.append(f"- {base_entry}")
            
            if observations:
                prompt_parts.append("\n## Recent Observations:")
                prompt_parts.extend(observations)
            
            if decisions:
                prompt_parts.append("\n## Recent Decisions:")
                prompt_parts.extend(decisions)
            
            if outcomes:
                prompt_parts.append("\n## Action Outcomes:")
                prompt_parts.extend(outcomes)
        
        # Add a list of known locations
        if state.known_locations:
            prompt_parts.extend([
                "",
                "# Known Locations"
            ])
            
            for location, info in state.known_locations.items():
                prompt_parts.append(f"- {location} (visited {info.get('visit_count', 1)} times)")
        
        # Add request for thinking, decision, and memory updates
        prompt_parts.extend([
            "",
            "# Instructions",
            "Based on the information above:",
            "1. OBSERVE: What do you see in the current game state? What's happening?",
            "2. THINK: Consider your options and strategy based on current observations and short-term memory.",
            "3. DECIDE: Choose a specific action to take.",
            "4. MEMORY: Note any information worth remembering.",
            "",
            "Provide your thinking in a natural way, without worrying about strict section formatting.",
            "Finally, conclude with a specific action in this format: ACTION: [press_key|wait] [button|frames]",
            "",
            "Example actions:",
            "ACTION: press_key a",
            "ACTION: press_key up",
            "ACTION: wait 30"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_thinking_and_action(self, response: str) -> Dict:
        """
        Parse the LLM response to extract thinking and action
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with parsed thinking and action
        """
        result = {
            "thinking": "",
            "action": None,
            "memory_updates": [],
            "task_update": None
        }
        
        # Split response into thinking and action parts
        thinking_lines = []
        action_line = None
        memory_updates = []
        task_update = None
        
        # Process response line by line
        lines = response.strip().split('\n')
        in_memory_section = False
        in_task_section = False
        in_observe_section = False
        in_think_section = False
        
        # Log the parsing start
        logger.debug(f"Parsing LLM response with {len(lines)} lines")
        
        current_section = "general"
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            line_lower = line.lower().strip()
            
            # Check for section headers
            if line_lower.startswith("observe:") or line_lower == "observe":
                in_observe_section = True
                in_think_section = False
                in_memory_section = False
                in_task_section = False
                current_section = "observe"
                logger.debug(f"Found OBSERVE section")
                continue
                
            if line_lower.startswith("think:") or line_lower == "think":
                in_observe_section = False
                in_think_section = True
                in_memory_section = False
                in_task_section = False
                current_section = "think"
                logger.debug(f"Found THINK section")
                continue
                
            # Check for action line
            if line_lower.startswith("action:"):
                action_line = line.strip()[7:].strip()  # Remove "ACTION:" prefix
                logger.debug(f"Found ACTION: {action_line}")
                continue
                
            # Check for memory update section
            if line_lower.startswith("memory update") or line_lower.startswith("memory updates"):
                in_observe_section = False
                in_think_section = False
                in_memory_section = True
                in_task_section = False
                current_section = "memory"
                logger.debug(f"Found MEMORY UPDATES section")
                continue
                
            # Check for task update section
            if line_lower.startswith("task update"):
                in_observe_section = False
                in_think_section = False
                in_memory_section = False
                in_task_section = True
                current_section = "task"
                logger.debug(f"Found TASK UPDATE section")
                continue
                
            # Process based on current section
            if in_memory_section:
                if line.strip() and not line.strip().startswith('#'):
                    # Determine memory type
                    memory_type = "short_term"
                    if ("important" in line_lower or 
                        "remember" in line_lower or 
                        "long-term" in line_lower or
                        "long term" in line_lower):
                        memory_type = "short_term"
                    
                    memory_updates.append({
                        "type": memory_type,
                        "content": line.strip()
                    })
                    logger.debug(f"Added {memory_type} memory: {line.strip()[:50]}...")
            elif in_task_section:
                if line.strip() and not line.strip().startswith('#'):
                    task_update = line.strip()
                    logger.debug(f"Set task update: {task_update}")
            else:
                # This is a thinking line
                # Add section prefix if we're tracking sections
                if current_section == "observe":
                    thinking_lines.append(line)
                elif current_section == "think":
                    thinking_lines.append(line)
                else:
                    thinking_lines.append(line)
        
        # Set thinking
        result["thinking"] = "\n".join(thinking_lines).strip()
        
        # Set memory updates
        result["memory_updates"] = memory_updates
        
        # Set task update
        result["task_update"] = task_update
        
        # Parse action
        if action_line:
            parts = action_line.strip().split()
            if len(parts) >= 2:
                action_type = parts[0].lower()
                if action_type == "press_key" and len(parts) >= 2:
                    button = parts[1].lower()
                    result["action"] = {"action_type": ActionType.PRESS_KEY, "button": button}
                    logger.debug(f"Parsed action: Press {button} button")
                elif action_type == "wait" and len(parts) >= 2:
                    try:
                        frames = int(parts[1])
                        result["action"] = {"action_type": ActionType.WAIT, "frames": frames}
                        logger.debug(f"Parsed action: Wait {frames} frames")
                    except ValueError:
                        # Default to 10 frames if parsing fails
                        result["action"] = {"action_type": ActionType.WAIT, "frames": 10}
                        logger.warning(f"Failed to parse frames in wait action, using default 10")
        
        # Default action if none parsed
        if not result["action"]:
            result["action"] = {"action_type": ActionType.WAIT, "frames": 10}
            logger.warning(f"No action found in response, using default wait 10 frames")
            
        # Log overall results
        action = result["action"]
        action_str = f"{action.get('action_type', 'unknown')}"
        if action.get('action_type') == ActionType.PRESS_KEY:
            action_str += f" {action.get('button', 'unknown')}"
        elif action.get('action_type') == ActionType.WAIT:
            action_str += f" {action.get('frames', 0)}"
            
        logger.info(f"PARSING RESULT: Action={action_str}, " + 
                  f"Memory updates={len(result['memory_updates'])}, " +
                  f"Task update={'Yes' if result['task_update'] else 'No'}, " +
                  f"Thinking={len(result['thinking'])} chars")
            
        return result
    
    def _log_thinking(self, state: PokemonAgentState, prompt: str, response: str, parsed_result: Dict) -> None:
        """
        Log the agent's thinking process to JSONL file
        
        Args:
            state: Current agent state
            prompt: The prompt sent to the LLM
            response: The LLM's response
            parsed_result: Parsed result with thinking and action
        """
        
        # Extract action
        action = parsed_result.get("action", {})
        action_str = ""
        if action:
            action_type = action.get("action_type", "")
            if action_type == ActionType.PRESS_KEY:
                action_str = f"press_key {action.get('button', '')}"
            elif action_type == ActionType.WAIT:
                action_str = f"wait {action.get('frames', 0)}"
        
        # Store for later use in _log_execution
        self._current_thinking = {
            "raw_response": response,       # Store the raw unparsed response
            "action": action_str
        }
        
        # No need to write to log here - will be done in _log_execution
    
    def _log_execution(self, state: PokemonAgentState, action: Dict, result: Dict, error: Optional[str] = None) -> None:
        """
        Log execution results to JSONL file, combining with thinking data
        
        Args:
            state: Current agent state
            action: Executed action
            result: Execution result (game state)
            error: Optional error information
        """
        # Get thinking data from previous call
        thinking_data = getattr(self, '_current_thinking', {})
        
        # Build consolidated log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": state.step_count,
            "location": result.get("location", "Unknown") if result else state.game_state.get("location", "Unknown"),
            "raw_response": thinking_data.get("raw_response", ""),  # Include raw LLM response
            "action": thinking_data.get("action", "")
        }
        
        # Add game state/result info
        if result:
            log_entry["dialog"] = result.get("dialog", "")
            log_entry["score"] = result.get("score", 0)
        
        # Add current task if available
        if state.current_task:
            log_entry["current_task"] = state.current_task
            
        # Add error information if present
        if error:
            log_entry["error"] = error
            
        # Get visual analysis if available
        visual_analysis = state.current_visual_analysis if hasattr(state, 'current_visual_analysis') else ""
            
        # Create a consolidated memory entry with all information from this step
        memory_entry = {
            "type": "consolidated",
            "step": state.step_count,
            "location": result.get("location", "Unknown") if result else state.game_state.get("location", "Unknown"),
            "raw_response": thinking_data.get("raw_response", ""),  # Include raw LLM response
            "observe": thinking_data.get("observe", ""),
            "think": thinking_data.get("think", ""),
            "decide": thinking_data.get("decide", ""),
            "memory": thinking_data.get("memory", ""),
            "action": thinking_data.get("action", ""),
            "dialog": result.get("dialog", "") if result else "",
            "result": "success" if error is None else "error",
            "visual_analysis": visual_analysis  # Include visual analysis in consolidated memory
        }
        
        # Add error details if present
        if error:
            memory_entry["error_details"] = error
            memory_entry["content"] = f"Error occurred: {error[:100]}" + ("..." if len(error) > 100 else "")
        elif state.last_error:
            # In case of LLM errors that were handled earlier
            memory_entry["error_details"] = state.last_error
            memory_entry["content"] = f"LLM error occurred: {state.last_error[:100]}" + ("..." if len(state.last_error) > 100 else "")
        else:
            # For successful executions, set content with location and dialog
            location = memory_entry["location"]
            dialog = memory_entry["dialog"]
            
            # Modify content to include visual analysis info
            content_parts = []
            content_parts.append(f"Observed at {location}")
            if dialog:
                content_parts.append(f"Dialog: {dialog[:50]}{'...' if len(dialog) > 50 else ''}")
            if visual_analysis:
                content_parts.append(f"Visual: {visual_analysis[:50]}{'...' if len(visual_analysis) > 50 else ''}")
            
            memory_entry["content"] = " | ".join(content_parts)
        
        # Add this consolidated entry to short-term memory
        state.add_to_short_term_memory(memory_entry)
        
        # Clear visual analysis
        state.current_visual_analysis = ""
        
        # Print the current memory state to console
        self._print_memory(state)
        
        # Write to JSONL file
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Clear thinking data to avoid duplicate entries
        self._current_thinking = {}
        
        # Minimal logging - just action summary
        action_type = action.get("action_type", "unknown")
        if action_type == ActionType.PRESS_KEY:
            button = action.get("button", "unknown")
            logger.info(f"Step {state.step_count}: press_key {button}")
        elif action_type == ActionType.WAIT:
            frames = action.get("frames", 0)
            logger.info(f"Step {state.step_count}: wait {frames}")
    
    def _print_memory(self, state: PokemonAgentState) -> None:
        """
        Print current short-term memory to console
        
        Args:
            state: Current Pokemon agent state
        """
        # Set log level to INFO temporarily to ensure output is displayed
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            # Print header
            logger.info("==== SHORT-TERM MEMORY CONTENTS ====")
            
            # Print short-term memory items (most recent first)
            if not state.short_term_memory:
                logger.info("  [No memories yet]")
            else:
                # Show last 5 items for brevity
                for i, memory in enumerate(reversed(state.short_term_memory[-5:])):
                    step = memory.get('step', 0)
                    mem_type = memory.get('type', 'unknown')
                    location = memory.get('location', 'Unknown')
                    
                    # For consolidated memories, print a summary
                    if mem_type == "consolidated":
                        action = memory.get('action', '')
                        observe_summary = memory.get('observe', '')[:50] + "..." if len(memory.get('observe', '')) > 50 else memory.get('observe', '')
                        
                        # Add visual analysis info if available
                        visual_info = ""
                        if memory.get('visual_analysis'):
                            visual_info = f" | Visual: {memory.get('visual_analysis', '')[:30]}..."
                            
                        logger.info(f"  {i+1}. Step {step} at {location}: Observed: {observe_summary} | Action: {action}{visual_info}")
                    else:
                        # For other memory types, print content
                        content = memory.get('content', '')[:50]
                        if len(memory.get('content', '')) > 50:
                            content += "..."
                        logger.info(f"  {i+1}. Step {step} [{mem_type}]: {content}")
            
            logger.info("====================================")
        finally:
            # Restore original logging level
            logger.setLevel(original_level)

    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the agent system
        
        Returns:
            Updated game state
        """
        try:
            # Initialize state for graph if first run
            if not hasattr(self, '_graph_state'):
                self._graph_state = {"pokemon_state": self.state, "messages": [], "recursion_depth": 0}
            
            # Reset recursion depth for this step
            self._graph_state["recursion_depth"] = 0
            
            # Run the graph with a fixed recursion limit
            logger.info("Starting graph execution for this step")
            result = self.graph.invoke(self._graph_state)
            
            # Update internal state
            self._graph_state = result
            self.state = result["pokemon_state"]
            
            # Ensure step count is correctly synchronized
            logger.info(f"Graph execution completed")
            
            # Update statistics
            self.stats["llm_calls"] += 1
            
            # If we have a location, update location statistics
            if self.state.game_state and "location" in self.state.game_state:
                location = self.state.game_state["location"]
                if location not in self.stats["locations_visited"]:
                    self.stats["locations_visited"][location] = 1
                else:
                    self.stats["locations_visited"][location] += 1
            
            # If we have an action, update action statistics
            if hasattr(self.state, "action") and self.state.action:
                action_type = self.state.action.get("action_type")
                if action_type == ActionType.PRESS_KEY:
                    button = self.state.action.get("button", "unknown")
                    if button not in self.stats["action_counts"]["press_key"]:
                        self.stats["action_counts"]["press_key"][button] = 1
                    else:
                        self.stats["action_counts"]["press_key"][button] += 1
                elif action_type == ActionType.WAIT:
                    self.stats["action_counts"]["wait"] += 1
            
            return self.state.game_state
            
        except Exception as e:
            logger.error(f"Error in step execution: {e}")
            self.state.last_error = str(e)
            self.stats["llm_errors"] += 1
            return self.state.game_state or {}
    
    def run(self, max_steps: int = 1000000) -> None:
        """
        Run the agent system for a specified number of steps
        
        Args:
            max_steps: Maximum number of steps to run
        """
        logger.info(f"===== Starting PokemonSingleAgent run with max_steps={max_steps} =====")
        
        try:
            step = 0
            
            while step < max_steps and self.pokemon_server.running:
                # Log each step with synchronized count
                logger.info(f"\n{'='*20} STEP {self.state.step_count + 1} {'='*20}")
                
                # Execute one step
                start_time = time.time()
                self.step()
                step_time = time.time() - start_time
                step += 1
                
                # Log progress and timing information
                logger.info(f"Step {self.state.step_count} completed in {step_time:.2f} seconds")
                
                # Log statistics every 10 steps
                if self.state.step_count % 10 == 0:
                    self._log_statistics()
                    
        except KeyboardInterrupt:
            logger.info("Run interrupted by user")
        except Exception as e:
            logger.error(f"Error during run: {e}")
        finally:
            logger.info(f"===== Run completed after {self.state.step_count} steps =====")
            # Stop the server
            try:
                self.pokemon_server.stop()
            except:
                pass
    
    def _log_statistics(self):
        """Log agent performance statistics"""
        # Calculate running time
        elapsed = time.time() - self.stats["start_time"]
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        # Calculate steps per hour
        steps_per_hour = int(self.state.step_count / (elapsed / 3600)) if elapsed > 0 else 0
        
        # Create statistics summary
        stats_summary = [
            "===== AGENT STATISTICS =====",
            f"Running time: {time_str}",
            f"Steps completed: {self.state.step_count} ({steps_per_hour} steps/hour)",
            f"LLM calls: {self.stats['llm_calls']} ({self.stats['llm_errors']} errors)",
            f"Memory: {len(self.state.short_term_memory)} short-term entries"
        ]
        
        # Location statistics
        if self.stats["locations_visited"]:
            top_locations = sorted(self.stats["locations_visited"].items(), key=lambda x: x[1], reverse=True)[:3]
            locations_str = ", ".join([f"{loc}: {count}" for loc, count in top_locations])
            stats_summary.append(f"Top locations: {locations_str}")
        
        # Action statistics
        if self.stats["action_counts"]["press_key"]:
            button_stats = []
            for button, count in sorted(self.stats["action_counts"]["press_key"].items(), 
                                      key=lambda x: x[1], reverse=True):
                button_stats.append(f"{button}: {count}")
            stats_summary.append(f"Button presses: {', '.join(button_stats)}")
        
        stats_summary.append(f"Wait commands: {self.stats['action_counts']['wait']}")
        stats_summary.append(f"Battles: {self.stats['battles_fought']}")
        stats_summary.append("============================")
        
        # Log all statistics
        for line in stats_summary:
            logger.info(line)
            
    def initialize(
        self, 
        headless: bool = True, 
        sound: bool = False,
        load_state_file: Optional[str] = None, 
        load_autosave: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize the game environment
        
        Args:
            headless: Whether to run without a GUI
            sound: Whether to enable sound
            load_state_file: Optional path to a saved state file to load
            load_autosave: Whether to load the latest autosave
            
        Returns:
            Initial game state
        """
        # Initialize the environment
        game_state = self.pokemon_server.initialize(
            headless=headless,
            sound=sound,
            load_state_file=load_state_file,
            load_autosave=load_autosave,
            session_id=self.session_id
        )
        
        # Update state
        self.state.game_state = game_state
        self.state.add_to_trace("initialize", "Game environment initialized")
        
        return game_state

def main():
    """Main function to run the agent"""
    parser = argparse.ArgumentParser(description="Pokemon Red Single Agent System")
    
    # Server settings
    parser.add_argument("--server_url", type=str, default="http://localhost:8080",
                      help="URL of the Pokemon Red server")
    
    # LLM settings
    parser.add_argument("--provider", type=str, default="claude",
                      choices=["claude", "openai", "gemini", "openrouter"],
                      help="LLM provider")
    parser.add_argument("--model", type=str, default=None,
                      help="Specific model name (if None, uses provider default)")
    parser.add_argument("--temperature", type=float, default=1,
                      help="Temperature for LLM generation")
    
    # Game settings
    parser.add_argument("--headless", action="store_true", 
                      help="Run in headless mode (no GUI)")
    parser.add_argument("--sound", action="store_true",
                      help="Enable sound")
    parser.add_argument("--load_state", type=str, default=None,
                      help="Load a saved state file")
    parser.add_argument("--load_autosave", action="store_true",
                      help="Load the latest autosave")
    
    # Run settings
    parser.add_argument("--max_steps", type=int, default=1000000,
                      help="Maximum number of steps to run")
    parser.add_argument("--session_id", type=str, default=None,
                      help="Session ID for resuming an existing session")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and run the agent
    agent = PokemonSingleAgent(
        server_url=args.server_url,
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        session_id=args.session_id
    )
    
    # Initialize the environment
    agent.initialize(
        headless=args.headless,
        sound=args.sound,
        load_state_file=args.load_state,
        load_autosave=args.load_autosave
    )
    
    # Run the agent
    agent.run(
        max_steps=args.max_steps
    )

if __name__ == "__main__":
    main() 