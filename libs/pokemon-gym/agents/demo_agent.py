import argparse
import logging
import time
from typing import Dict, List, Any, Optional
import json
import datetime
import re

import requests
from PIL import Image
import base64
import io
import copy
import os
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# System prompts for different providers
CLAUDE_SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

You have two tools available to control the game:
1. press_key - Press a single button (A, B, Start, Select, Up, Down, Left, Right)
2. wait - Wait for a specified number of frames

The A button is typically used to confirm actions or attack, B serves as a cancel or secondary action button, Start is used to pause the game or access menus, Select allows you to switch options or settings, and the directional buttons (Up, Down, Left, Right) control movement.

IMPORTANT: You can only take ONE action at a time. Choose the most appropriate single button press or wait action based on the current game state.

Before each action, explain your reasoning briefly, then use the appropriate tool to execute your chosen command.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

OPENAI_SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by deciding what button to press.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

You can control the game by clearly stating which button you want to press. For example:
- "I'll press the A button to talk to this NPC"
- "I should press UP to move toward the exit"
- "Let me press B to cancel this menu"
- "I need to press START to open the menu"
- "I'll press DOWN to move to the next option"
- "I should press SELECT to switch items"
- "Let me wait 30 frames to let the animation finish"

Available buttons are: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT.
You can also wait for a specified number of frames if needed.

IMPORTANT: 
1. You can only take ONE action at a time
2. You must vary your actions based on the game state - don't just press A repeatedly
3. Explore the game world by moving in different directions
4. Be sure to read dialogue text and respond appropriately
5. Plan your moves strategically to make progress in the game

Before pressing a button, explain your reasoning briefly, then clearly state which button you'll press.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

# Default system prompt (will be selected based on provider)
SYSTEM_PROMPT = CLAUDE_SYSTEM_PROMPT

# Available tools
AVAILABLE_TOOLS = [
    {
        "name": "press_key",
        "description": "Press a single button on the Game Boy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "enum": ["a", "b", "start", "select", "up", "down", "left", "right"],
                    "description": "The button to press. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                }
            },
            "required": ["button"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for a specified number of frames.",
        "input_schema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of frames to wait."
                }
            },
            "required": ["frames"],
        },
    }
]


class AIServerAgent:
    """AI Agent that controls Pokemon Red through the evaluator server API"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8080", 
                 provider: str = "claude", 
                 model_name: str = None, 
                 temperature: float = 1.0, 
                 max_tokens: int = 4000,
                 max_history: int = 30,
                 log_file: str = "agent_log.jsonl",
                 max_retries: int = 5,
                 retry_delay: float = 1.0):
        """
        Initialize the AI Agent
        
        Args:
            server_url: URL of the evaluation server
            provider: LLM provider ("claude", "openrouter", or "gemini")
            model_name: Model name for the selected provider (defaults based on provider)
            temperature: Temperature parameter for Claude
            max_tokens: Maximum tokens for Claude to generate
            max_history: Maximum number of messages to keep in history
            log_file: File to save generated content
            max_retries: Maximum number of retries for API calls
            retry_delay: Base delay between retries in seconds
        """
        # Server connection
        self.server_url = server_url
        self.session = requests.Session()
        self.initialized = False
        
        # Provider and model config
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history = max_history
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup provider-specific clients and models
        if self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
            self.model_name = model_name or "claude-3-5-sonnet-20241022"
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
            # Initialize the Gemini model
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            logger.info(f"Using Gemini provider with model: {self.model_name}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Choose 'claude', 'openai', 'openrouter', or 'gemini'")
        
        # Chat history
        self.message_history = []
        self.current_state = None
        self.running = True
        self.step_count = 0
        
        # Logging generated content
        self.log_file = log_file
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{os.path.splitext(self.log_file)[0]}_{self.provider}_{timestamp}.jsonl"
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Will log all generated content to: {self.log_file}")
        
        # Added for OpenAI
        self.pending_tool_responses = []
        self.user_message_with_results = None
    
    def log_step_data(self, step_num: int, user_message: Any, assistant_response, action_taken: Dict[str, Any]):
        """
        Log data from a step to the log file
        
        Args:
            step_num: Step number
            user_message: The message sent to Claude
            assistant_response: Claude's response object
            action_taken: Details of the action that was taken
        """
        # Extract text blocks and tool use from Claude's response
        text_content = []
        tool_uses = []
        
        # Handle different response formats based on API version
        if hasattr(assistant_response, 'content'):
            for block in assistant_response.content:
                if block.type == "text":
                    text_content.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append({
                        "name": block.name,
                        "input": block.input
                    })
        
        # Simplify user message if it's complex
        simplified_user_message = ""
        if isinstance(user_message, list):
            for item in user_message:
                if isinstance(item, dict) and item.get("type") == "text":
                    simplified_user_message += item.get("text", "") + "\n"
        else:
            simplified_user_message = str(user_message)
        
        # Create log entry
        log_entry = {
            "step": step_num,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_message": simplified_user_message,
            "assistant_response": {
                "text": text_content,
                "tool_uses": tool_uses
            },
            "action_taken": action_taken
        }
        
        # Append to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def initialize(self, headless: bool = True, sound: bool = False,
                  load_state_file: str = None, load_autosave: bool = False,
                  session_id: str = None) -> Dict[str, Any]:
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
            logger.info("Initializing environment...")
            
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
            
            response = self.session.post(
                f"{self.server_url}/initialize",
                headers={"Content-Type": "application/json"},
                json=init_params
            )
            
            response.raise_for_status()
            self.current_state = response.json()
            
            # Set initialization flag
            self.initialized = True
            
            # Create initial conversation history
            self.message_history = [{"role": "user", "content": "You may now begin playing Pokemon Red."}]
            
            # Log initial state
            with open(self.log_file, 'a', encoding='utf-8') as f:
                initial_entry = {
                    "step": "initial",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "game_state": {
                        "location": self.current_state.get('location', ''),
                        "coordinates": self.current_state.get('coordinates', []),
                        "money": self.current_state.get('money', 0),
                        "badges": self.current_state.get('badges', []),
                        "score": self.current_state.get('score', 0.0),  # Add score to log
                    }
                }
                f.write(json.dumps(initial_entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Initialization successful, location: {self.current_state['location']}")
            
            # Initialize tools
            self._initialize_tools()
            
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
    
    def _prepare_tools(self):
        """Prepare tools format based on provider"""
        if self.provider == "claude":
            # Claude uses original tools format
            return AVAILABLE_TOOLS
        else:
            # OpenRouter and Gemini use OpenAI format
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "press_key",
                        "description": "Press a single button on the Game Boy.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "button": {
                                    "type": "string",
                                    "enum": ["a", "b", "start", "select", "up", "down", "left", "right"],
                                    "description": "The button to press. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                                }
                            },
                            "required": ["button"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "wait",
                        "description": "Wait for a specified number of frames.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "frames": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "Number of frames to wait."
                                }
                            },
                            "required": ["frames"]
                        }
                    }
                }
            ]
    
    def _call_api_with_retry(self, api_func=None, *args, **kwargs):
        """
        Call an API function with retry mechanism
        
        Args:
            api_func: Function to call (only used for Claude)
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            API response
        """
        retries = 0
        last_exception = None
        
        while retries < self.max_retries:
            try:
                if self.provider == "claude":
                    # For Claude, use original API call style
                    if api_func:
                        return api_func(*args, **kwargs)
                    else:
                        # Direct call using message history and tools
                        messages = kwargs.get('messages', self.message_history)
                        return self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            system=CLAUDE_SYSTEM_PROMPT,
                            messages=messages,
                            tools=AVAILABLE_TOOLS,
                            temperature=self.temperature,
                        )
                elif self.provider == "gemini":
                    # For Gemini, use google.generativeai package
                    messages = kwargs.get('messages', self.message_history)
                    
                    # Convert to Gemini's chat format
                    gemini_messages = []
                    
                    # First, add a system message if available
                    system_prompt = CLAUDE_SYSTEM_PROMPT  # Use Claude's system prompt for Gemini
                    if system_prompt:
                        gemini_messages.append({"role": "user", "parts": [system_prompt]})
                        gemini_messages.append({"role": "model", "parts": ["I understand and will follow these instructions."]})
                    
                    # Add conversation messages
                    for msg in messages:
                        role = "user" if msg["role"] == "user" else "model"
                        
                        if isinstance(msg["content"], list):
                            # For multimodal content
                            parts = []
                            for item in msg["content"]:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        parts.append(item["text"])
                                    elif item.get("type") == "image":
                                        # Handle image content for Gemini
                                        src = item["source"]
                                        if src.get("type") == "base64":
                                            # For Gemini, create image from base64
                                            image_data = base64.b64decode(src["data"])
                                            img = Image.open(io.BytesIO(image_data))
                                            parts.append(img)
                                    elif item.get("type") == "image_url":
                                        # Handle OpenAI-style image URLs
                                        image_url = item["image_url"]["url"]
                                        if image_url.startswith("data:"):
                                            # Extract base64 data from data URL
                                            _, base64_data = image_url.split(",", 1)
                                            image_data = base64.b64decode(base64_data)
                                            img = Image.open(io.BytesIO(image_data))
                                            parts.append(img)
                            gemini_messages.append({"role": role, "parts": parts})
                        else:
                            # For simple text content
                            gemini_messages.append({"role": role, "parts": [msg["content"]]})
                    
                    # For debugging
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Gemini messages: {[(m['role'], len(m['parts'])) for m in gemini_messages]}")
                    
                    # Start a chat session and get response
                    chat = self.client.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
                    response = chat.send_message(
                        gemini_messages[-1]["parts"] if gemini_messages else "Hello",
                    )
                    
                    # Create response object similar to OpenAI for compatibility
                    text_response = response.text
                    
                    # Create a custom response object with structure similar to OpenAI for compatibility
                    custom_response = type('obj', (object,), {
                        'choices': [
                            type('obj', (object,), {
                                'message': type('obj', (object,), {
                                    'content': text_response,
                                    'tool_calls': []  # Empty tool calls since we handle functions differently
                                })
                            })
                        ],
                        'text': text_response,  # For compatibility
                        'model_name': self.model_name,  # For logging
                    })
                    
                    return custom_response
                else:
                    # For OpenAI, OpenRouter, use OpenAI compatible API
                    # Extract important params from kwargs
                    messages = kwargs.get('messages', self.message_history)
                    
                    # Clean up message history to avoid duplicate tool messages
                    # This ensures we only keep the most relevant tool responses
                    cleaned_messages = self._clean_message_history(messages)
                    
                    # Select appropriate system prompt based on provider
                    system_prompt = OPENAI_SYSTEM_PROMPT if self.provider in ["openai", "openrouter"] else CLAUDE_SYSTEM_PROMPT
                    
                    # Convert to OpenAI format if needed
                    openai_messages = [{"role": "system", "content": system_prompt}]
                    
                    for msg in cleaned_messages:
                        if msg["role"] == "user" and isinstance(msg["content"], list):
                            # Handle multimodal content (text + image)
                            content_list = []
                            for item in msg["content"]:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        content_list.append({"type": "text", "text": item["text"]})
                                    elif item.get("type") == "image":
                                        # Convert Claude image format to OpenAI format
                                        src = item["source"]
                                        if src.get("type") == "base64":
                                            content_list.append({
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{src.get('media_type', 'image/png')};base64,{src['data']}"
                                                }
                                            })
                            openai_messages.append({"role": "user", "content": content_list})
                        elif msg["role"] == "assistant" and isinstance(msg["content"], list):
                            # Handle assistant's response (might include tool uses in Claude format)
                            # For simplicity, extract text only for OpenAI format
                            text_content = ""
                            for item in msg["content"]:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_content += item.get("text", "") + "\n"
                                # Convert tool uses to text descriptions for OpenAI
                                elif isinstance(item, dict) and item.get("type") == "tool_use":
                                    tool_name = item.get("name", "unknown")
                                    tool_input = item.get("input", {})
                                    if tool_name == "press_key" and "button" in tool_input:
                                        text_content += f"I'll press the {tool_input['button']} button.\n"
                                    elif tool_name == "wait" and "frames" in tool_input:
                                        text_content += f"I'll wait for {tool_input['frames']} frames.\n"
                            if text_content:
                                openai_messages.append({"role": "assistant", "content": text_content})
                        else:
                            # Simple text messages
                            if msg["role"] in ["user", "assistant", "system"]:
                                # Skip tool messages, convert assistant tool_calls to plain text
                                if msg["role"] == "assistant" and "tool_calls" in msg:
                                    # Convert tool calls to text
                                    text_content = msg.get("content", "") or ""
                                    # Add text descriptions of each tool call
                                    for tc in msg.get("tool_calls", []):
                                        if isinstance(tc, dict) and "function" in tc:
                                            func = tc["function"]
                                            if "name" in func and "arguments" in func:
                                                try:
                                                    args = json.loads(func["arguments"])
                                                    if func["name"] == "press_key" and "button" in args:
                                                        text_content += f"I'll press the {args['button']} button.\n"
                                                    elif func["name"] == "wait" and "frames" in args:
                                                        text_content += f"I'll wait for {args['frames']} frames.\n"
                                                except:
                                                    pass
                                    openai_messages.append({"role": "assistant", "content": text_content})
                                elif msg["role"] != "tool":  # Skip tool messages entirely
                                    openai_messages.append(msg)
                    
                    # For OpenAI, log message history for debugging
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Messages for {self.provider}: {json.dumps(openai_messages, indent=2)}")
                    
                    # For OpenAI, we'll use text mode instead of tools
                    tools = []
                    if self.provider != "openai" and self.provider != "openrouter":
                        # Only add tools for non-OpenAI providers
                        tools = []
                        for tool in AVAILABLE_TOOLS:
                            tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "description": tool["description"],
                                    "parameters": tool["input_schema"]
                                }
                            })
                    
                    # Call OpenAI compatible API
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        messages=openai_messages,
                        tools=tools if tools else None,
                        temperature=self.temperature,
                    )
                    
            except Exception as e:
                last_exception = e
                retries += 1
                wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                logger.warning(f"API call failed (attempt {retries}/{self.max_retries}): {e}")
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"API call failed after {self.max_retries} attempts: {last_exception}")
        raise last_exception
    
    def _extract_action_from_text(self, text):
        """
        Extract action from text response for OpenAI text-only mode
        
        Args:
            text: The text response from OpenAI
            
        Returns:
            Action data dictionary
        """
        # Convert to lowercase for easier matching
        text_lower = text.lower()
        
        # Default action if nothing is found
        action_data = {"action_type": "press_key", "button": "a", "reason": "Default action"}
        
        # Enhanced pattern matching for button presses
        # Direct button mentions
        button_patterns = {
            # A button patterns
            r'\b(?:press|use|hit|push|click)(?:\s+the)?(?:\s+|\s*[":,-]\s*)a\s+button': "a",
            r'\bi(?:\'|_| am|\'ll| will| should| need to)(?:\s+|\s*[":,-]\s*)press(?:\s+the)?(?:\s+|\s*[":,-]\s*)a\s+button': "a",
            r'\bpressing(?:\s+the)?(?:\s+|\s*[":,-]\s*)a\s+button': "a",
            r'\bpress(?:\s+|\s*[":,-]\s*)a\b': "a",
            r'\bbutton(?:\s+|\s*[":,-]\s*)a\b': "a",
            
            # B button patterns
            r'\b(?:press|use|hit|push|click)(?:\s+the)?(?:\s+|\s*[":,-]\s*)b\s+button': "b",
            r'\bi(?:\'|_| am|\'ll| will| should| need to)(?:\s+|\s*[":,-]\s*)press(?:\s+the)?(?:\s+|\s*[":,-]\s*)b\s+button': "b",
            r'\bpressing(?:\s+the)?(?:\s+|\s*[":,-]\s*)b\s+button': "b",
            r'\bpress(?:\s+|\s*[":,-]\s*)b\b': "b",
            r'\bbutton(?:\s+|\s*[":,-]\s*)b\b': "b",
            r'\bcancel': "b",  # Often B is used to cancel
            
            # Directional buttons
            r'\b(?:press|use|hit|push|click|move)(?:\s+|\s*[":,-]\s*)(?:the\s+)?up(?:\s+button)?': "up",
            r'\b(?:press|use|hit|push|click|move)(?:\s+|\s*[":,-]\s*)(?:the\s+)?down(?:\s+button)?': "down",
            r'\b(?:press|use|hit|push|click|move)(?:\s+|\s*[":,-]\s*)(?:the\s+)?left(?:\s+button)?': "left",
            r'\b(?:press|use|hit|push|click|move)(?:\s+|\s*[":,-]\s*)(?:the\s+)?right(?:\s+button)?': "right",
            r'\bi(?:\'|_| am|\'ll| will| should| need to)(?:\s+|\s*[":,-]\s*)(?:go|head|move|walk)(?:\s+|\s*[":,-]\s*)up': "up",
            r'\bi(?:\'|_| am|\'ll| will| should| need to)(?:\s+|\s*[":,-]\s*)(?:go|head|move|walk)(?:\s+|\s*[":,-]\s*)down': "down",
            r'\bi(?:\'|_| am|\'ll| will| should| need to)(?:\s+|\s*[":,-]\s*)(?:go|head|move|walk)(?:\s+|\s*[":,-]\s*)left': "left",
            r'\bi(?:\'|_| am|\'ll| will| should| need to)(?:\s+|\s*[":,-]\s*)(?:go|head|move|walk)(?:\s+|\s*[":,-]\s*)right': "right",
            
            # Start and Select
            r'\b(?:press|use|hit|push|click)(?:\s+|\s*[":,-]\s*)(?:the\s+)?start(?:\s+button)?': "start",
            r'\b(?:press|use|hit|push|click)(?:\s+|\s*[":,-]\s*)(?:the\s+)?select(?:\s+button)?': "select",
            r'\b(?:open|access)(?:\s+|\s*[":,-]\s*)(?:the\s+)?menu': "start",  # Often Start is used to open menu
        }
        
        # Use regex to find patterns in the text
        for pattern, button in button_patterns.items():
            if re.search(pattern, text_lower):
                logger.info(f"Matched pattern '{pattern}' to button '{button}'")
                return {"action_type": "press_key", "button": button}
        
        # Check for general movement or navigation terms if no specific button was found
        # Only consider these if the more specific patterns above didn't match
        if "move" in text_lower or "walk" in text_lower or "go" in text_lower or "navigate" in text_lower:
            # Look for directions
            if "up" in text_lower or "north" in text_lower:
                return {"action_type": "press_key", "button": "up"}
            elif "down" in text_lower or "south" in text_lower:
                return {"action_type": "press_key", "button": "down"}
            elif "left" in text_lower or "west" in text_lower:
                return {"action_type": "press_key", "button": "left"}
            elif "right" in text_lower or "east" in text_lower:
                return {"action_type": "press_key", "button": "right"}
        
        # Check for dialog or menu interaction
        if "talk" in text_lower or "speak" in text_lower or "interact" in text_lower or "confirm" in text_lower or "select" in text_lower:
            return {"action_type": "press_key", "button": "a"}
        
        # Check for wait instructions
        if "wait" in text_lower:
            # Try to extract number of frames
            frame_matches = re.findall(r'wait (?:for )?(\d+) frames?', text_lower)
            if frame_matches:
                try:
                    frames = int(frame_matches[0])
                    return {"action_type": "wait", "frames": frames}
                except:
                    pass
            
            # Default wait if no specific frames mentioned
            return {"action_type": "wait", "frames": 30}
        
        # Log that we're using default action
        logger.info(f"No action pattern found in text: '{text[:100]}...' - using default action")
        return action_data

    def _clean_message_history(self, messages):
        """
        Completely clean message history to avoid accumulation.
        This is a more aggressive approach to solve message accumulation issues.
        
        Args:
            messages: Original message history
            
        Returns:
            Cleaned message history
        """
        # Check if message list is empty
        if not messages:
            return []
            
        cleaned = []
        
        # Select the appropriate system prompt based on provider
        system_prompt = OPENAI_SYSTEM_PROMPT if self.provider in ["openai", "openrouter"] else CLAUDE_SYSTEM_PROMPT
        
        # Keep system message if present
        for msg in messages:
            if msg["role"] == "system":
                cleaned.append({"role": "system", "content": system_prompt})
                break
        
        # Add a default system message if none was found
        if not cleaned:
            cleaned.append({"role": "system", "content": system_prompt})
        
        # For OpenAI/OpenRouter, simplify to basic chat history without tools
        if self.provider == "openai" or self.provider == "openrouter":
            # Find the last 10 messages that are either user or assistant (no tools)
            recent_messages = []
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    # Skip any message with tool_calls
                    if msg["role"] == "assistant" and "tool_calls" in msg:
                        # Convert tool_calls to text
                        text_content = msg.get("content", "") or ""
                        # Add text descriptions of each tool call
                        for tc in msg.get("tool_calls", []):
                            if isinstance(tc, dict) and "function" in tc:
                                func = tc["function"]
                                if "name" in func and "arguments" in func:
                                    try:
                                        args = json.loads(func["arguments"])
                                        if func["name"] == "press_key" and "button" in args:
                                            text_content += f"I'll press the {args['button']} button.\n"
                                        elif func["name"] == "wait" and "frames" in args:
                                            text_content += f"I'll wait for {args['frames']} frames.\n"
                                    except:
                                        pass
                        if text_content:
                            recent_messages.append({"role": "assistant", "content": text_content})
                    else:
                        recent_messages.append(msg)
            
            # Keep only the last 10 messages
            if len(recent_messages) > 10:
                recent_messages = recent_messages[-10:]
            
            # Add recent messages to cleaned list
            cleaned.extend(recent_messages)
            return cleaned
        
        # For Claude and other providers, use existing logic
        # Find the last sent user message
        last_user_msg = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_msg = messages[i]
                break
        
        # If a user message is found, add it
        if last_user_msg:
            cleaned.append(last_user_msg)
            
            # For other providers, find possible tool calls and responses
            # Only keep the last tool call and response (if exists)
            for i in range(len(messages) - 1, -1, -1):
                # Find assistant messages with tool calls
                if messages[i]["role"] == "assistant" and "tool_calls" in messages[i]:
                    # Add this assistant message
                    cleaned.append(messages[i])
                    
                    # Find corresponding tool response (only process first tool call)
                    if "tool_calls" in messages[i] and messages[i]["tool_calls"]:
                        # Handle both object and dictionary access patterns
                        tool_call = messages[i]["tool_calls"][0]
                        tool_call_id = tool_call.id if hasattr(tool_call, 'id') else tool_call.get("id")
                        
                        # Find corresponding tool response
                        for j in range(i + 1, len(messages)):
                            if (messages[j]["role"] == "tool" and 
                                messages[j].get("tool_call_id") == tool_call_id):
                                cleaned.append(messages[j])
                                break
                    
                    # Process tool chain only once
                    break
            
        return cleaned
    
    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on the next action based on current state
        
        Args:
            state: Current game state
            
        Returns:
            Game state after executing the action
        """
        # Prepare state information for Claude
        screenshot_b64 = state['screenshot_base64']
        
        # Create message content with the game state information (keep original format)
        content = [
            {"type": "text", "text": "Here is the current state of the game:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            },
            {"type": "text", "text": f"\nGame state information:"},
            {"type": "text", "text": f"Location: {state['location']}"},
            {"type": "text", "text": f"Coordinates: {state['coordinates']}"},
            {"type": "text", "text": f"Score: {state.get('score', 0.0)}"},
            {"type": "text", "text": f"Dialog: {state['dialog']}"},
            {"type": "text", "text": f"Pokemons: {state['pokemons']}"},
            {"type": "text", "text": f"Inventory: {state['inventory']}"},
            {"type": "text", "text": f"Valid moves: {state['valid_moves']}"},
            {"type": "text", "text": f"Money: {state['money']}"},
            {"type": "text", "text": f"Badges: {state['badges']}"},
        ]
        
        # Add dialog information if available
        if state['dialog']:
            content.append({"type": "text", "text": f"Dialog: {state['dialog']}"})
        
        # Add party pokemon information
        if state['pokemons']:
            pokemon_info = "\nParty Pokemon:\n"
            for i, pokemon in enumerate(state['pokemons']):
                pokemon_info += f"{i+1}. {pokemon['nickname']} ({pokemon['species']}) Lv.{pokemon['level']} " \
                               f"HP: {pokemon['hp']['current']}/{pokemon['hp']['max']}\n"
            content.append({"type": "text", "text": pokemon_info})
        
        # Add inventory information
        if state['inventory']:
            inventory_info = "\nInventory:\n"
            for item in state['inventory']:
                inventory_info += f"- {item['item']}: {item['quantity']}\n"
            content.append({"type": "text", "text": inventory_info})
        
        # Add collision map and valid moves (if available)
        collision_map = state.get('collision_map')
        
        if collision_map:
            content.append({"type": "text", "text": f"\nCollision map:\n{collision_map}"})
        
        # Add message to history - keep original format for all providers
        self.message_history.append({"role": "user", "content": content})
        
        # Get model response with retry
        try:
            if self.provider == "claude":
                # Use original Claude API call
                response = self._call_api_with_retry(
                    self.client.messages.create,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=self.message_history,
                    tools=AVAILABLE_TOOLS,
                    temperature=self.temperature,
                )
            elif self.provider == "gemini":
                # Use Gemini API call
                response = self._call_api_with_retry()
            else:
                # Use OpenAI-compatible call with _call_api_with_retry without args
                response = self._call_api_with_retry()
        except Exception as e:
            logger.error(f"Failed to get response from {self.provider} after retries: {e}")
            # Default to a simple action if API calls fail completely
            logger.warning("Falling back to default action (press A)")
            
            # Create a minimal response for logging
            if self.provider == "claude":
                response = type('obj', (object,), {
                    'content': [
                        type('obj', (object,), {'type': 'text', 'text': 'API call failed, using default action'})
                    ]
                })
            else:
                # Mock OpenAI format response
                response = type('obj', (object,), {
                    'choices': [
                        type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': 'API call failed, using default action'
                            })
                        })
                    ]
                })
            
            # Skip normal processing and return a default action
            next_state = self.take_action("press_key", keys=["a"])
            
            # Add a failure note to history (keep format consistent with original)
            if self.provider == "claude":
                self.message_history.append({
                    "role": "assistant", 
                    "content": [{"type": "text", "text": "API call failed, used default action (press A)"}]
                })
            else:
                self.message_history.append({
                    "role": "assistant", 
                    "content": "API call failed, used default action (press A)"
                })
            
            # Add log entry
            self.log_step_data(
                step_num=self.step_count - 1,
                user_message=content,
                assistant_response=response,
                action_taken={"action_type": "press_key", "button": "a", "reason": "API failure fallback"}
            )
            
            return next_state
        
        # Extract action data and process response based on provider
        action_data = {}
        assistant_content = []
        
        if self.provider == "claude":
            # Process Claude response (keep original format)
            tool_calls = [block for block in response.content if hasattr(block, 'type') and block.type == "tool_use"]
            
            # Collect Claude's response
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        logger.info(f"[Claude] {block.text}")
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        logger.info(f"[Claude] Using tool: {block.name if hasattr(block, 'name') else 'unknown'}")
                        assistant_content.append({"type": "tool_use", **dict(block)})
            
            # Prepare action data
            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call.name if hasattr(tool_call, 'name') else None
                tool_input = tool_call.input if hasattr(tool_call, 'input') else {}
                tool_id = tool_call.id if hasattr(tool_call, 'id') else None
                
                if tool_name == "press_key":
                    button = tool_input.get("button") if isinstance(tool_input, dict) else None
                    action_data = {"action_type": "press_key", "button": button, "tool_id": tool_id}
                elif tool_name == "wait":
                    frames = tool_input.get("frames") if isinstance(tool_input, dict) else None
                    action_data = {"action_type": "wait", "frames": frames, "tool_id": tool_id}
            else:
                # Default action if no tool call
                action_data = {"action_type": "press_key", "button": "a", "reason": "No tool call found"}
        elif self.provider == "gemini":
            # Gemini handling is similar to OpenAI/OpenRouter but with specific adaptations
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                
                # Get text content
                if message.content:
                    logger.info(f"[{self.provider}] {message.content}")
                    assistant_content = message.content
                    
                    # For Gemini, parse the text response to determine action
                    action_data = self._extract_action_from_text(message.content)
                    
                # No tool calls support through Gemini right now, just use text parsing
                if not action_data:
                    action_data = {"action_type": "press_key", "button": "a", "reason": "No valid action found in text"}
            else:
                # Malformed response, default action
                action_data = {"action_type": "press_key", "button": "a", "reason": "Malformed response"}
        else:
            # Process OpenAI-compatible response (OpenRouter or Gemini)
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                
                # Get text content
                if message.content:
                    logger.info(f"[{self.provider}] {message.content}")
                    assistant_content = message.content
                    
                    # For OpenAI, parse the text response to determine action
                    # Look for keywords indicating button presses or wait actions
                    action_data = self._extract_action_from_text(message.content)
                    
                # Extract tool calls if present - only used for non-OpenAI providers
                if self.provider != "openai" and self.provider != "openrouter" and hasattr(message, 'tool_calls') and message.tool_calls:
                    # Log all tool calls
                    if len(message.tool_calls) > 1:
                        logger.info(f"[{self.provider}] Multiple tool calls detected: {len(message.tool_calls)}")
                        for idx, tc in enumerate(message.tool_calls):
                            # Safe access to function name
                            function_name = tc.function.name if hasattr(tc, 'function') and hasattr(tc.function, 'name') else tc.get('function', {}).get('name')
                            logger.info(f"Tool call {idx+1}: {function_name}")
                    
                    # Process tool calls (may be multiple with OpenRouter)
                    tool_call = None
                    
                    # Try to find a 'press_key' or 'wait' tool call as they are higher priority
                    for tc in message.tool_calls:
                        # Safe access to function name
                        function_name = tc.function.name if hasattr(tc, 'function') and hasattr(tc.function, 'name') else tc.get('function', {}).get('name')
                        if function_name in ["press_key", "wait"]:
                            tool_call = tc
                            break
                    
                    # If no priority tool call found, use the first one
                    if not tool_call and message.tool_calls:
                        tool_call = message.tool_calls[0]
                    
                    if tool_call:
                        # Safe access to function name
                        function_name = tool_call.function.name if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name') else tool_call.get('function', {}).get('name')
                        logger.info(f"[{self.provider}] Selected tool call: {function_name}")
                        
                        try:
                            # Safe access to function arguments
                            function_args = tool_call.function.arguments if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments') else tool_call.get('function', {}).get('arguments')
                            
                            # Parse args
                            args = json.loads(function_args)
                            
                            # Safe access to tool call ID
                            tool_id = tool_call.id if hasattr(tool_call, 'id') else tool_call.get('id')
                            
                            # Extract action
                            if function_name == "press_key":
                                button = args.get("button")
                                if button:
                                    action_data = {"action_type": "press_key", "button": button, "tool_id": tool_id}
                            elif function_name == "wait":
                                frames = args.get("frames")
                                if frames:
                                    action_data = {"action_type": "wait", "frames": frames, "tool_id": tool_id}
                            
                            # Log selected action
                            logger.info(f"Selected action: {action_data}")
                        except json.JSONDecodeError:
                            function_args = tool_call.function.arguments if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments') else tool_call.get('function', {}).get('arguments')
                            logger.error(f"Failed to parse tool arguments: {function_args}")
                
                # Default action if no valid tool call found
                if not action_data:
                    action_data = {"action_type": "press_key", "button": "a", "reason": "No valid tool call found"}
            else:
                # Malformed response, default action
                action_data = {"action_type": "press_key", "button": "a", "reason": "Malformed response"}
        
        # Log the response and action before executing
        self.log_step_data(
            step_num=self.step_count,
            user_message=content,
            assistant_response=response,
            action_taken=action_data
        )
        
        # Execute the action
        if action_data["action_type"] == "press_key":
            next_state = self.take_action("press_key", keys=[action_data["button"]])
        elif action_data["action_type"] == "wait":
            next_state = self.take_action("wait", frames=action_data["frames"])
        else:
            # Fallback
            next_state = self.take_action("press_key", keys=["a"])
        
        # Add assistant's response to history (keep provider-specific format)
        if self.provider == "claude":
            # Claude format (list of content blocks)
            self.message_history.append({"role": "assistant", "content": assistant_content})
            
            # Create tool_result response
            tool_result_content = []
            
            # Add tool result if a tool was used
            if "tool_id" in action_data:
                tool_id = action_data["tool_id"]
                
                # Create result message
                if action_data["action_type"] == "press_key":
                    result_message = f"Button '{action_data['button']}' pressed successfully. New location: {next_state['location']}, Coordinates: {next_state['coordinates']}"
                elif action_data["action_type"] == "wait":
                    result_message = f"Waited for {action_data['frames']} frames. New location: {next_state['location']}, Coordinates: {next_state['coordinates']}"
                else:
                    result_message = f"Action executed. New location: {next_state['location']}, Coordinates: {next_state['coordinates']}"
                
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_message
                })
            
            # Add screenshot
            tool_result_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": next_state['screenshot_base64'],
                },
            })
            
            # Add tool result response to history
            self.message_history.append({"role": "user", "content": tool_result_content})
        elif self.provider == "gemini":
            # Gemini may have special requirements for message history format
            logger.info(f"Gemini: Adding result directly to user message")
            
            # Add assistant's response as plain text
            self.message_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # Create result message
            if action_data["action_type"] == "press_key":
                result_message = f"Button '{action_data['button']}' pressed successfully. New location: {next_state['location']}, Coordinates: {next_state['coordinates']}"
            elif action_data["action_type"] == "wait":
                result_message = f"Waited for {action_data['frames']} frames. New location: {next_state['location']}, Coordinates: {next_state['coordinates']}"
            else:
                result_message = f"Action executed. New location: {next_state['location']}, Coordinates: {next_state['coordinates']}"
            
            # Add result as user message with new image
            # Prepare image for Gemini
            new_user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Result: {result_message}\nCurrent location: {next_state['location']}, Coordinates: {next_state['coordinates']}"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": next_state['screenshot_base64']
                        }
                    }
                ]
            }
            self.message_history.append(new_user_msg)
            
            # For debugging
            logger.info(f"Message history size: {len(self.message_history)} messages")
            roles = [msg["role"] for msg in self.message_history]
            logger.info(f"Message roles: {roles}")
        elif self.provider == "openai" or self.provider == "openrouter":
            # Add assistant's response as plain text
            self.message_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # Add result as user message with new image
            new_user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Result: {result_message}\nCurrent location: {next_state['location']}, Coordinates: {next_state['coordinates']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{next_state['screenshot_base64']}"}}
                ]
            }
            self.message_history.append(new_user_msg)
            
            # For debugging
            logger.info(f"Message history size: {len(self.message_history)} messages")
            roles = [msg["role"] for msg in self.message_history]
            logger.info(f"Message roles: {roles}")
        else:
            # No tool call, just add assistant response and new game state/screenshot
            self.message_history.append({"role": "assistant", "content": assistant_content})
            self.message_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Current location: {next_state['location']}, Coordinates: {next_state['coordinates']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{next_state['screenshot_base64']}"}}
                ]
            })
        
        # Check if history needs summarization
        if len(self.message_history) > self.max_history:
            self._summarize_history()
        
        return next_state
    
    def _summarize_history(self):
        """Summarize conversation history to save context space"""
        logger.info("Summarizing conversation history...")
        
        try:
            if self.provider == "claude":
                # Use original Claude summarization flow
                # Create summary request
                summary_messages = copy.deepcopy(self.message_history)
                summary_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please create a detailed summary of our conversation history so far. This summary will replace the full conversation history to manage the context window."
                        }
                    ]
                })
                
                # Get Claude's summary
                response = self._call_api_with_retry(
                    self.client.messages.create,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=summary_messages,
                    temperature=self.temperature
                )
                
                # Extract summary text
                summary_text = " ".join([
                    block.text for block in response.content 
                    if hasattr(block, 'type') and block.type == "text" and hasattr(block, 'text')
                ])
            elif self.provider == "gemini":
                # Gemini-specific summarization
                # Simplify history for summarization request
                simplified_messages = []
                for msg in self.message_history:
                    if msg["role"] in ["user", "assistant"]:
                        if isinstance(msg["content"], list):
                            # Extract text from content blocks
                            text_parts = []
                            for item in msg["content"]:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            if text_parts:
                                simplified_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})
                        else:
                            # Already text content
                            simplified_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add summary request
                simplified_messages.append({
                    "role": "user", 
                    "content": "Please create a detailed summary of our conversation history so far. This summary will replace the full conversation history to manage the context window."
                })
                
                # Get summary using Gemini API
                response = self._call_api_with_retry(messages=simplified_messages)
                
                # Extract summary text from Gemini response
                summary_text = response.choices[0].message.content if hasattr(response, 'choices') else response.text
            else:
                # OpenAI-compatible summarization
                # Simplify history for summarization request
                simplified_messages = []
                for msg in self.message_history:
                    if msg["role"] in ["user", "assistant"]:
                        if isinstance(msg["content"], list):
                            # Extract text from content blocks
                            text_parts = []
                            for item in msg["content"]:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            if text_parts:
                                simplified_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})
                        else:
                            # Already text content
                            simplified_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add summary request
                simplified_messages.append({
                    "role": "user", 
                    "content": "Please create a detailed summary of our conversation history so far. This summary will replace the full conversation history to manage the context window."
                })
                
                # Get summary using OpenAI-compatible API
                response = self._call_api_with_retry(messages=simplified_messages)
                
                # Extract summary text from OpenAI response
                summary_text = response.choices[0].message.content
            
            # Log the summary
            logger.info(f"Generated summary:\n{summary_text}")
            with open(self.log_file, 'a', encoding='utf-8') as f:
                summary_entry = {
                    "step": f"summary_{self.step_count}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "summary": summary_text
                }
                f.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')
            
            # Keep recent messages
            recent_msgs = self.message_history[-2:] if len(self.message_history) >= 2 else []
            
            # Replace history with summary message
            if self.provider == "claude":
                # Claude format
                self.message_history = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"CONVERSATION HISTORY SUMMARY: {summary_text}"
                            },
                            {
                                "type": "text",
                                "text": "You may now continue playing Pokemon Red. Make your next decision based on the current game state."
                            }
                        ]
                    }
                ] + recent_msgs
            elif self.provider == "gemini":
                # Gemini format (similar to OpenAI)
                self.message_history = [
                    {
                        "role": "user",
                        "content": f"CONVERSATION HISTORY SUMMARY: {summary_text}\n\nYou may now continue playing Pokemon Red. Make your next decision based on the current game state."
                    }
                ] + recent_msgs
            else:
                # OpenAI format
                self.message_history = [
                    {
                        "role": "user",
                        "content": f"CONVERSATION HISTORY SUMMARY: {summary_text}\n\nYou may now continue playing Pokemon Red. Make your next decision based on the current game state."
                    }
                ] + recent_msgs
            
            logger.info("Conversation history successfully summarized")
            
        except Exception as e:
            logger.error(f"Failed to summarize history: {e}")
            # Fallback to truncating history to avoid context issues
            logger.warning("Truncating history as fallback")
            # Keep a few most recent messages
            if len(self.message_history) > 6:
                self.message_history = self.message_history[-6:]
    
    def run(self, max_steps: int = 100) -> None:
        """
        Run the AI Agent
        
        Args:
            max_steps: Maximum number of steps to run
        """
        if not self.initialized:
            raise RuntimeError("Environment not initialized, please call initialize() first")
        
        logger.info(f"Starting AI Agent, max steps: {max_steps}")
        
        try:
            # Process initial state
            logger.info("Processing initial state...")
            current_state = self.current_state
            
            while self.running and self.step_count < max_steps:
                # Decide and execute action
                logger.info(f"Step {self.step_count+1}/{max_steps}")
                
                try:
                    current_state = self.decide_action(current_state)
                    
                    # Display current state information
                    location = current_state['location']
                    coords = current_state['coordinates']
                    party_size = len(current_state['pokemons'])
                    score = current_state.get('score', 0.0)
                    
                    logger.info(f"Location: {location}, Coordinates: {coords}, Party size: {party_size}, Score: {score:.1f}")
                    
                    # Small delay between steps to avoid overwhelming API
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error in step {self.step_count+1}: {e}")
                    logger.warning("Attempting to continue with next step after error")
                    # Add a longer delay after error
                    time.sleep(2)
        
        except KeyboardInterrupt:
            logger.info("User interrupted, stopping run")
        except Exception as e:
            logger.error(f"Run error: {e}")
        finally:
            logger.info(f"Run ended, executed {self.step_count} steps")
            
            # Log final state
            with open(self.log_file, 'a', encoding='utf-8') as f:
                final_entry = {
                    "step": "final",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "total_steps": self.step_count,
                    "final_location": self.current_state.get('location', '') if self.current_state else None,
                    "final_badges": self.current_state.get('badges', []) if self.current_state else None
                }
                f.write(json.dumps(final_entry, ensure_ascii=False) + '\n')
    
    def stop(self) -> Dict[str, Any]:
        """Stop the environment"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        try:
            response = self.session.post(f"{self.server_url}/stop")
            response.raise_for_status()
            self.initialized = False
            self.running = False
            logger.info("Environment stopped")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Environment stop error: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Server response: {e.response.text}")
            raise

    def _initialize_tools(self):
        self.pending_tool_responses = []
        self.user_message_with_results = None


def save_screenshot(screenshot_base64: str, filename: str) -> None:
    """
    Save a screenshot from base64 encoded image data
    
    Args:
        screenshot_base64: base64 encoded image data
        filename: filename to save
    """
    image_data = base64.b64decode(screenshot_base64)
    image = Image.open(io.BytesIO(image_data))
    
    # Upscale the image for better visibility
    width, height = image.size
    image = image.resize((width*3, height*3), Image.NEAREST)
    
    image.save(filename)
    logger.info(f"Screenshot saved to {filename}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Pokemon AI Agent")
    parser.add_argument("--server", type=str, default="http://localhost:8080", help="Evaluation server URL")
    parser.add_argument("--steps", type=int, default=1000000, help="Number of steps to run")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--sound", action="store_true", help="Enable sound")
    parser.add_argument("--provider", type=str, default="claude", choices=["claude", "openai", "openrouter", "gemini"], 
                      help="LLM provider to use (claude, openai, openrouter, gemini)")
    parser.add_argument("--model", type=str, default=None, help="Model name for the selected provider")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for Claude")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum tokens for Claude to generate")
    parser.add_argument("--log-file", type=str, default="agent_log.jsonl", help="File to save agent logs")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries for API calls")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Base delay between retries in seconds")
    parser.add_argument("--load-state", type=str, help="Path to a saved state file to load")
    parser.add_argument("--load-autosave", action="store_true", help="Load the latest autosave")
    parser.add_argument("--session", type=str, help="Session ID to continue (e.g., session_20250404_180209)")
    
    args = parser.parse_args()
    
    # Create AI Agent
    agent = AIServerAgent(
        server_url=args.server,
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_file=args.log_file,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    try:
        # Initialize environment
        initial_state = agent.initialize(
            headless=args.headless, 
            sound=args.sound,
            load_state_file=args.load_state,
            load_autosave=args.load_autosave,
            session_id=args.session
        )
        
        # Run AI Agent
        logger.info(f"Starting AI Agent using {args.provider}, max steps: {args.steps}")
        agent.run(max_steps=args.steps)
        
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Ensure environment is stopped
        if agent.initialized:
            agent.stop()


if __name__ == "__main__":
    main() 