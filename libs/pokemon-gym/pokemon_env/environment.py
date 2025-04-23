import time
import logging
import base64
import io
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from PIL import Image

from pokemon_env.emulator import Emulator
from pokemon_env.action import Action, ActionType, PressKey, Wait

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Represents the state of the game at a particular point in time."""
    
    # Basic state information
    player_name: str
    rival_name: str
    money: int
    location: str
    coordinates: Tuple[int, int]
    badges: List[str]
    valid_moves: List[str]
    inventory: List[Dict[str, Any]]
    dialog: str
    pokemons: List[Dict[str, Any]]
    screenshot: Image.Image  # PIL Image of the current screen
    
    @property
    def screenshot_base64(self, upscale: int = 2) -> str:
        """Convert the screenshot to a base64 string."""
        # Resize if needed
        if upscale > 1:
            new_size = (self.screenshot.width * upscale, self.screenshot.height * upscale)
            screenshot = self.screenshot.resize(new_size)
        else:
            screenshot = self.screenshot
            
        # Convert to base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return base64.standard_b64encode(buffered.getvalue()).decode()


class PokemonEnvironment:
    """Environment for Pokemon Red that provides a clean interface for agents."""
    
    def __init__(self, rom_path: str, headless: bool = True, sound: bool = False):
        """
        Initialize the Pokemon environment.
        
        Args:
            rom_path: Path to the Pokemon ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()
        logger.info("emulator initialized")
        # Store gameplay information
        self.steps_taken = 0
        self.game_history: Dict[int, Dict] = {}
        self.action_times: Dict[int, float] = {}
        
        # Store the current state
        self._current_state = self._get_current_state()
        logger.info("current state initialized")
        
    def step(self, action: Action) -> GameState:
        """
        Take a step in the environment using the provided action.
        
        Args:
            action: The action to take
            
        Returns:
            The new state after taking the action
        """
        # Record the start time for this action
        start_time = time.time()
        
        # Process the action based on its type
        logger.info(f"Processing action: {action}")
        if action.action_type == ActionType.PRESS_KEY:
            assert isinstance(action, PressKey)
            self.emulator.press_buttons(action.keys)
        elif action.action_type == ActionType.WAIT:
            assert isinstance(action, Wait)
            self.emulator.tick(action.frames)
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
        
        # Update the current state
        logger.info("updating current state")
        self._current_state = self._get_current_state()
        
        # Increment step counter
        self.steps_taken += 1
        
        # Record action execution time
        end_time = time.time()
        self.action_times[self.steps_taken] = end_time - start_time
        
        # Store the state in history
        self.game_history[self.steps_taken] = {
            'action': action.to_dict(),
            'state': {
                **self._current_state.__dict__
            },
            'execution_time': self.action_times[self.steps_taken]
        }
        
        return self._current_state
    
    def _get_current_state(self) -> GameState:
        """Get the current state of the game."""
        memory_info = self.emulator.get_state_from_memory()
        screenshot = self.emulator.get_screenshot()
        
        return GameState(
            player_name=memory_info.get('player', {}).get('name', 'UNKNOWN'),
            rival_name=memory_info.get('player', {}).get('rival_name', 'UNKNOWN'),
            money=memory_info.get('player', {}).get('money', 0),
            location=memory_info.get('player', {}).get('location', 'UNKNOWN'),
            coordinates=memory_info.get('player', {}).get('coordinates', (0, 0)),
            badges=memory_info.get('player', {}).get('badges', []),
            valid_moves=memory_info.get('valid_moves', []),
            inventory=memory_info.get('inventory', []),
            dialog=memory_info.get('dialog', 'UNKNOWN'),
            pokemons=memory_info.get('pokemons', []),
            screenshot=screenshot
        )


    @property
    def state(self) -> GameState:
        """Get the current state of the game."""
        return self._current_state
    
    def get_collision_map(self) -> Optional[str]:
        """Get a text representation of the current collision map."""
        return self.emulator.get_collision_map()
    
    def get_valid_moves(self) -> List[str]:
        """Get a list of valid movement directions."""
        return self.emulator.get_valid_moves()
    
    def get_game_history(self) -> Dict[int, Dict]:
        """Get the entire game history."""
        return self.game_history
    
    def get_average_action_time(self) -> float:
        """Get the average time taken per action."""
        if not self.action_times:
            return 0.0
        return sum(self.action_times.values()) / len(self.action_times)
    
    def save_state(self, state_filename: str) -> None:
        """
        Save the current environment state to a file.
        
        Args:
            state_filename: Path to save the state file
        """
        self.emulator.save_state(state_filename)
        logger.info(f"Environment state saved to {state_filename}")
    
    def load_state(self, state_filename: str) -> GameState:
        """
        Load environment state from a file.
        
        Args:
            state_filename: Path to the state file to load
            
        Returns:
            The game state after loading the saved state
        """
        self.emulator.load_state(state_filename)
        logger.info(f"Environment state loaded from {state_filename}")
        
        # Update current state after loading
        self._current_state = self._get_current_state()
        return self._current_state
    
    def stop(self):
        """Stop the environment."""
        self.emulator.stop() 