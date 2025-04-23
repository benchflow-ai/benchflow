from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class ActionType(Enum):
    PRESS_KEY = "press_key"
    WAIT = "wait"


class Action(ABC):
    """Base class for all actions that can be performed in the Pokemon environment."""
    
    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """Returns the type of this action."""
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Converts this action to a dictionary representation."""
        pass


class PressKey(Action):
    """Action to press one or more keys on the Game Boy."""
    
    VALID_KEYS = ["a", "b", "start", "select", "up", "down", "left", "right"]
    
    def __init__(self, keys: List[str]):
        """
        Initialize a press key action.
        
        Args:
            keys: List of keys to press in sequence
        """
        # Validate keys
        for key in keys:
            if key not in self.VALID_KEYS:
                raise ValueError(f"Invalid key: {key}. Valid keys are: {self.VALID_KEYS}")
        
        self.keys = keys
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.PRESS_KEY
    
    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "keys": self.keys
        }
    
    def __str__(self) -> str:
        return f"PressKey: {', '.join(self.keys)}"


class Wait(Action):
    """Action to wait for a specified number of frames."""
    
    def __init__(self, frames: int = 60):
        """
        Initialize a wait action.
        
        Args:
            frames: Number of frames to wait (default: 60)
        """
        if frames <= 0:
            raise ValueError("Frames must be a positive integer")
        
        self.frames = frames
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.WAIT
    
    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "frames": self.frames
        }
    
    def __str__(self) -> str:
        return f"Wait: {self.frames} frames" 