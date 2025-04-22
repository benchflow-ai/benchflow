import argparse
import base64
import io
import logging
import os
import time
import requests
import threading
import pygame
from PIL import Image
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define key mappings for Pokemon Red
KEY_MAPPING = {
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_z: "a",      # Z key for A button
    pygame.K_x: "b",      # X key for B button
    pygame.K_RETURN: "start",  # Enter key for Start button
    pygame.K_RSHIFT: "select",  # Right Shift for Select button
    pygame.K_SPACE: "wait",  # Space for wait command
    pygame.K_F5: "save",    # F5 to save state
    pygame.K_F7: "load",    # F7 to load saved state
}

# Number of frames to wait when using the wait command
DEFAULT_WAIT_FRAMES = 30
DEFAULT_SAVE_FILENAME = "manual_save.state"

class HumanAgent:
    """Human Agent that allows keyboard control of Pokemon Red through the evaluator server"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        """
        Initialize the Human Agent
        
        Args:
            server_url: URL of the evaluation server
        """
        self.server_url = server_url
        self.session = requests.Session()
        self.initialized = False
        self.current_state = None
        self.running = True
        self.step_count = 0
        self.score = 0.0
        self.saved_state_path = None
        
        # Initialize pygame for display and keyboard input
        pygame.init()
        pygame.display.set_caption("Pokemon Red - Human Control")
        
        # Game window dimensions
        self.screen_width = 480  # 160 * 3
        self.screen_height = 432  # 144 * 3
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # For showing info text
        self.font = pygame.font.SysFont('Arial', 14)
        
    def initialize(self, headless: bool = False, sound: bool = False, 
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
            
            # Set initial score
            if 'score' in self.current_state:
                self.score = self.current_state['score']
            
            # Set the initial saved state path if loading from a file
            if load_state_file:
                self.saved_state_path = load_state_file
            
            logger.info(f"Initialization successful, location: {self.current_state['location']}")
            
            # Display the initial state
            self.update_display(self.current_state)
            
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
            
            # Update score
            if 'score' in self.current_state:
                self.score = self.current_state['score']
            
            # Update the display with the new state
            self.update_display(self.current_state)
            
            return self.current_state
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Action execution error: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Server response: {e.response.text}")
            raise
    
    def save_state(self, filename: str = None) -> Dict[str, Any]:
        """
        Save the current game state
        
        Args:
            filename: Optional custom filename for the state
            
        Returns:
            Response from the server
        """
        if not self.initialized:
            raise RuntimeError("Environment not initialized")
        
        try:
            # Prepare request data
            request_data = {}
            if filename:
                request_data["filename"] = filename
            
            # Send request
            response = self.session.post(
                f"{self.server_url}/save_state",
                headers={"Content-Type": "application/json"},
                json=request_data
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Store the saved state path
            if 'state_file' in result:
                self.saved_state_path = result['state_file']
                logger.info(f"Game state saved to {self.saved_state_path}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Save state error: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Server response: {e.response.text}")
            raise

    def update_display(self, state: Dict[str, Any]) -> None:
        """
        Update the pygame display with the current game state
        
        Args:
            state: Current game state
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(state['screenshot_base64'])
            image = Image.open(io.BytesIO(image_data))
            
            # Resize image for display
            image = image.resize((self.screen_width, self.screen_height), Image.NEAREST)
            
            # Convert PIL Image to pygame surface
            mode = image.mode
            size = image.size
            data = image.tobytes()
            
            pygame_image = pygame.image.fromstring(data, size, mode)
            self.screen.blit(pygame_image, (0, 0))
            
            # Display information text
            info_text = f"Location: {state['location']} | Coordinates: {state['coordinates']} | Step: {self.step_count} | Score: {self.score:.1f}"
            location_text = self.font.render(info_text, True, (255, 255, 255))
            
            # Add a semi-transparent background for text readability
            text_bg = pygame.Surface((location_text.get_width(), location_text.get_height()))
            text_bg.set_alpha(128)
            text_bg.fill((0, 0, 0))
            self.screen.blit(text_bg, (5, 5))
            self.screen.blit(location_text, (5, 5))
            
            # Add control instructions at the bottom
            controls_text = self.font.render("Controls: Arrows = Move | Z = A | X = B | Enter = Start | R-Shift = Select | Space = Wait | F5 = Save | F7 = Load", True, (255, 255, 255))
            controls_bg = pygame.Surface((controls_text.get_width(), controls_text.get_height()))
            controls_bg.set_alpha(128)
            controls_bg.fill((0, 0, 0))
            self.screen.blit(controls_bg, (5, self.screen_height - 20))
            self.screen.blit(controls_text, (5, self.screen_height - 20))
            
            pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Display update error: {e}")
    
    def run(self) -> None:
        """
        Run the Human Agent, handling keyboard input
        """
        if not self.initialized:
            raise RuntimeError("Environment not initialized, please call initialize() first")
        
        logger.info("Starting Human Agent, press ESC to quit")
        
        try:
            clock = pygame.time.Clock()
            
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key in KEY_MAPPING:
                            key = KEY_MAPPING[event.key]
                            
                            if key == "wait":
                                # Execute wait action
                                logger.info(f"Waiting for {DEFAULT_WAIT_FRAMES} frames")
                                self.take_action("wait", frames=DEFAULT_WAIT_FRAMES)
                            elif key == "save":
                                # Save game state
                                logger.info("Saving game state...")
                                self.save_state(DEFAULT_SAVE_FILENAME)
                            elif key == "load":
                                # Load saved game state
                                if self.saved_state_path:
                                    logger.info(f"Loading game state from {self.saved_state_path}...")
                                    self.initialize(headless=False, sound=True, load_state_file=self.saved_state_path)
                                else:
                                    logger.warning("No saved state available to load")
                            else:
                                # Execute press key action
                                logger.info(f"Pressing key: {key}")
                                self.take_action("press_key", keys=[key])
                
                # Cap the frame rate
                clock.tick(30)
        
        except KeyboardInterrupt:
            logger.info("User interrupted, stopping run")
        except Exception as e:
            logger.error(f"Run error: {e}")
        finally:
            self.stop()
            pygame.quit()
    
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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Pokemon Human Agent")
    parser.add_argument("--server", type=str, default="http://localhost:8080", help="Evaluation server URL")
    parser.add_argument("--sound", action="store_true", help="Enable sound")
    parser.add_argument("--load-state", type=str, help="Path to a saved state file to load")
    parser.add_argument("--load-autosave", action="store_true", help="Load the latest autosave")
    parser.add_argument("--session", type=str, help="Session ID to continue (e.g., session_20250404_180209)")
    
    args = parser.parse_args()
    
    # Create Human Agent
    agent = HumanAgent(server_url=args.server)
    
    try:
        # Initialize environment (never headless for human agent)
        agent.initialize(
            headless=False, 
            sound=args.sound,
            load_state_file=args.load_state,
            load_autosave=args.load_autosave,
            session_id=args.session
        )
        
        # Run Human Agent
        agent.run()
        
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
