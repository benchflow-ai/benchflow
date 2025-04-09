import os
import dotenv
from pathlib import Path

def load_env_vars():
    """Load environment variables from .env file."""
    # Try to find .env file in the current directory or parent directories
    env_path = Path('.env')
    if not env_path.exists():
        # Try the project root directory
        root_env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env_path = root_env_path / '.env'
    
    if env_path.exists():
        print(f"Loading environment variables from {env_path}")
        dotenv.load_dotenv(env_path)
        
        # Check if the required API keys are loaded
        keys = {
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
            "BF_TOKEN": os.getenv("BF_TOKEN")
        }
        
        # Print which keys are available (without showing the actual keys)
        print("Available API keys:")
        for key_name, key_value in keys.items():
            print(f"  {key_name}: {'✓ Available' if key_value else '✗ Not found'}")
            
        return True
    else:
        print("Warning: .env file not found")
        return False

if __name__ == "__main__":
    # Test the function
    load_env_vars()
