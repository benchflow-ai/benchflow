import os
import sys
from pathlib import Path

def find_repo_root():
    """
    Find the repository root by looking for the .env file or the examples directory.
    Returns the absolute path to the repository root.
    """
    # Start with the current working directory
    current_dir = Path(os.getcwd()).absolute()
    
    # Check if we're already at the repo root (has .env or examples dir)
    if (current_dir / '.env').exists() or (current_dir / 'examples').is_dir():
        return current_dir
    
    # Try to find the repo root by going up the directory tree
    for parent in current_dir.parents:
        if (parent / '.env').exists() or (parent / 'examples').is_dir():
            return parent
    
    # If we can't find it, use the current directory as a fallback
    return current_dir

def get_env_path():
    """
    Get the absolute path to the .env file.
    """
    repo_root = find_repo_root()
    return repo_root / '.env'

def get_examples_dir():
    """
    Get the absolute path to the examples directory.
    """
    repo_root = find_repo_root()
    return repo_root / 'examples'

def get_requirements_path(relative_path):
    """
    Get the absolute path to a requirements file.
    
    Args:
        relative_path: The path relative to the repo root, e.g., 'examples/rarebench/rarebench_requirements.txt'
    """
    repo_root = find_repo_root()
    return repo_root / relative_path

def add_repo_to_path():
    """
    Add the repository root to the Python path if it's not already there.
    This allows imports from the repository root.
    """
    repo_root = str(find_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
