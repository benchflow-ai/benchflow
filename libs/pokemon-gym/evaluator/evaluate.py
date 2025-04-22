import csv
import os
import ast
from .milestones import all_difficulty_ratings

class PokemonEvaluator:
    """
    Class for evaluating Pokemon game progress and calculating scores
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        # Initialize sets to track achievements
        self.pokemon_seen = set()
        self.badges_earned = set()
        self.locations_visited = set()
        self.total_score = 0.0
    
    def evaluate_pokemon(self, pokemon_name):
        """Evaluate a new Pokemon"""
        if pokemon_name and pokemon_name not in self.pokemon_seen:
            if pokemon_name in all_difficulty_ratings:
                pokemon_score = all_difficulty_ratings[pokemon_name]
                self.total_score += pokemon_score
                self.pokemon_seen.add(pokemon_name)
                print(f"New Pokemon: {pokemon_name}, Score: +{pokemon_score}")
    
    def evaluate_badge(self, badge_name):
        """Evaluate a new badge"""
        if badge_name and badge_name not in self.badges_earned:
            if badge_name in all_difficulty_ratings:
                badge_score = all_difficulty_ratings[badge_name]
                self.total_score += badge_score
                self.badges_earned.add(badge_name)
                print(f"New Badge: {badge_name}, Score: +{badge_score}")
    
    def evaluate_location(self, location_name):
        """Evaluate a new location"""
        location_name = location_name.replace(" ", "_")
        if location_name and location_name not in self.locations_visited:
            if location_name in all_difficulty_ratings:
                location_score = all_difficulty_ratings[location_name]
                self.total_score += location_score
                self.locations_visited.add(location_name)
                print(f"New Location: {location_name}, Score: +{location_score}")
    
    def evaluate_row(self, row):
        """Evaluate a single row from the CSV data"""
        # Check for Pokemon
        if row.get('pokemons'):
            try:
                pokemons_str = row.get('pokemons', '[]')
                pokemon_list = ast.literal_eval(pokemons_str)
                
                for pokemon_dict in pokemon_list:
                    if isinstance(pokemon_dict, dict) and 'species' in pokemon_dict:
                        self.evaluate_pokemon(pokemon_dict['species'])
            except Exception as e:
                print(f"Error parsing Pokemon data: {e}")
        
        # Check for Badges
        if row.get('badges'):
            try:
                badges_str = row.get('badges', '[]')
                badge_list = ast.literal_eval(badges_str)
                for badge in badge_list:
                    self.evaluate_badge(badge)
            except Exception as e:
                print(f"Error parsing Badge data: {e}")
        
        # Check for Locations
        if row.get('location'):
            try:
                location = row.get('location')
                self.evaluate_location(location)
            except Exception as e:
                print(f"Error parsing Location data: {e}")
    
    def evaluate_csv(self, csv_file_path):
        """
        Evaluate a Pokemon gameplay CSV file and calculate score based on milestones achieved
        
        Args:
            csv_file_path (str): Path to the CSV file to evaluate
        
        Returns:
            float: Total score achieved
        """
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at {csv_file_path}")
            return 0
        
        # Read the CSV file
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    self.evaluate_row(row)
        
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return 0
        
        # Print summary
        self.print_summary()
        
        return self.total_score
    
    def load_state_from_session(self, session_dir):
        """
        Load evaluation state from an existing session directory
        
        Args:
            session_dir (str): Path to the session directory
            
        Returns:
            bool: True if state was loaded successfully, False otherwise
        """
        # Reset current state
        self.reset()
        
        # Look for the gameplay data CSV file
        csv_path = os.path.join(session_dir, "gameplay_data.csv")
        if os.path.exists(csv_path):
            print(f"Loading evaluation state from {csv_path}")
            self.evaluate_csv(csv_path)
            return True
        else:
            print(f"No gameplay data file found in {session_dir}")
            return False
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n=== Evaluation Summary ===")
        print(f"Total Unique Pokemon: {len(self.pokemon_seen)}")
        print(f"Total Badges Earned: {len(self.badges_earned)}")
        print(f"Total Locations Visited: {len(self.locations_visited)}")
        print(f"Total Score: {self.total_score:.2f}")
    
    def reset(self):
        """Reset evaluator state"""
        self.pokemon_seen.clear()
        self.badges_earned.clear()
        self.locations_visited.clear()
        self.total_score = 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Pokemon gameplay CSV file")
    parser.add_argument("csv_file", help="Path to the gameplay CSV file to evaluate")
    
    args = parser.parse_args()
    
    evaluator = PokemonEvaluator()
    evaluator.evaluate_csv(args.csv_file)
