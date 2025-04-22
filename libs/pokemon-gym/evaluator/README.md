# Pokemon Evaluator

This module provides evaluation metrics and scoring system for Pokemon Red gameplay. It works in conjunction with the server to track and assess player performance.

## Components

The evaluator consists of these key components:

- `evaluate.py`: Implementation of the evaluation metrics
- `milestones.py`: Definition of game milestones and scoring rules

## Evaluation System

The evaluation system scores gameplay based on:

- **Pokemon Collection**: Points for each unique Pokemon caught
- **Badges Earned**: Points for each gym badge obtained 
- **Locations Visited**: Points for discovering new areas
- **Milestone Events**: Points for key game events (getting first Pokemon, defeating rivals, etc.)

The evaluation state persists across saved/loaded states and continued sessions.

## Running with the Server

To use the evaluator with the server:

```bash
python -m server.evaluator_server
```

The server will start at http://localhost:8080 by default. See the [Server Documentation](../server/README.md) for more details on the API endpoints and server configuration.

## Milestone Events

The evaluator tracks various milestone events in the game, including:

- Starting the game
- Getting your first Pokemon
- Defeating your rival
- Obtaining gym badges
- Reaching new locations
- Catching different Pokemon
- Progressing through the story

Each milestone awards points that contribute to the player's total score.

## Usage in Code

The `PokemonEvaluator` class can be used to evaluate gameplay:

```python
from evaluator.evaluate import PokemonEvaluator

# Create an evaluator instance
evaluator = PokemonEvaluator()

# Update with new game state
evaluator.update(memory_info, location, party)

# Get current score
score = evaluator.get_score()

# Get detailed evaluation
evaluation = evaluator.get_evaluation()
print(f"Total score: {evaluation['total_score']}")
print(f"Pokemon caught: {evaluation['pokemon_caught']}")
print(f"Badges obtained: {evaluation['badges_obtained']}")
```

## Extending the Evaluator

To add new milestone events or scoring criteria:

1. Define new milestone events in `milestones.py`
2. Update the detection logic in `evaluate.py`

## Session Management

The evaluator maintains state between sessions, storing:

- Current progress through the game
- Achieved milestones and scores
- Pokemon caught and badges obtained

This allows for continuous evaluation across multiple gameplay sessions. 