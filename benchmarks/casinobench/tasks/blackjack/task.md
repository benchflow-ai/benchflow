---
schema_version: '1.0'
metadata:
  author_name: benchflow
  category: games
  tags:
  - casino
  - casinobench
  - house
  - blackjack
# The in-sandbox env-0 service is provisioned by the Environment plane from the
# manifest below; the game is selected by CASINOBENCH_GAME (env_var task_selection).
benchflow:
  environment:
    manifest: ../../environment.toml
agent:
  timeout_sec: 1800.0
verifier:
  type: test-script
  timeout_sec: 120.0
  service: main
  # reward = net chips (final - starting): did-nothing -> 0, loss < 0, profit > 0.
  reward_range: [-1000.0, 5000.0]
environment:
  network_mode: public
  cpus: 1
  memory_mb: 2048
  env:
    CASINOBENCH_GAME: 'six-deck-blackjack-s17'
    CASINOBENCH_HANDS: '10'
oracle:
  env: {}
---

## prompt

You have a seat at the CasinoBench casino. Play **blackjack** (the
`six-deck-blackjack-s17` table: six decks, dealer stands on all 17, double-down
allowed) and finish with as many chips as you can.

The casino runs as a mock service inside this sandbox; reach it with the
`casino` CLI (the seven tools):

    casino lobby                            # open games, tables, your bankroll
    casino rules six-deck-blackjack-s17     # the rules
    casino join six-deck-blackjack-s17      # take your seat
    # then repeat until observe reports "done": true
    casino observe                          # request_id + your view + legal actions
    casino act <request_id> '<one legal action, copied from legal_actions>'
    casino cashier                          # your bankroll and realized PnL

Choose actions only from the `legal_actions` returned by `casino observe`. Your
score is your net chips at the end, so play to win and manage risk.
