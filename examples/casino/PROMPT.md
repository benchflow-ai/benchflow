Play the casino games and win as many chips as you can, using the `casino`
command (your seat is already configured):

  casino lobby                       — open games, your bankroll, queue state
  casino rules <game_id>             — a game's rules
  casino join <game_id>              — take a seat (or queue) at a game
  casino observe [--wait N]          — {request_id, observation, legal_actions, events}
  casino act <request_id> '<json>'   — play ONE of the legal actions
  casino cashier                     — your bankroll
  casino leave                       — leave your table or queue

House etiquette (enforced by the casino):
- To wait for your turn or for opponents, use `casino observe --wait 30` —
  it blocks until something happens. Never busy-loop plain observe.
- If a game queue hasn't matched after a couple of minutes, `casino leave`
  and pick another game (the casino will also time you out of stale queues).
- If you sit silent on your turn too long the casino plays a default action
  for you; repeated silence sits you out. You may stop playing at any time —
  say so and stop.

Play through the `casino` CLI. Begin with `casino lobby`.
