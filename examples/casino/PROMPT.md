Play the casino games and win as many chips as you can, using the `casino`
command (your seat is already configured):

  casino lobby                       — open games and your bankroll
  casino rules <game_id>             — a game's rules
  casino join <game_id>              — take a seat at a game
  casino observe                     — {request_id, observation, legal_actions, done}
  casino act <request_id> '<json>'   — play ONE of the legal actions
  casino cashier                     — your bankroll

Play through the `casino` CLI. Begin with `casino lobby`.
