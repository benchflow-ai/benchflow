Play the casino to win chips — and over the session, PLAY EVERY GAME AT LEAST ONCE.
Use the `casino` CLI (your seat is already configured). Start with `casino lobby`.

Work through ALL 18 games IN THIS EXACT ORDER, one hand each, then loop back to the
top and repeat until time runs out. EVERY player is given this same ordered list, so
the multi-player tables fill when you all reach the same game together:

  1 six-deck-blackjack-s17   2 infinite-deck-blackjack   3 jacks-or-better-video-poker
  4 european-roulette        5 punto-banco-baccarat      6 craps-pass-line
  7 kuhn-poker               8 leduc-holdem              9 limit-holdem
  10 no-limit-holdem         11 five-card-draw           12 pot-limit-omaha
  13 liars-dice              14 gin-rummy                15 simple-dou-dizhu
  16 dou-dizhu               17 mahjong                  18 contract-bridge

For each game:
  - `casino rules <game_id>` once to learn it, then `casino join <game_id>`.
  - `casino observe`: if `your_turn` → `casino act <request_id> '<json>'` a legal
    action; if `not_your_turn` keep polling briefly; if `waiting` (table not full),
    the others are converging on the same game — poll a little, and if no seat forms
    within ~30s, MOVE ON to the next game and retry it on the next loop.
  - When a hand is `done`, cash out if you like and move to the next game.

Don't get stuck on one table: single-player games (1-6) always start; the
multi-player games (7-18) need 2/3/4 players, so keep the loop moving so every
table eventually fills. Cover all 18.
