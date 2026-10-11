# Live match review — 2026-10-11

## Scope

- Reviewed all 11 persisted match records from the MuMu session (383 actions).
- Six records use the current deck: Hog Rider, Electro Spirit, Knight, Cannon,
  Musketeer, Skeletons, Fireball and Zap (279 actions).
- One additional game from the five-game batch reached the result screen but
  was not persisted because result recognition failed; its live log was also
  checked for timing and placement failures.

## Main finding

The policy treated all eight cards as interchangeable cycle/offence cards.
Only 5 of 29 current-deck Hog deployments were followed by Electro Spirit in
the same lane within five seconds (17%). In three current-deck matches it never
formed the pair at all. It also opened empty boards with Knight, Musketeer,
Skeletons, Fireball or Zap, spending defensive resources before seeing the
opponent's push.

Examples include opening with Knight then Hog; Electro Spirit then Skeletons;
and Electro Spirit, Zap, Fireball, Skeletons in the first four seconds. Cannon
was frequently placed on the edge (most common cell `(1,19)`, nine times)
instead of the central pull pocket. This matches the visible failures: Hog into
defenders, empty spells, and no answer left for the counter-push.

## Changes derived from the review

- The only empty-board attack package is now Hog + Electro Spirit.
- Hog requires Electro Spirit in hand and at least five elixir; with a one-crown
  lead it waits for eight elixir.
- Electro Spirit is reserved on an empty board and may follow a living Hog only
  in the same lane. It remains available as a defender during a real push.
- Knight, Musketeer and Cannon are held for defence. Skeletons may cycle in the
  back only at full elixir when Hog is not available.
- Hog focuses the more damaged surviving princess tower instead of splitting
  damage. A one-tower lead shifts the policy toward defending the win.
- Cannon is constrained to the central pull pocket, shifted toward the attacked
  lane, and is blocked against a pure air push or an outranging-only push.
- Fireball targets low-HP/value clusters, defensive air, a finishing tower, or
  a building supporting the Hog lane. Zap requires a real light-unit cluster or
  an Inferno/Sparky reset.
- PPO now has an explicit, small same-lane Hog → Electro Spirit reward. Tower
  damage and winning remain dominant, so the combo is not rewarded if it does
  not create value.

## Perception defect found during review

Live unit coordinates were flipped twice before entering the policy network.
The detector had already converted bottom-up tiles into the top-down 18×32
policy grid, but the live bridge inverted them again. Thus the action guard saw
the correct side while the network could see the same push on the opposite half
of the arena. The second flip has been removed and simulator/live parity tests
now cover the corrected route.
