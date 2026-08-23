# Gomoku MCTS/RL Engine — Design

Date: 2026-08-23

## 1. Purpose

A self-contained Gomoku (five-in-a-row) system with three parts:

1. A game environment plus a terminal UI supporting human-vs-computer and
   computer-vs-computer play.
2. An AlphaZero-style trainer: a policy/value network guided by PUCT Monte Carlo
   tree search, learning purely from self-play, running on Apple Silicon via the
   PyTorch MPS backend.
3. Five selectable difficulty levels, each labelled with an ELO rating measured
   by offline round-robin play rather than assigned by hand.

Success criteria:

- The TUI is playable end-to-end against a bot before any network exists.
- The training pipeline is proven correct by an integration test that learns a
  3x3, three-in-a-row variant to near-perfect play in seconds.
- Level 5 beats the heuristic anchor bot decisively; levels 1 through 5 form a
  monotonically increasing rating ladder with gaps of at least ~150 ELO.
- Self-play throughput is high enough that an overnight run on this machine
  produces a bot that beats a casual human player on a 9x9 board.

## 2. Scope and rules

- **Rules: freestyle.** Five or more stones in a row wins. No forbidden moves,
  no opening balance rule, no swap. Black moves first.
- **Board size is a parameter.** Default 9x9. 15x15 is supported by every
  component and is the eventual target; 9x9 is where training is validated
  because it is roughly 10-20x cheaper per game.
- Draw when the board fills with no five-in-a-row.

Explicitly out of scope: Renju forbidden moves, swap2, pondering on the
opponent's clock, opening books, network play, GUI.

## 3. First-player advantage

Freestyle Gomoku is a first-player win with perfect play. Left untreated this
degrades training in three ways:

1. **Value-head parity shortcut.** The value target `z` is from the perspective
   of the side to move. If black wins most self-play games, the network can
   minimise value loss by emitting a near-constant positive value whenever it is
   black to move, ignoring the position. This is the fastest available loss
   reduction early in training and it starves the positional signal.
2. **Uninformative Q values.** A near-constant value head makes the PUCT Q term
   carry no information, so search degenerates toward following the policy prior
   plus whatever terminal results fall inside its horizon.
3. **Self-play diversity collapse.** Once black finds one attack white cannot
   refute, games become near-deterministic, the replay buffer narrows to a
   single line, and generalisation stalls. This is the true "local minimum"
   risk, and it is an exploration failure rather than an optimisation one.

None of these prevent learning — Connect-4 is likewise a first-player win and
trains fine — but untreated they waste most of an overnight run.

### Mitigations (all required)

- **Randomised multi-ply openings.** Each self-play game begins with 2-4 random
  plies drawn from cells near the centre. This is the measure that actually
  decorrelates the outcome from side-to-move: a random opening frequently leaves
  *white* better, so `z` stops being predictable from parity alone and the value
  head has to read the position. Uniform play from the empty board would not
  achieve this, because black simply wins from there.
- **Dirichlet noise at the root**, epsilon = 0.25, alpha ~= 10 / (average legal
  moves) -- approximately 0.15 for 9x9, 0.05 for 15x15.
- **Temperature 1.0 for the first N plies**, where N ~= the board dimension, then
  temperature -> 0 (argmax) for the remainder of the game.
- **Eight-fold dihedral augmentation** of every training sample.
- **Colour-paired matches in the arena.** Every pairing plays each sampled
  opening once from each side. This does not belong in self-play -- there both
  players are the same network and the encoding is already side-to-move
  relative, so swapping colours yields either the identical encoded position or
  an illegal one. In the arena the two players differ, and pairing is what keeps
  colour bias out of the measured ratings.

### Efficiency measure

- **Playout-cap randomisation.** Most moves are searched at a low simulation
  count (~100); a sampled fraction (~25%) uses the full count (~600). Policy
  targets are recorded only from full-count moves. This yields roughly 3x more
  games per hour at equivalent target quality and is the single highest-value
  knob on a laptop.

### Diagnostics

Each training generation logs, so that the failure modes above are observable
rather than assumed away:

- **Black win rate** — expected high and stable (70-95%).
- **Value loss versus two baselines**: a constant predictor (the variance of
  `z`), and a parity predictor that outputs the mean `z` conditioned on
  side-to-move. Failing to beat the parity baseline is the exact signature of
  the shortcut, and is the measurement that matters.
- **Policy entropy** — an early collapse toward zero means exploration died.
- **Distinct opening count and game-length histogram** — narrowing indicates
  diversity collapse.

## 4. Architecture

```
gomoku/
  board.py       # numpy board, incremental win detection, legal moves, dihedral symmetry
  game.py        # GameState: step, terminal test, encoding to (C,H,W) planes
  net.py         # PolicyValueNet: residual trunk, policy head, value head; device selection
  evaluator.py   # Evaluator interface; BatchedEvaluator; ServerEvaluator (phase 2)
  mcts.py        # PUCT search: virtual loss, root Dirichlet noise, batched leaf collection
  players.py     # Player interface: Human, Random, Heuristic (anchor), MCTS
  difficulty.py  # five levels -> (sims, temperature, policy-only flag, ELO) from config
  selfplay.py    # game generation -> (state, pi, z) samples; color pairing; playout caps
  replay.py      # replay buffer with on-disk shards, resumable
  train.py       # training loop, checkpointing, resume, JSONL metrics
  arena.py       # head-to-head matches, round-robin, logistic-MLE rating fit -> elo.json
  tui/
    app.py       # Textual application: board grid, cursor, status pane, level picker
    render.py    # board rendering, last-move and winning-line highlight
  cli.py         # gomoku play | train | selfplay | arena
tests/
```

### Component contracts

**`board.Board`** — owns the stone array and nothing else. Exposes
`legal_moves()`, `play(move)`, `winner`, and the eight dihedral transforms.
Win detection scans only the four lines through the last move placed, making it
O(board dimension) per move rather than O(cells).

**`game.GameState`** — wraps a `Board` with side-to-move and move history, and
produces the network input encoding. Four planes: current player's stones,
opponent's stones, a one-hot last-move indicator, and a constant side-to-move
plane. The network is fully convolutional with global pooling on the heads, so a
network trained at 9x9 can be fine-tuned at 15x15 without changing shapes.

**`net.PolicyValueNet`** — a residual trunk (default six blocks, 64 channels)
with a policy head over board cells and a `tanh` value head. Device is selected
as MPS when available, otherwise CPU.

**`evaluator.Evaluator`** — the single boundary between search and hardware.
`evaluate(states) -> (policies, values)` for a batch. Two implementations:

- `BatchedEvaluator`: in-process. Many self-play games advance concurrently;
  their pending MCTS leaves are gathered into one batch per forward pass.
- `ServerEvaluator` (phase 2): worker processes run MCTS trees and send leaves
  over a queue to a single inference process holding the model on MPS.

This boundary is the reason the phasing works. On MPS a forward pass is
latency-bound at roughly 1-2 ms regardless of batch size up to ~256, so
evaluating leaves one at a time costs about 60 seconds per game — far too slow.
Batching recovers 50-100x evaluation throughput for about 2x latency, at which
point single-core Python tree search becomes the bottleneck; the multiprocess
evaluator then removes that too. Phase 1 ships `BatchedEvaluator`, which is
sufficient to validate that learning works. `ServerEvaluator` is a drop-in
substitution afterwards, with no change to MCTS or the training loop.

**`mcts.MCTS`** — PUCT with virtual loss so several leaves can be in flight per
tree, root Dirichlet noise, and a batched collect-then-evaluate loop. Returns
visit-count distributions used both to pick moves and as policy targets.

**`players.Player`** — `select_move(state) -> move`. `HeuristicPlayer` is a
threat-based rule bot (build and block open threes and fours, otherwise prefer
proximity to existing stones); it is both the pre-network opponent for the TUI
and the fixed rating anchor.

**`difficulty`** — five levels driven entirely by configuration over one trained
checkpoint. Level 1 is policy-only with high temperature. Levels 2 through 5 use
MCTS at 25, 100, 400 and 1600 simulations with temperature falling to zero.

**`arena`** — plays color-paired round-robin matches among the five levels plus
the anchor bot, fits ratings by logistic maximum likelihood (with a weak Gaussian prior so
undefeated players stay finite) and the anchor pinned at 1200 ELO, and writes `elo.json`. The TUI displays those measured
numbers.

## 5. Data flow

Self-play: `GameState` → `MCTS` → (via `Evaluator`) → `PolicyValueNet`; visit
counts become `pi`, the final result becomes `z`, and `(state, pi, z)` triples
with their eight symmetries enter the replay buffer. Training samples the buffer,
minimising cross-entropy on the policy plus mean squared error on the value plus
weight decay, then checkpoints. The next generation of self-play uses the new
checkpoint.

Play: the TUI constructs two `Player` objects from the chosen mode and level and
alternates `select_move` calls, redrawing after each move.

## 6. Error handling

- Illegal or out-of-bounds moves raise immediately; the TUI rejects them at the
  input layer rather than relying on the exception.
- Checkpoints are written atomically (temporary file, then rename) and record
  generation, configuration and optimiser state, so an interrupted multi-day run
  resumes exactly.
- Replay shards are validated on load; a corrupt shard is skipped with a warning
  rather than aborting the run.
- If MPS is unavailable the system falls back to CPU with a logged notice; no
  code path requires MPS.
- Missing `elo.json` makes the TUI show levels without ratings, rather than
  failing or inventing numbers.

## 7. Testing

Test-driven throughout. The suite is fast enough to run on every change:

- **Board**: win detection along all four directions including board edges,
  overline (six in a row) counting as a win under freestyle, draw on a full
  board, symmetry transforms being involutive and permuting moves consistently.
- **MCTS**: visit counts concentrate on a forced win; virtual loss does not
  corrupt statistics; Dirichlet noise appears only at the root; a search with a
  perfect evaluator picks the winning move.
- **Encoding**: planes round-trip; the side-to-move plane flips correctly.
- **Self-play**: random openings are legal, of the configured length, and
  diverse across seeds; playout-cap randomisation records targets only from
  full-count moves; the temperature schedule switches at the configured ply.
- **Arena**: colour pairing plays each opening once from each side.
- **Arena**: the rating fit recovers known ratings from synthetic results.
- **Integration**: a 3x3 board with a three-in-a-row win condition trains to
  near-perfect play within seconds, exercising the entire pipeline — self-play,
  buffer, training, checkpoint, arena — without waiting on a real run.

## 8. Build order

1. Board, game state, encoding, symmetry — with tests.
2. Heuristic and random players; the Textual TUI, playable immediately.
3. Network, evaluator interface, `BatchedEvaluator`, MCTS.
4. Self-play with the mitigations, replay buffer, training loop, diagnostics.
5. The 3x3 integration test; then a real 9x9 training run.
6. Arena, rating fit, difficulty levels wired into the TUI.
7. `ServerEvaluator` for multi-day throughput.
