# Gomoku MCTS

An AlphaZero-style Gomoku (five-in-a-row) engine: a policy/value network guided
by PUCT Monte Carlo tree search, trained purely from self-play, with a terminal
UI and five difficulty levels whose ELO ratings are **measured** rather than
assigned.

Runs on Apple Silicon via the PyTorch MPS backend, and falls back to CPU
everywhere — no code path requires a GPU.

```
   A B C D E F G H I
 1 . . . . . . . . .
 2 . . . . . . . . .
 3 . . . O . . . . .
 4 . . . X X X . . .
 5 . . . . O[X]. . .
 6 . . . O . . . . .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . . . . . . . . .
```

## Included model

`runs/9x9/checkpoint.pt` is a trained 9×9 model: 200 generations of self-play,
about 2.6 hours on an M-series Mac. A 6-block, 64-channel residual network.

Its ratings, measured by the arena in `runs/elo.json`:

| Level | Name     | Simulations | Temperature | Measured ELO | Gap |
|------:|----------|------------:|------------:|-------------:|----:|
| 1     | Beginner | 0 (policy only) | 1.0     |  950 |     |
| 2     | Casual   |          50 | 0.5         | 1204 | +254 |
| —     | *heuristic anchor* | — | —         | *1200* | |
| 3     | Club     |         100 | 0.3         | 1371 | +167 |
| 4     | Strong   |         400 | 0.0         | 1476 | +105 |
| 5     | Expert   |        1600 | 0.0         | 1522 |  +46 |

The ratings are anchored by pinning a fixed rule-based bot at a nominal 1200,
so they are comparable across runs but are not calibrated against human ratings.
Level 1 is weaker than that rule-based bot, level 2 is about even with it (they
drew 10–10), and levels 3–5 beat it.

**Known limitation: the top of the ladder is compressed.** The 3→4 and 4→5 steps
are 105 and 46 ELO against a design target of roughly 150. This is search
saturation rather than a tuning mistake: each 4× increase in simulations buys
less than the one before (100→400 gains 105, 400→1600 gains 46), because
additional search cannot outrun the quality of the network's own evaluation.
Level 5 takes only ~0.5s per move, so there is compute headroom — but widening
that gap meaningfully needs a better-trained network, not a bigger budget.

## Install

Requires Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Play

Play the included model straight away:

```bash
.venv/bin/gomoku play --checkpoint runs/9x9/checkpoint.pt --elo runs/elo.json --level 3
```

With no checkpoint it falls back to the rule-based bot, so the game is playable
before you train anything:

```bash
.venv/bin/gomoku play
```

| Key | Action |
|-----|--------|
| arrows | move the cursor |
| enter / space | place a stone |
| `1`–`5` | switch difficulty (starts a new game) |
| `n` | new game |
| `q` | quit |

Watch the engine play itself with `--mode pc-vs-pc`. Board size is a parameter
throughout: `--size 15` works, though the included model was trained at 9×9.

## Train

```bash
# A short run first — check the diagnostics before committing hours.
.venv/bin/gomoku train --run-dir runs/mine --size 9 --generations 5 --games 64

# Then the long run. Interrupt and re-run to resume from the checkpoint.
.venv/bin/gomoku train --run-dir runs/mine --generations 200 --games 128 --workers 6
```

`--workers N` moves self-play into N processes that share one batched inference
server; leave it at 0 to run in-process.

### Reading the diagnostics

Every generation appends a record to `<run-dir>/metrics.jsonl`. Four of its
fields exist to catch failures that a falling loss curve will not show you:

```bash
.venv/bin/python -c "
from gomoku.metrics import MetricsWriter
for r in MetricsWriter('runs/mine/metrics.jsonl').read_all():
    print(r['generation'], round(r['value_loss_on_fresh'], 3),
          'vs parity', round(r['value_baseline_parity'], 3),
          '| entropy', round(r['policy_entropy'], 2),
          '| prefixes', r['distinct_play_prefixes'])
"
```

- **`value_loss_on_fresh` must fall below `value_baseline_parity`.** The parity
  baseline is what a value head scores by predicting the average outcome *for
  each colour* — that is, by knowing only that black tends to win. Failing to
  beat it means the value head has learned the first-player advantage and
  nothing about the position. Both numbers are computed on positions held out
  of that generation's training batches; when a generation is too small to hold
  any out, the log says so rather than reporting a flattering in-sample number.
- **`policy_entropy`** should decline gradually. A collapse toward zero means
  exploration died.
- **`distinct_play_prefixes`** counts distinct opening sequences the *policy*
  chose. If it falls while games are still being played, self-play has narrowed
  to a single line.
- **`black_win_rate`** should be high but stable — this is a first-player-win
  game.

For reference, the included model's final generation: value loss **0.316**
against a parity baseline of **0.414** and a constant baseline of 0.580; entropy
1.89; 128 distinct prefixes across 128 games.

## Measure ratings

```bash
.venv/bin/gomoku arena --checkpoint runs/mine/checkpoint.pt --out runs/mine/elo.json \
    --games-per-pair 20
```

Every pairing plays each sampled opening twice with the colours swapped, because
the first-player advantage is large enough that unpaired results would mostly
measure who drew black. Ratings are fit by Bradley-Terry maximum likelihood with
a weak Gaussian prior (so an undefeated player's rating stays finite), then
translated to pin the heuristic bot at 1200.

The written file records the checkpoint generation and board size. `gomoku play`
refuses to display ratings measured on a different board, so a 9×9 rating never
appears next to a 15×15 game.

## How it works

| Module | Responsibility |
|--------|----------------|
| `board.py` | stones, legality, incremental win detection |
| `game.py` | immutable `GameState`, 4-plane encoding |
| `symmetry.py` | the eight dihedral transforms |
| `net.py` | residual policy/value network, checkpoint I/O |
| `evaluator.py` | the only boundary between search and hardware |
| `mcts.py` | PUCT search, virtual loss, batched leaf evaluation |
| `selfplay.py` | game generation and training targets |
| `replay.py` | bounded sample window with resumable shards |
| `train.py` | the training loop and its diagnostics |
| `arena.py` | colour-paired matches, Bradley-Terry rating fit |
| `difficulty.py` | five levels over one checkpoint |
| `tui/` | the Textual interface |

Two choices are worth calling out.

**The network is board-size agnostic.** The policy head is a 1×1 convolution and
the value head reduces by global average pooling, so one set of weights accepts
any board size and a 9×9 checkpoint can be fine-tuned at 15×15.

**Search never imports torch.** It talks to hardware only through
`Evaluator.evaluate(states) -> (priors, values)`. That is what let the
multiprocess inference server drop in without a line changing in `mcts.py` or
`selfplay.py`.

## The first-player advantage

Freestyle Gomoku is a proven first-player win, which distorts self-play in ways
that do not announce themselves. The value target becomes nearly a function of
*whose turn it is* rather than of the position, so the fastest available loss
reduction is to output "black is winning" and stop reading the board. Games then
collapse toward a single line and the model stops generalising.

Four mitigations are built in and on by default:

- **Random 2–4 ply openings** near the centre. This is the one that actually
  decorrelates the outcome from side-to-move: a random opening frequently leaves
  *white* better. Playing from the empty board would not, because black simply
  wins from there.
- **Dirichlet noise at the search root**, including after the tree is re-rooted
  for the next move — otherwise noise reaches only move one of each game.
- **A temperature schedule** counted in plies *since the opening*.
- **Eight-fold dihedral augmentation** of every training sample.

Colour pairing deliberately appears only in the arena, never in self-play: there
both players are the same network and the encoding is already side-to-move
relative, so swapping colours would produce the same position back.

`docs/superpowers/specs/2026-08-23-gomoku-mcts-design.md` covers the reasoning in
full.

## Tests

```bash
.venv/bin/pytest -q          # 256 tests
.venv/bin/pytest -q -m "not slow"   # skip the training and multiprocess tests
```

The most valuable one is `tests/test_learning.py`: it trains a real model on a
3×3 board (tic-tac-toe) in about 20 seconds and asserts it learns — exercising
self-play, augmentation, the replay buffer, the training loop, checkpointing and
search on the same code paths the 9×9 game uses. Four of its eight assertions
measure learning; the other four are search-dominated sanity checks, since 100
simulations solve tic-tac-toe almost regardless of network quality.

## Scope

Freestyle rules: five *or more* in a row wins, no forbidden moves, no opening
balance rule. Renju, swap2, pondering, opening books and network play are out of
scope.

## License

MIT — see [LICENSE](LICENSE).
