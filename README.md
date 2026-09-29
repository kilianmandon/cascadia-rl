# cascadia-rl

The board game [Cascadia](https://www.flatout.games/cascadia) as a Gymnasium
environment, with a transformer actor-critic trained on it using
action-masked PPO. Work in progress.

## Why

Cascadia is a hex-grid tile-laying game. Each turn you take one of four
habitat-tile/wildlife-token pairs from a shared pool, place the tile on the
frontier of your map at one of six rotations, and place the token on a tile
that accepts it. Scoring combines per-species wildlife patterns with contiguous
habitat corridors.

Two things make it awkward in the usual way: the action space is large and
almost entirely illegal at any moment (tile placement alone spans
`6 * 30 * 30` indices, of which only the few frontier hexes are valid), and the
board is a growing irregular region rather than a fixed grid, so a plain
convolutional encoder fits badly.

## Approach

The state is encoded as a set of tokens, one per occupied hex and one per
frontier hex, plus a global token for the pool, phase and held pair.
`ActorCriticTransformer` runs attention over these with a learned bias indexed
by the relative hex offset between token pairs, so nothing assumes a board
shape. Per-hex heads emit placement logits; the global token carries the value
estimate and the draft-phase logits. Everything is scattered into one flat
categorical over the action space, masked to legal actions before sampling.

Training is CleanRL's PPO with that masking, over asynchronous vectorised
environments.

## Results

Single-player game, average final score:

| Policy | Score |
| --- | --- |
| PPO (transformer) | 75 (n=100) |
| Random legal action | 42 (n=100) |
| Greedy, depth 3 | 41 (n=5) |

The greedy baseline is no better than random, which is less surprising than it
looks: it maximises immediate score delta, while most of Cascadia's points come
from corridors and wildlife patterns that only pay off much later.

## Usage

```bash
uv sync

# train
PYTHONPATH=src uv run python src/ppo.py

# playable UI
PYTHONPATH=src uv run python -m cascadia_ui --players 2
```

`src/ppo.py --help` lists all options, including toggles for which scoring
rules are active (habitat corridors, each animal, the largest-habitat bonus,
tile rotation) so the game can be simplified for faster experiments.

In the UI, hover a legal empty hex during land placement and use the mouse
wheel or `A` / `D` to rotate the preview; the arrow keys step through game
history. To watch a policy play, pass a callable that takes a deep-copied
`GameState` and returns an `Action`:

```python
from cascadia_ui import launch

launch(players=2, policies={1: lambda state: my_policy(state)})
```

## Layout

`base_types.py`, `game.py` and `scoring.py` hold the rules; `state_encoding.py`
converts game states to tokens and action indices to actions;
`single_player_env.py` is the Gymnasium environment;
`actor_critic_transformer.py` and `ppo.py` are the model and training loop;
`policy_evaluation.py` holds the baselines, `env_benchmarking.py` measures
environment throughput. Python 3.12+, dependencies pinned in `uv.lock`.

## Next

Replacing pure policy-gradient learning with an AlphaZero-style MCTS. Placement
is deterministic given the drawn tiles and the draft branches over a small known
distribution, so states are cheap to roll forward — `action_transition` on a
deep-copied state is already what the greedy baseline does.

## License

MIT. Cascadia is a game by Flatout Games; this is an unaffiliated
reimplementation for RL research and contains no artwork or other assets from
the published game.
