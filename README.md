## Standalone UI

Launch a playable local UI from the repository root:

```bash
PYTHONPATH=src python -m cascadia_ui --players 2
```

It displays the active player's hex grid, pool, selected tiles, live score
breakdown, and per-player policy/manual controls.  In the land-placement phase,
hover a legal empty hex and use the mouse wheel or `A` / `D` to rotate the
transparent tile preview.  The left/right arrow keys (or buttons) move through
deep-copied game-state history; making a move is intentionally disabled while
viewing an earlier snapshot.

To inspect an RL policy, embed the UI and pass a callable for each policy
player.  Each callable receives a deep-copied `GameState` and returns the
existing `Action` type:

```python
from cascadia_ui import launch

launch(players=2, policies={1: lambda state: my_policy(state)})
```
