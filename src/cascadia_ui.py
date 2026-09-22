"""Standalone Tk UI for playing and inspecting Cascadia policies.

Run from the repository root with:
    PYTHONPATH=src python -m cascadia_ui --players 2

For policy inspection, pass a mapping of player indexes to callables to
``CascadiaUI(..., policies={0: policy})``.  A policy is simply
``policy(game_state) -> Action``.  The UI shows its proposed action before
executing it and keeps immutable history snapshots for back/forward review.
"""

from __future__ import annotations

import argparse
import copy
import math
import tkinter as tk
from tkinter import ttk
from typing import Callable, Mapping

import numpy as np
import torch

from base_types import Action, ActionKind, Animal, BasePlate, BuiltPlate, GamePhase, GameState, Landscape
from config import Config
from game import action_transition, place_animal_mask, place_land_plate_mask, reroll_all_mask, reroll_three_mask
from ppo import Agent
from scoring import get_scoring_method_for, land_extra_points, score_land
from state_encoding import action_from_index, compute_action_mask, compute_state


LAND_COLORS = {
    Landscape.RIVER: "#164b83", Landscape.MOUNTAIN: "#8b9099",
    Landscape.WETLAND: "#79c7b4", Landscape.FOREST: "#27633c",
    Landscape.PLAINS: "#e8c95a",
}
ANIMAL_COLORS = {
    Animal.SALMON: "#f28c9d", Animal.DEER: "#c69b6d", Animal.GRIZZLY: "#70452c",
    Animal.FOX: "#ed822f", Animal.HAWK: "#8ac7e8",
}
SHORT_ANIMAL = {Animal.GRIZZLY: "B", Animal.DEER: "D", Animal.SALMON: "S", Animal.HAWK: "H", Animal.FOX: "F"}
SHORT_LAND = {Landscape.MOUNTAIN: "M", Landscape.FOREST: "F", Landscape.PLAINS: "P", Landscape.WETLAND: "W", Landscape.RIVER: "R"}


class CascadiaUI(tk.Tk):
    """A human-play and policy-inspection shell around the project game model."""

    def __init__(self, state: GameState, policies: Mapping[int, Callable[[GameState], Action]] | None = None):
        super().__init__()
        self.config = Config()
        self.title("Cascadia — player & policy inspector")
        self.minsize(1120, 700)
        self.history = [copy.deepcopy(state)]
        self.history_index = 0
        self.policies = dict(policies or {})
        self.autoplay = [tk.BooleanVar(value=False) for _ in state.players]
        self.rotation = 0
        self.hover_cell: tuple[int, int] | None = None
        self.hover_pool: tuple[int, str] | None = None
        self.mixed_mode = False
        self.mixed_land: int | None = None
        self.mixed_animal: int | None = None
        self.pool_hits: list[tuple[tuple[float, float, float, float], int, str]] = []
        self.status = tk.StringVar()
        self._build()
        self.bind("<a>", lambda _: self.rotate(-1))
        self.bind("<d>", lambda _: self.rotate(1))
        self.bind("<Left>", lambda _: self.back())
        self.bind("<Right>", lambda _: self.forward())
        self.refresh()

    @property
    def state(self) -> GameState:
        return self.history[self.history_index]

    def _build(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=3)
        root.rowconfigure(1, weight=1)

        controls = ttk.Frame(root)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="← Back", command=self.back).pack(side="left")
        ttk.Button(controls, text="Forward →", command=self.forward).pack(side="left", padx=4)
        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=8)
        self.reroll_all_button = ttk.Button(controls, text="Reroll all  ●", command=lambda: self.take(ActionKind.REROLL_ALL))
        self.reroll_all_button.pack(side="left")
        self.reroll_three_button = ttk.Button(controls, text="Reroll three  ●", command=lambda: self.take(ActionKind.REROLL_THREE))
        self.reroll_three_button.pack(side="left", padx=4)
        self.mixed_button = ttk.Button(controls, text="Take mixed  ●", command=self.toggle_mixed)
        self.mixed_button.pack(side="left")
        ttk.Label(controls, textvariable=self.status).pack(side="right")

        left = ttk.Frame(root)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(left, text="Pool", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        self.pool = tk.Canvas(left, width=270, height=360, background="#f6f3ed", highlightthickness=0)
        self.pool.pack(fill="x", pady=(4, 12))
        self.pool.bind("<Motion>", self.on_pool_motion)
        self.pool.bind("<Leave>", lambda _: self.set_pool_hover(None))
        self.pool.bind("<Button-1>", self.on_pool_click)
        ttk.Label(left, text="Scores", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        self.score = ttk.Treeview(left, show="headings", height=7)
        headings = ("Player", "Cones", "Animals", "Land + bonus", "Total")
        self.score["columns"] = headings
        for column, width in zip(headings, (55, 52, 104, 130, 48)):
            self.score.heading(column, text=column)
            self.score.column(column, width=width, anchor="center", stretch=column in ("Animals", "Land + bonus"))
        self.score.pack(fill="x", pady=(4, 12))
        self.auto_frame = ttk.LabelFrame(left, text="Player controls", padding=6)
        self.auto_frame.pack(fill="x")

        board_frame = ttk.Frame(root)
        board_frame.grid(row=1, column=1, sticky="nsew")
        board_frame.rowconfigure(1, weight=1)
        board_frame.columnconfigure(0, weight=1)
        self.phase_label = ttk.Label(board_frame, font=("TkDefaultFont", 13, "bold"))
        self.phase_label.grid(row=0, column=0, sticky="w", pady=(0, 3))
        self.board = tk.Canvas(board_frame, background="#eef3ed", highlightthickness=0)
        self.board.grid(row=1, column=0, sticky="nsew")
        self.board.bind("<Configure>", lambda _: self.draw_board())
        self.board.bind("<Motion>", self.on_board_motion)
        self.board.bind("<Leave>", lambda _: self.set_cell_hover(None))
        self.board.bind("<Button-1>", self.on_board_click)
        self.board.bind("<MouseWheel>", self.on_wheel)
        self.board.bind("<Button-4>", lambda _: self.rotate(1))
        self.board.bind("<Button-5>", lambda _: self.rotate(-1))

    def _action_preview(self) -> Action | None:
        if self.history_index != len(self.history) - 1:
            return None
        player = self.state.active_player
        if not self.autoplay[player].get() or player not in self.policies:
            return None
        try:
            return self.policies[player](copy.deepcopy(self.state))
        except Exception as exc:  # A policy is inspection code; do not crash the UI for it.
            self.status.set(f"Policy preview failed: {exc}")
            return None

    def refresh(self) -> None:
        state = self.state
        
        stop_when_remaining = 81 - 20*len(state.players)
        gameover = (len(state.bag.base_plates) <= stop_when_remaining) and state.game_phase==GamePhase.PICKING

        phase = state.game_phase.name.replace("_", " ").title()
        active = state.active_player + 1
        held = state.players[state.active_player].to_place
        held_text = ""
        if held:
            held_text = f"  Selected: {held[0].name.title()} + {held[1].left_landscape.name.title()}/{held[1].right_landscape.name.title()}"
        preview = self._action_preview()
        preview_text = f"  Policy preview: {self.describe(preview)}" if preview else ""
        self.phase_label.configure(text=f"Player {active} is active — {phase}.{held_text}{preview_text}")
        self.reroll_all_button.configure(state="normal" if state.game_phase is GamePhase.PICKING and reroll_all_mask(state, state.active_player) else "disabled")
        self.reroll_three_button.configure(state="normal" if state.game_phase is GamePhase.PICKING and reroll_three_mask(state, state.active_player) else "disabled")
        self.mixed_button.configure(state="normal" if state.game_phase is GamePhase.PICKING and state.players[state.active_player].pine_cones else "disabled")
        self.mixed_button.configure(text=("Cancel mixed" if self.mixed_mode else "Take mixed  ●"))
        self.draw_pool(preview)
        self.draw_scores()
        if not gameover:
            self.draw_autoplay()
        self.draw_board(preview)
        if preview and not gameover:
            self.after(350, self.maybe_autoplay)

    def draw_autoplay(self) -> None:
        for child in self.auto_frame.winfo_children(): child.destroy()
        for index, variable in enumerate(self.autoplay):
            enabled = index in self.policies
            ttk.Checkbutton(self.auto_frame, text=f"Player {index + 1}: {'Policy' if enabled else 'Manual (no policy)'}",
                            variable=variable, state="normal" if enabled else "disabled", command=self.refresh).pack(anchor="w")

    def draw_scores(self) -> None:
        for item in self.score.get_children(): self.score.delete(item)
        try: extras = land_extra_points(self.state)
        except Exception: extras = [{land: 0 for land in Landscape} for _ in self.state.players]
        for idx, player in enumerate(self.state.players):
            animal_scores = {}
            for animal in Animal:
                try: animal_scores[animal] = get_scoring_method_for(animal)(self.state, idx)
                except Exception: animal_scores[animal] = 0
            try: lands = score_land(self.state, idx)
            except Exception: lands = {land: 0 for land in Landscape}
            animals_text = " ".join(f"{SHORT_ANIMAL[a]}:{animal_scores[a]}" for a in Animal)
            lands_text = " ".join(f"{SHORT_LAND[l]}:{lands[l]}+{extras[idx][l]}" for l in Landscape)
            total = sum(animal_scores.values()) + sum(lands.values()) + sum(extras[idx].values())
            tags = ("active",) if idx == self.state.active_player else ()
            self.score.insert("", "end", values=(f"P{idx + 1}", player.pine_cones, animals_text, lands_text, total), tags=tags)
        self.score.tag_configure("active", background="#dbeeda")

    def center_for(self, cell: tuple[int, int], radius: float) -> tuple[float, float]:
        w, h = max(self.board.winfo_width(), 1), max(self.board.winfo_height(), 1)
        i, j = cell
        return (w / 2 + (j - i / 2) * math.sqrt(3) * radius, h / 2 + i * 1.5 * radius)

    @staticmethod
    def hex_points(cx: float, cy: float, radius: float, orientation: int, left: bool) -> list[float]:
        # Point-down axial hex, rotated counter-clockwise in mathematical coordinates.
        vertex_indices = (0, 5, 4, 3) if left else (0, 1, 2, 3)
        angle_offset = -math.pi / 2 - orientation * math.pi / 3
        pts = []
        for index in vertex_indices:
            angle = angle_offset + index * math.pi / 3
            pts.extend((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        return pts

    def draw_plate(self, canvas: tk.Canvas, plate: BasePlate, center: tuple[float, float], radius: float,
                   orientation: int = 0, ghost: bool = False, tags: str = "") -> None:
        if plate is None:
            return
        cx, cy = center
        opts = {"outline": "#36443a", "width": 1, "tags": tags}
        if ghost: opts["stipple"] = "gray50"
        canvas.create_polygon(self.hex_points(cx, cy, radius, orientation, True), fill=LAND_COLORS[plate.left_landscape], **opts)
        canvas.create_polygon(self.hex_points(cx, cy, radius, orientation, False), fill=LAND_COLORS[plate.right_landscape], **opts)
        if isinstance(plate, BuiltPlate) and plate.built_animal is not None:
            self.draw_animal(canvas, cx, cy, plate.built_animal, radius * .27, ghost)
        else:
            for n, animal in enumerate(plate.animals):
                angle = -math.pi / 2 + (n - (len(plate.animals) - 1) / 2) * .75
                self.draw_animal(canvas, cx + math.cos(angle) * radius * .42, cy + math.sin(angle) * radius * .42, animal, radius * .12, ghost)

    def draw_animal(self, canvas: tk.Canvas, x: float, y: float, animal: Animal, radius: float, ghost: bool = False) -> None:
        if animal is None:
            return
        opts = {"fill": ANIMAL_COLORS[animal], "outline": "#2e3030"}
        if ghost: opts["stipple"] = "gray50"
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, **opts)
        canvas.create_text(x, y, text=SHORT_ANIMAL[animal], font=("TkDefaultFont", max(7, int(radius))), fill="#172018")

    def draw_board(self, preview: Action | None = None) -> None:
        if not hasattr(self, "board"): return
        self.board.delete("all")
        grid = self.state.players[self.state.active_player].plate_grid
        radius = min(58, max(25, min(self.board.winfo_width() / 12, self.board.winfo_height() / 8)))
        for cell, plate in grid.items(): self.draw_plate(self.board, plate, self.center_for(cell, radius), radius, plate.orientation)
        phase = self.state.game_phase
        held = self.state.players[self.state.active_player].to_place
        candidate = self.hover_cell
        if preview and preview.kind is ActionKind.PLACE_LAND:
            candidate = tuple(preview.params.get("index_place", (None, None))[:2])
            self.rotation = preview.params.get("index_place", (0, 0, self.rotation))[2]
        if phase is GamePhase.PLACING_LAND and held and candidate in self.legal_land_cells():
            self.draw_plate(self.board, held[1], self.center_for(candidate, radius), radius, self.rotation, True)
        if phase is GamePhase.PLACING_ANIMAL and held:
            animal_cell = candidate
            if preview and preview.kind is ActionKind.PLACE_ANIMAL: animal_cell = preview.params.get("index_place")
            if animal_cell in self.legal_animal_cells():
                x, y = self.center_for(animal_cell, radius)
                self.draw_animal(self.board, x, y, held[0], radius * .27, True)
        if phase is GamePhase.PLACING_LAND and held:
            self.board.create_text(10, 10, anchor="nw", text=f"Rotate preview: {self.rotation * 60}° (wheel or A/D)", fill="#33483a")

    def draw_pool(self, preview: Action | None) -> None:
        self.pool.delete("all"); self.pool_hits.clear()
        pool = self.state.pool_state
        for index, (plate, animal) in enumerate(zip(pool.plate_pool, pool.animal_pool)):
            y = 15 + index * 84
            highlight = self.hover_pool and self.hover_pool[0] == index
            if preview and ((preview.kind is ActionKind.TAKE_PAIR and preview.params.get("take_idx") == index) or
                            (preview.kind is ActionKind.TAKE_MIXED and index in (preview.params.get("take_idx_land"), preview.params.get("take_idx_animal")))):
                highlight = True
            if highlight: self.pool.create_rectangle(4, y - 8, 266, y + 66, fill="#d6ead7", outline="")
            self.draw_plate(self.pool, plate, (48, y + 28), 31)
            self.pool.create_text(104, y + 28, text="+", font=("TkDefaultFont", 15, "bold"))
            self.draw_animal(self.pool, 143, y + 28, animal, 17)
            selected = ("  land ✓" if self.mixed_land == index else "") + ("  animal ✓" if self.mixed_animal == index else "")
            self.pool.create_text(180, y + 28, text=f"Pair {index + 1}{selected}", anchor="w", fill="#26352b")
            self.pool_hits.extend([((8, y - 5, 94, y + 61), index, "land"), ((110, y - 5, 174, y + 61), index, "animal")])

    def legal_land_cells(self) -> set[tuple[int, int]]:
        if self.state.game_phase is not GamePhase.PLACING_LAND: return set()
        return set(place_land_plate_mask(self.state, self.state.active_player, self.config)["index_place"])

    def legal_animal_cells(self) -> set[tuple[int, int]]:
        if self.state.game_phase is not GamePhase.PLACING_ANIMAL: return set()
        return {cell for cell in place_animal_mask(self.state, self.state.active_player)["index_place"] if cell is not None}

    def cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        radius = min(58, max(25, min(self.board.winfo_width() / 12, self.board.winfo_height() / 8)))
        # Checking the relevant small candidate set avoids tricky inverse axial rounding.
        candidates = self.legal_land_cells() | self.legal_animal_cells() | set(self.state.players[self.state.active_player].plate_grid)
        if not candidates: return None
        return min(candidates, key=lambda c: (self.center_for(c, radius)[0] - x) ** 2 + (self.center_for(c, radius)[1] - y) ** 2)

    def on_board_motion(self, event: tk.Event) -> None: self.set_cell_hover(self.cell_at(event.x, event.y))
    def set_cell_hover(self, cell: tuple[int, int] | None) -> None:
        if cell != self.hover_cell: self.hover_cell = cell; self.draw_board(self._action_preview())
    def on_wheel(self, event: tk.Event) -> None: self.rotate(1 if event.delta > 0 else -1)
    def rotate(self, amount: int) -> None:
        if self.state.game_phase is GamePhase.PLACING_LAND:
            self.rotation = (self.rotation + amount) % 6; self.draw_board(self._action_preview())

    def on_board_click(self, _: tk.Event) -> None:
        if self.state.game_phase is GamePhase.PLACING_LAND and self.hover_cell in self.legal_land_cells():
            self.take(ActionKind.PLACE_LAND, index_place=(*self.hover_cell, self.rotation))
        elif self.state.game_phase is GamePhase.PLACING_ANIMAL and self.hover_cell in self.legal_animal_cells():
            self.take(ActionKind.PLACE_ANIMAL, index_place=self.hover_cell)

    def pool_hit(self, x: float, y: float) -> tuple[int, str] | None:
        return next(((index, side) for (x1, y1, x2, y2), index, side in self.pool_hits if x1 <= x <= x2 and y1 <= y <= y2), None)
    def on_pool_motion(self, event: tk.Event) -> None: self.set_pool_hover(self.pool_hit(event.x, event.y))
    def set_pool_hover(self, hit: tuple[int, str] | None) -> None:
        if hit != self.hover_pool: self.hover_pool = hit; self.draw_pool(self._action_preview())
    def on_pool_click(self, event: tk.Event) -> None:
        hit = self.pool_hit(event.x, event.y)
        if not hit or self.state.game_phase is not GamePhase.PICKING: return
        index, side = hit
        if not self.mixed_mode:
            self.take(ActionKind.TAKE_PAIR, take_idx=index); return
        if side == "land": self.mixed_land = index
        else: self.mixed_animal = index
        if self.mixed_land is not None and self.mixed_animal is not None:
            self.take(ActionKind.TAKE_MIXED, take_idx_land=self.mixed_land, take_idx_animal=self.mixed_animal)
            self.mixed_mode = False; self.mixed_land = self.mixed_animal = None
        self.refresh()

    def toggle_mixed(self) -> None:
        self.mixed_mode = not self.mixed_mode; self.mixed_land = self.mixed_animal = None; self.refresh()
    def take(self, kind: ActionKind, **params: object) -> None:
        if self.history_index != len(self.history) - 1:
            self.status.set("Go to the newest state before making a different move."); return
        next_state = copy.deepcopy(self.state)
        try:
            action_transition(next_state, Action(kind, params), next_state.active_player, self.config)
        except Exception as exc:
            self.status.set(f"Move rejected: {exc}"); return
        self.history.append(next_state); self.history_index += 1
        self.status.set(""); self.refresh()
    def back(self) -> None:
        if self.history_index: self.history_index -= 1; self.status.set(""); self.refresh()
    def forward(self) -> None:
        if self.history_index < len(self.history) - 1: self.history_index += 1; self.status.set(""); self.refresh()
    def maybe_autoplay(self) -> None:
        if self.history_index != len(self.history) - 1: return
        player = self.state.active_player
        if not self.autoplay[player].get() or player not in self.policies: return
        action = self._action_preview()
        if action: self.take(action.kind, **action.params)
    @staticmethod
    def describe(action: Action | None) -> str:
        if not action: return ""
        return action.kind.name.replace("_", " ").title() + (f" {action.params}" if action.params else "")

def build_policy():
    config = Config()
    model = Agent(config)
    # model.load_state_dict(torch.load('/Users/kilianmandon/Downloads/small_model.pt', map_location='cpu')["model_state_dict"])

    def policy(game_state: GameState) -> Action:
        player_idx = game_state.active_player
        action_mask = torch.tensor(compute_action_mask(game_state, player_idx, config))
        state = torch.tensor(compute_state(game_state, player_idx, config)).float()

        state = state[None, ...]
        action_mask = action_mask[None, ...]

        action_idx, log_prob, entropy, value = model.get_action_and_value(state, action_mask=action_mask)
        # action_idx = action_mask.int().argmax(dim=-1)
        action_idx = action_idx.item()
        action = action_from_index(game_state, player_idx, action_idx, config)

        return action

    return policy



def launch(players: int = 2, policies: Mapping[int, Callable[[GameState], Action]] | None = None) -> CascadiaUI:
    state = GameState(players, np.random.default_rng(5))
    state.init_game()
    app = CascadiaUI(state, policies)
    app.mainloop()
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the standalone Cascadia UI")
    parser.add_argument("--players", type=int, default=1, choices=range(1, 6))
    policies = {
        0: build_policy()
    }
    args = parser.parse_args()
    launch(args.players, policies)
