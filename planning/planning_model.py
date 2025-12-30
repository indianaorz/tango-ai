# planning_model.py
from __future__ import annotations

import copy
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Import Critic + tensorizer (same code path as viewer)
# -----------------------------------------------------------------------------
sys.path.append(os.getcwd())

try:
    from viewer.critic_infer import CriticRunner, _tensorize_frame_v5  # type: ignore
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "viewer"))
    from critic_infer import CriticRunner, _tensorize_frame_v5  # type: ignore

# =============================================================================
# MMBN6 CONSTANTS
# =============================================================================

INVALID_CHIP_IDS = {0, 255, 65535}

# Raw codes in your pipeline are parity-packed (same as viewer):
#   raw_code = 2 * code_idx + parity
# where code_idx in [0..26], 26 == '*'
WILDCARD_CODE_IDX = 26  # '*' (wildcard) in normalized code space

MAX_SLOTS = 5  # Max total selectable slots in a chip window (chips + optional beast-out consumes 1)
MAX_CANDIDATES_LIMIT = 20000  # Safety cap (matches viewer-style guard)

# =============================================================================
# Planner config
# =============================================================================

@dataclass(frozen=True)
class PlannerConfig:
    # How many frames to ignore after entering window (let UI settle)
    window_warmup_frames: int = 30

    # Input controls (press/hold/release cadence)
    button_hold_frames: int = 4
    button_wait_frames: int = 12

    # Critic eval batching (avoid OOM)
    eval_batch_size: int = 4096

    # If your tensorizer uses folder_len for embeddings
    folder_len: int = 30


# =============================================================================
# Helpers: chip code normalization + candidate generation
# =============================================================================

def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x) != 0.0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "t")
    return False


def _normalize_code_idx(raw_code: Any) -> int:
    """
    Normalize parity-packed raw code to code_idx:
      code_idx = raw_code // 2
    """
    c = _as_int(raw_code, 0)
    if c < 0:
        return -1
    return c // 2


def _is_valid_chip_id(cid: Any) -> bool:
    try:
        v = int(cid)
    except Exception:
        return False
    return v not in INVALID_CHIP_IDS and v > 0


def _clamp_list(xs: Sequence[Any], n: int, fill: Any) -> List[Any]:
    out = list(xs)[:n]
    if len(out) < n:
        out.extend([fill] * (n - len(out)))
    return out


def _is_chain_valid(chain_indices: List[int], all_ids: List[int], all_codes: List[int]) -> bool:
    """
    Viewer-equivalent legality check.

    A set of selected chips is legal if EITHER:
      1) All selected chips have the same ID (codes irrelevant), OR
      2) All selected chips have the same normalized code_idx (raw_code//2),
         allowing wildcards ('*') as compatible with any code.

    NOTE: Order does not affect legality, but order matters for execution,
    so we keep permutations distinct.
    """
    if not chain_indices:
        return True

    c_ids = [int(all_ids[i]) for i in chain_indices]
    first_id = c_ids[0]
    if all(x == first_id for x in c_ids):
        return True

    active_code_idx: Optional[int] = None
    for i in chain_indices:
        cidx = _normalize_code_idx(all_codes[i])
        if cidx == WILDCARD_CODE_IDX:
            continue
        if active_code_idx is None:
            active_code_idx = cidx
        elif cidx != active_code_idx:
            return False

    return True


def _generate_chip_chains(
    ids: List[int],
    codes: List[int],
    valid_indices: List[int],
    *,
    max_slots: int = MAX_SLOTS,
    max_candidates_limit: int = MAX_CANDIDATES_LIMIT,
) -> List[List[int]]:
    """
    Viewer-style chain enumeration:
      - Allows ANY order (permutations), since selection order matters in MMBN.
      - DFS with pruning via legality check.
      - Safety limit to avoid blowups.
    """
    valid_chains: List[List[int]] = [[]]  # Empty selection always allowed

    # seed singletons
    stack: List[Tuple[List[int], set[int]]] = []
    for i in valid_indices:
        stack.append(([i], {i}))

    while stack:
        if len(valid_chains) >= max_candidates_limit:
            # Don't spam logs in-game; one line is enough.
            print(f"[Planner] Candidate chain cap hit ({max_candidates_limit}).")
            break

        chain, used = stack.pop()
        valid_chains.append(chain)

        if len(chain) >= max_slots:
            continue

        # try extend with any remaining index (order matters)
        for i in valid_indices:
            if i in used:
                continue

            test = chain + [i]
            if _is_chain_valid(test, ids, codes):
                new_used = set(used)
                new_used.add(i)
                stack.append((test, new_used))

    return valid_chains


def _valid_cross_selections(derived_player: Dict[str, Any]) -> List[int]:
    """
    Port of viewer/app.py cross selection constraints.

    Returns a list of "selection commands":
      - 0 means "no change"
      - 1..10 means choose that cross (if allowed)

    We infer "side history" (Gregar vs Falzar) based on used_cross_mask and current cross.
    """
    used_mask = derived_player.get("used_cross_mask", []) or []
    used_mask = list(used_mask)
    if len(used_mask) < 11:
        used_mask = used_mask + [False] * (11 - len(used_mask))
    else:
        used_mask = used_mask[:11]

    current_cross = _as_int((derived_player.get("active_cross") or {}).get("idx", 0), 0)

    has_gregar_history = any(bool(used_mask[i]) for i in range(1, 6))
    has_falzar_history = any(bool(used_mask[i]) for i in range(6, 11))

    if 1 <= current_cross <= 5:
        has_gregar_history = True
    if 6 <= current_cross <= 10:
        has_falzar_history = True

    valid = [0]  # always allow no-change

    if has_gregar_history and not has_falzar_history:
        pool = [1, 2, 3, 4, 5]
    elif has_falzar_history and not has_gregar_history:
        pool = [6, 7, 8, 9, 10]
    else:
        pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for idx in pool:
        if idx == current_cross:
            continue
        if bool(used_mask[idx]):
            continue
        valid.append(idx)

    return valid


def _valid_beast_options(derived_player: Dict[str, Any]) -> List[bool]:
    """
    Viewer-equivalent beast option logic.

    - If already in Beast: cannot press beast again now => [False]
    - Else if ever used Beast: cannot use again => [False]
    - Else: can choose to beast or not => [False, True]
    """
    beast_state = derived_player.get("beast", {}) or {}
    is_active = _as_bool(beast_state.get("active", False))
    has_ever = _as_bool(beast_state.get("ever", False))

    if is_active:
        return [False]
    if has_ever:
        return [False]
    return [False, True]


# =============================================================================
# CRITIC PLANNER STRATEGY
# =============================================================================

class PlanningAgentStrategy:
    """
    In-game planner that:
      1) Enumerates all legal chip selections + form options (cross/beast),
      2) Hallucinates the post-window state for each candidate,
      3) Evaluates them with the critic in batches,
      4) Executes the best choice through cursor navigation.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        key_bit_positions: Dict[str, int],
        *,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.device = device
        self.key_bits = key_bit_positions
        self.cfg = config or PlannerConfig()

        print(f"[Planner] Initializing Critic-Based Planner: {model_path}")
        self.critic: Optional[CriticRunner] = None
        try:
            self.critic = CriticRunner(
                ckpt_path=model_path,
                device=str(device),
                use_amp=True,
                # We will batch externally; keep internal reasonably large.
                batch_seqs=max(2048, self.cfg.eval_batch_size),
            )
            print("[Planner] Critic loaded successfully.")
        except Exception as e:
            print(f"[Planner] ❌ FATAL: Failed to load critic: {e}")
            self.critic = None

        # Execution state
        self.action_queue: deque[int] = deque()
        self.current_targets: deque[Dict[str, Any]] = deque()
        self.frames_in_window: int = 0
        self.plan_generated: bool = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset_state(self, port: int) -> None:
        self.action_queue.clear()
        self.current_targets.clear()
        self.frames_in_window = 0
        self.plan_generated = False

    def decide_action(self, port: int, game_state: dict) -> dict:
        """
        Loop:
          - If not in chip window: reset, no-op.
          - Warm up a bit.
          - If executing queued button holds: continue.
          - If have navigation targets: move/select.
          - If plan already generated: no-op (wait for window to close).
          - Else: brute-force plan and begin execution.
        """
        inside_window = _as_bool(game_state.get("inside_window", 0))

        if not inside_window:
            if self.frames_in_window > 0:
                # leaving window -> reset
                self.reset_state(port)
            return self._no_op()

        self.frames_in_window += 1

        if self.frames_in_window < self.cfg.window_warmup_frames:
            return self._no_op()

        if self.action_queue:
            mask = self.action_queue.popleft()
            return self._create_response(mask, debug_msg="executing_queue")

        if self.current_targets:
            return self._navigate_to_next_target(game_state)

        if self.plan_generated:
            return self._no_op()

        self.plan_generated = True
        return self._run_brute_force_planning(game_state)

    # -------------------------------------------------------------------------
    # Planning: enumerate -> hallucinate -> critic eval -> choose -> execute
    # -------------------------------------------------------------------------

    def _run_brute_force_planning(self, game_state: dict) -> dict:
        if self.critic is None:
            return self._no_op()

        try:
            static: dict = {}  # keep empty unless you have stable static data in-game
            derived_player = self._reconstruct_derived_player(game_state)

            hand_ids, hand_codes, hand_vis = self._extract_hand(game_state)

            candidates = self._generate_all_valid_moves(
                hand_ids=hand_ids,
                hand_codes=hand_codes,
                hand_vis=hand_vis,
                derived_player=derived_player,
            )

            if not candidates:
                print("[Planner] No valid moves found. Pressing OK.")
                self.current_targets.append({"type": "ok"})
                return self._no_op()

            # Build tensor dicts for critic
            tensor_rows: List[Dict[str, torch.Tensor]] = []
            for cand in candidates:
                tensor_rows.append(
                    self._create_hallucinated_tensor(
                        cand=cand,
                        game_state=game_state,
                        static=static,
                        derived_player=derived_player,
                        hand_ids=hand_ids,
                        hand_codes=hand_codes,
                    )
                )

            # Evaluate in batches
            values = self._critic_eval_rows(tensor_rows)

            best_idx = int(np.argmax(values))
            best_move = candidates[best_idx]
            best_val = float(values[best_idx])

            print(
                f"[Planner] Evaluated {len(candidates)} candidates. "
                f"Best V(s)={best_val:.3f} | chain={best_move['chain']} cross_sel={best_move['cross_sel']} beast={best_move['do_beast']}"
            )

            self._build_execution_path(best_move)
            return self._navigate_to_next_target(game_state)

        except Exception as e:
            print(f"[Planner] ❌ Error in planning: {e}")
            import traceback

            traceback.print_exc()
            self.current_targets.append({"type": "ok"})
            return self._no_op()

    def _critic_eval_rows(self, rows: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """
        Runs critic.model(xb) over hallucinated rows in safe batches.

        Expected tensor row format: key -> Tensor[...]
        We stack each key into xb[key] = Tensor[B, 1, ...] (sequence length 1).
        """
        assert self.critic is not None

        if not rows:
            return np.zeros((0,), dtype=np.float32)

        # For consistent stacking, lock key order from first row.
        keys = list(rows[0].keys())
        out_vals: List[np.ndarray] = []

        bs = max(1, int(self.cfg.eval_batch_size))
        dev = getattr(self.critic, "device", self.device)

        for s in range(0, len(rows), bs):
            chunk = rows[s : s + bs]

            xb: Dict[str, torch.Tensor] = {}
            for k in keys:
                t_list = [d[k] for d in chunk]
                stacked = torch.stack(t_list, dim=0).unsqueeze(1)  # [B, T=1, ...]
                xb[k] = stacked.to(dev, non_blocking=True)

            with torch.no_grad():
                v = self.critic.model(xb).squeeze(-1).squeeze(-1).detach().cpu().numpy()
            out_vals.append(v.astype(np.float32, copy=False))

        return np.concatenate(out_vals, axis=0)

    def _generate_all_valid_moves(
        self,
        *,
        hand_ids: List[int],
        hand_codes: List[int],
        hand_vis: List[float],
        derived_player: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Viewer-equivalent combinatorics:
          1) chip chains across currently valid (visible) chips
          2) cross selection options (0 = no change)
          3) beast options
          Constraints:
            - cannot change cross AND beast-out in same turn
            - beast-out consumes 1 chip slot (max chips reduced by 1 for that selection)
        """
        n_hand = min(len(hand_ids), len(hand_codes), len(hand_vis))
        valid_indices: List[int] = []
        for i in range(n_hand):
            if float(hand_vis[i] or 0.0) <= 0.0:
                continue
            if _is_valid_chip_id(hand_ids[i]):
                valid_indices.append(i)

        chains = _generate_chip_chains(
            hand_ids,
            hand_codes,
            valid_indices,
            max_slots=MAX_SLOTS,
            max_candidates_limit=MAX_CANDIDATES_LIMIT,
        )

        cross_sels = _valid_cross_selections(derived_player)
        beast_opts = _valid_beast_options(derived_player)

        candidates: List[Dict[str, Any]] = []
        for ch in chains:
            for cr_sel in cross_sels:
                for do_beast in beast_opts:
                    # cannot do both
                    if cr_sel > 0 and do_beast:
                        continue

                    max_allowed = MAX_SLOTS - 1 if do_beast else MAX_SLOTS
                    if len(ch) > max_allowed:
                        continue

                    candidates.append(
                        {
                            "chain": ch,          # list[int] hand indices in selection order
                            "cross_sel": cr_sel,  # 0=no change, else cross idx
                            "do_beast": bool(do_beast),
                        }
                    )

        return candidates

    def _create_hallucinated_tensor(
        self,
        *,
        cand: Dict[str, Any],
        game_state: dict,
        static: dict,
        derived_player: Dict[str, Any],
        hand_ids: List[int],
        hand_codes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Creates the tensor dict for critic, imagining the window is closed and
        our selections have been committed.

        Critical fields:
          - derived held_chips reflects selected chain
          - derived active_cross updates only if cross_sel != 0
          - derived beast active toggles if do_beast
          - frame inside_window = False
          - frame player_chip = first selected chip id (on-deck causal link)
        """
        chain: List[int] = list(cand["chain"])
        cross_sel: int = _as_int(cand.get("cross_sel", 0), 0)
        do_beast: bool = bool(cand.get("do_beast", False))

        sim_ids = [int(hand_ids[i]) for i in chain]
        sim_codes = [int(hand_codes[i]) for i in chain]
        sim_on_deck = int(sim_ids[0]) if sim_ids else 0

        # Derived (deep copy)
        sim_derived = copy.deepcopy(derived_player)

        # Apply cross selection semantics:
        # 0 means "no change" (keep current)
        if cross_sel > 0:
            sim_derived.setdefault("active_cross", {})
            sim_derived["active_cross"]["idx"] = int(cross_sel)

        # Apply beast selection
        sim_derived.setdefault("beast", {})
        if do_beast:
            sim_derived["beast"]["active"] = True
            sim_derived["beast"]["ever"] = True  # once used, it's used

        # Selected chips become held chips after closing window
        sim_derived["held_chips"] = [{"id": int(i), "code": int(c)} for i, c in zip(sim_ids, sim_codes)]

        # Frame hallucination
        sim_frame = dict(game_state)
        sim_frame["inside_window"] = False
        sim_frame["player_chip"] = int(sim_on_deck)

        # Tensorize
        return _tensorize_frame_v5(
            sim_frame,
            sim_derived,
            static,
            folder_len=int(self.cfg.folder_len),
        )

    # -------------------------------------------------------------------------
    # Adapters: extract hand + reconstruct derived from in-game state
    # -------------------------------------------------------------------------

    def _extract_hand(self, game_state: dict) -> Tuple[List[int], List[int], List[float]]:
        """
        Accepts multiple naming conventions so you can call it from your in-game pipe
        without having to perfectly mirror viewer keys.

        Supported keys (first match wins):
          - ids:  window_hand_id | hand_slots | chip_slots
          - codes: window_hand_code | hand_codes | chip_codes
          - vis: window_hand_vis | hand_vis | chip_vis
        """
        ids = game_state.get("window_hand_id", None)
        if ids is None:
            ids = game_state.get("hand_slots", None)
        if ids is None:
            ids = game_state.get("chip_slots", None)
        ids_l = [int(x) for x in (list(ids) if isinstance(ids, (list, tuple)) else [])]

        codes = game_state.get("window_hand_code", None)
        if codes is None:
            codes = game_state.get("hand_codes", None)
        if codes is None:
            codes = game_state.get("chip_codes", None)
        codes_l = [int(x) for x in (list(codes) if isinstance(codes, (list, tuple)) else [])]

        vis = game_state.get("window_hand_vis", None)
        if vis is None:
            vis = game_state.get("hand_vis", None)
        if vis is None:
            vis = game_state.get("chip_vis", None)

        if isinstance(vis, (list, tuple)):
            vis_l = [float(x) for x in vis]
        else:
            # If no explicit visibility, assume all present entries are visible.
            n = max(len(ids_l), len(codes_l))
            vis_l = [1.0] * n

        # Clamp all to a consistent length (10 slots typical)
        n = min(10, max(len(ids_l), len(codes_l), len(vis_l)))
        ids_l = _clamp_list(ids_l, n, 0)
        codes_l = _clamp_list(codes_l, n, 0)
        vis_l = _clamp_list(vis_l, n, 1.0)

        return ids_l, codes_l, vis_l

    def _reconstruct_derived_player(self, game_state: dict) -> Dict[str, Any]:
        """
        Build the subset of 'derived.player' used by the solver + tensorizer.

        Expected/optional in-game fields:
          - player_cross_id OR derived.active_cross.idx
          - used_cross_mask OR player_used_crosses_list
          - beast_mode / beast_active / beast_ever
          - folder_used_mask (optional)
        """
        # Current cross
        cross_idx = _as_int(game_state.get("player_cross_id", None), 0)
        if cross_idx == 0:
            # allow nested dict if you already provide it
            d = game_state.get("derived", {}) or {}
            p = d.get("player", {}) or {}
            cross_idx = _as_int((p.get("active_cross") or {}).get("idx", 0), 0)

        # Used crosses
        used_cross_mask = game_state.get("used_cross_mask", None)
        if used_cross_mask is None:
            used_cross_mask = game_state.get("player_used_crosses_list", None)
        if used_cross_mask is None:
            d = game_state.get("derived", {}) or {}
            p = d.get("player", {}) or {}
            used_cross_mask = p.get("used_cross_mask", [])

        used_cross_mask_l = list(used_cross_mask) if isinstance(used_cross_mask, (list, tuple)) else []
        if len(used_cross_mask_l) < 11:
            used_cross_mask_l += [False] * (11 - len(used_cross_mask_l))
        used_cross_mask_l = used_cross_mask_l[:11]

        # Beast state
        beast_active = _as_bool(game_state.get("beast_active", game_state.get("beast_mode", 0)))
        beast_ever = _as_bool(game_state.get("beast_ever", False))
        # allow nested derived override
        d = game_state.get("derived", {}) or {}
        p = d.get("player", {}) or {}
        beast_nested = p.get("beast", None)
        if isinstance(beast_nested, dict):
            beast_active = _as_bool(beast_nested.get("active", beast_active))
            beast_ever = _as_bool(beast_nested.get("ever", beast_ever))

        folder_used_mask = game_state.get("folder_used_mask", None)
        if folder_used_mask is None:
            folder_used_mask = p.get("folder_used_mask", [])
        folder_used_mask_l = list(folder_used_mask) if isinstance(folder_used_mask, (list, tuple)) else []

        return {
            "active_cross": {"idx": int(cross_idx)},
            "used_cross_mask": used_cross_mask_l,
            "beast": {"active": bool(beast_active), "ever": bool(beast_ever)},
            "folder_used_mask": folder_used_mask_l,
            "held_chips": [],  # inside window
            "used_chip_id": 0,
        }

    # -------------------------------------------------------------------------
    # Execution: build a target list and navigate
    # -------------------------------------------------------------------------

    def _build_execution_path(self, best_move: Dict[str, Any]) -> None:
        """
        Convert move into targets:
          - optional beast
          - optional cross change (if cross_sel != 0)  [stub: keep your own UI mapping]
          - chips in chain order
          - ok
        """
        cross_sel = _as_int(best_move.get("cross_sel", 0), 0)
        do_beast = bool(best_move.get("do_beast", False))

        # Beast first (matches UI constraint: can't also cross-change in same turn anyway)
        if do_beast:
            self.current_targets.append({"type": "beast"})

        # Cross change target (only if your in-game UI supports selecting it here)
        # NOTE: The viewer solver models cross selection as part of the same window commit.
        # If your in-game automation can navigate to cross and pick it, wire it here.
        if cross_sel > 0:
            self.current_targets.append({"type": "cross", "idx": int(cross_sel)})

        # Chips (hand indices in order)
        for idx in best_move.get("chain", []):
            self.current_targets.append({"type": "chip", "idx": int(idx)})

        self.current_targets.append({"type": "ok"})

    def _navigate_to_next_target(self, game_state: dict) -> dict:
        """
        Minimal cursor navigation logic.

        Assumptions (you can adapt to your actual in-game signals):
          - game_state["selected_menu_index"] is current cursor slot
            0..9 are chips (2 rows of 5), 10 is OK
          - OK can be confirmed by A or START
          - Beast can be triggered by R
          - Chip select is A
        """
        if not self.current_targets:
            return self._no_op()

        target = self.current_targets[0]
        curr_idx = _as_int(game_state.get("selected_menu_index", 0), 0)

        # BEAST
        if target["type"] == "beast":
            self.current_targets.popleft()
            return self._press("R")

        # CROSS (stub)
        if target["type"] == "cross":
            # You need to define how to navigate to cross selection in your in-game UI.
            # If cross selection is not accessible here, just drop it.
            # Keeping behavior deterministic: we drop it instead of guessing controls.
            self.current_targets.popleft()
            return self._no_op()

        # OK
        if target["type"] == "ok":
            # If already on OK slot, press A; else use START shortcut
            self.current_targets.popleft()
            if curr_idx == 10:
                return self._press("A")
            return self._press("START")

        # CHIP SELECT
        if target["type"] == "chip":
            tgt_idx = _as_int(target.get("idx", 0), 0)

            if curr_idx == tgt_idx:
                self.current_targets.popleft()
                return self._press("A")

            # If at OK, move up to chips
            if curr_idx == 10:
                return self._press("UP")

            # 0..4 top row, 5..9 bottom row
            curr_x, curr_y = curr_idx % 5, curr_idx // 5
            tgt_x, tgt_y = tgt_idx % 5, tgt_idx // 5

            if curr_y < tgt_y:
                return self._press("DOWN")
            if curr_y > tgt_y:
                return self._press("UP")
            if curr_x < tgt_x:
                return self._press("RIGHT")
            if curr_x > tgt_x:
                return self._press("LEFT")

        return self._no_op()

    # -------------------------------------------------------------------------
    # Button IO
    # -------------------------------------------------------------------------

    def _press(self, key_name: str) -> dict:
        mask = self._get_key_mask(key_name)

        for _ in range(int(self.cfg.button_hold_frames)):
            self.action_queue.append(mask)
        for _ in range(int(self.cfg.button_wait_frames)):
            self.action_queue.append(0)

        return self._create_response(mask, debug_msg=f"press_{key_name}")

    def _get_key_mask(self, btn_name: str) -> int:
        # Your existing alias mapping
        alias = {"A": "Z", "B": "X", "L": "A", "R": "S", "START": "RETURN"}
        name = alias.get(btn_name, btn_name)
        if name in self.key_bits:
            return 1 << int(self.key_bits[name])
        return 0

    def _create_response(self, mask: int, debug_msg: str) -> dict:
        bin_str = format(int(mask) & 0xFFFF, "016b")
        return {
            "button_command": {"type": "key_press", "key": bin_str},
            "debug": {"plan_action": debug_msg},
        }

    def _no_op(self) -> dict:
        return self._create_response(0, "no_op")
