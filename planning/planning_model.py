# planning/planning_model.py
from __future__ import annotations

import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

INVALID_CHIP_IDS = {255, 65535}
CODE_INDEXES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*"

# Cross names aligned with your viewer mapping (idx 0..10)
_CROSS_NAME_BY_ID: Dict[int, str] = {
    0: "Normal",
    1: "Fire",
    2: "Elec",
    3: "Slash",
    4: "Erase",
    5: "Charge",
    6: "Aqua",
    7: "Thawk",
    8: "Tengu",
    9: "Grnd",
    10: "Dust",
}

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return _as_bool(v, default=default)

def _code_to_char(raw_code: int) -> str:
    # raw_code parity encodes S/M; letter is raw_code//2
    try:
        idx = int(raw_code) // 2
    except Exception:
        idx = 0
    if 0 <= idx < len(CODE_INDEXES):
        return CODE_INDEXES[idx]
    return "?"


# BN6 Cross IDs (your "emotion idx" convention: 1..10 are crosses, 0 = no change)
_VERSION_CROSS_IDXS: Dict[str, List[int]] = {
    "Gregar": [1, 2, 3, 4, 5],
    "Falzar": [6, 7, 8, 9, 10],
}

# Code normalization: your data uses 0..127 where letters are paired (A/B variants etc)
# c//2 turns 52/53 -> 26 which matches your wildcard code index.
WILDCARD_CODE_IDX = 26

# Chip selection rules
MAX_CHIPS_PER_TURN = 5
MAX_CHIPS_IF_BEAST = 4  # selecting Beast Out consumes 1 "slot"


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return default
            return int(float(s))
        return int(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y"):
            return True
        if s in ("0", "false", "no", "n"):
            return False
    return default


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []


def _chip_valid(chip_id: Any) -> bool:
    try:
        cid = int(chip_id)
    except Exception:
        return False
    return cid not in INVALID_CHIP_IDS and cid > 0


def _normalize_code_idx(raw_code: int) -> int:
    # normalize paired codes (e.g., 52/53 -> 26)
    return int(raw_code) // 2


def _is_chain_valid(chain_indices: List[int], all_ids: List[int], all_codes: List[int]) -> bool:
    """
    BN chip legality (as used in your UI solver earlier):
      - If all selected chips have the same ID => always valid.
      - Else => all non-wildcard codes must match (after normalization via //2).
    """
    if not chain_indices:
        return True

    c_ids = [int(all_ids[i]) for i in chain_indices]
    first_id = c_ids[0]
    if all(x == first_id for x in c_ids):
        return True

    active_code_idx: Optional[int] = None
    for i in chain_indices:
        code_idx = _normalize_code_idx(int(all_codes[i]))
        if code_idx == WILDCARD_CODE_IDX:
            continue
        if active_code_idx is None:
            active_code_idx = code_idx
        elif code_idx != active_code_idx:
            return False

    return True


def _generate_chip_chains_ordered(
    ids: List[int],
    codes: List[int],
    valid_indices: List[int],
    *,
    max_len: int,
    max_chains: int,
) -> List[List[int]]:
    """
    Generate ALL ordered valid chip chains (permutations) up to max_len.
    Includes empty chain.

    Worst-case (10 compatible chips, max_len=5): 36,101 chains.
    """
    out: List[List[int]] = [[]]

    if not valid_indices:
        return out

    # DFS stack: (current_chain, used_mask_bitset)
    # valid_indices are 0..9 so we can use 10-bit mask.
    stack: List[Tuple[List[int], int]] = []

    for i in valid_indices:
        stack.append(([i], (1 << i)))

    while stack:
        if len(out) >= max_chains:
            print(f"[Plan][perm] Hit safety limit of {max_chains} chains. Truncating.")
            break

        chain, used = stack.pop()
        out.append(chain)

        if len(chain) >= max_len:
            continue

        for i in valid_indices:
            if used & (1 << i):
                continue
            test = chain + [i]
            if _is_chain_valid(test, ids, codes):
                stack.append((test, used | (1 << i)))

    return out


# =============================================================================
# Chips DB (lightweight)
# =============================================================================

@dataclass(frozen=True)
class ChipInfo:
    chip_id: int
    name: str = ""
    element: str = ""
    damage: int = 0


class ChipsDB:
    def __init__(self, path: str | Path):
        self._by_id: Dict[int, ChipInfo] = {}
        p = Path(path)
        if not p.exists():
            print(f"⚠️ [Plan] Warning: Chips DB not found at {p} (will sort chips by id only)")
            return

        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ [Plan] Warning: failed reading chips db {p}: {e!r}")
            return

        def _pick_damage(d: Any) -> int:
            if d is None:
                return 0
            if isinstance(d, (int, float)):
                return int(d)
            if isinstance(d, str):
                s = d.strip()
                if not s:
                    return 0
                if "-" in s:
                    parts = [x.strip() for x in s.split("-") if x.strip()]
                    for part in reversed(parts):
                        if part and part[0].isdigit():
                            return _as_int(part, 0)
                return _as_int(s, 0)
            return _as_int(d, 0)

        by_id: Dict[int, ChipInfo] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                cid = _as_int(k, -1)
                if cid < 0 or not isinstance(v, dict):
                    continue
                by_id[cid] = ChipInfo(
                    chip_id=cid,
                    name=str(v.get("name") or v.get("Name") or ""),
                    element=str(v.get("element") or v.get("Element") or ""),
                    damage=_pick_damage(v.get("damage") or v.get("Damage")),
                )
        elif isinstance(obj, list):
            for row in obj:
                if not isinstance(row, dict):
                    continue
                cid = _as_int(
                    row.get("id")
                    or row.get("chip_id")
                    or row.get("ChipID")
                    or row.get("SId")
                    or row.get("MId")
                    or -1,
                    -1,
                )
                if cid < 0:
                    continue
                by_id[cid] = ChipInfo(
                    chip_id=cid,
                    name=str(row.get("name") or row.get("Name") or ""),
                    element=str(row.get("element") or row.get("Element") or ""),
                    damage=_pick_damage(row.get("damage") or row.get("Damage")),
                )
        else:
            print(f"⚠️ [Plan] Warning: chips db json is neither dict nor list at {p}")

        # Logging knobs (default: ONLY print chosen plan lines)
        self.LOG_NAV = _env_flag("PLAN_LOG_NAV", default=False)          # nav key spam
        self.LOG_ENUM = _env_flag("PLAN_LOG_ENUM", default=False)        # perm/candidate counts
        self.LOG_TOPK = _env_flag("PLAN_LOG_TOPK", default=False)        # top-k debug list


        self._by_id = by_id

    def get(self, chip_id: int) -> ChipInfo:
        return self._by_id.get(int(chip_id), ChipInfo(chip_id=int(chip_id)))


# =============================================================================
# Planning Critic runner (matches your viewer/critic_planner_infer.py)
# =============================================================================

class PlanningCriticRunner:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Planning Critic ckpt not found: {p}")

        try:
            from critic_planning_rl.model import PlanningCritic, PlanningConfig  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed importing critic_planning_rl.model (PlanningCritic/PlanningConfig)."
            ) from e

        print(f"[PlanningCritic] Loading {p} on {self.device}...")
        state = torch.load(str(p), map_location="cpu")

        # sanitize DDP/compile prefixes
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k.replace("_orig_mod.", "")
                new_state[nk] = v
            state = new_state

        self.model = PlanningCritic(PlanningConfig())
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        print("[PlanningCritic] Loaded successfully.")

    @torch.no_grad()
    def infer_rows(self, rows: List[Dict[str, Any]]) -> List[Optional[float]]:
        if not rows:
            return []
        batch = self._tensorize_batch(rows)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        symlog_preds = self.model(batch)  # [B]
        preds = symlog_preds.detach().float().cpu().numpy()
        real_yields = np.sign(preds) * (np.expm1(np.abs(preds)))
        return [float(x) for x in real_yields]

    def _tensorize_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        p_hp_list, e_hp_list, turn_list = [], [], []
        grid_t_list, grid_o_list = [], []
        fp_list, fe_list, cp_list, ce_list = [], [], [], []
        held_id_list, held_code_list = [], []
        draw_id_list, draw_code_list = [], []
        sel_id_list, sel_code_list, sel_cross_list, sel_beast_list = [], [], [], []

        for row in rows:
            p_hp_list.append(float(row.get("p_hp_open", 0)) / 1000.0)
            e_hp_list.append(float(row.get("e_hp_open", 0)) / 1000.0)
            turn_list.append(float(row.get("turn_index_open", 0)) / 50.0)

            grid_t_list.append(torch.tensor(row.get("grid_tile_open", [0] * 18), dtype=torch.long).clamp(0, 31))
            grid_o_list.append(torch.tensor(row.get("grid_owner_open", [2] * 18), dtype=torch.long).clamp(0, 2))

            def to_float(k: str, n: int) -> torch.Tensor:
                lst = row.get(k, [])
                return torch.tensor([1.0 if x else 0.0 for x in lst] + [0.0] * max(0, n - len(lst)))[:n]

            fp_list.append(to_float("folder_used_mask_p_open", 30))
            fe_list.append(to_float("folder_used_mask_e_open", 30))
            cp_list.append(to_float("used_cross_mask_p_open", 11))
            ce_list.append(to_float("used_cross_mask_e_open", 11))

            held_id_list.append(torch.tensor(row.get("held_before_id", [0] * 5), dtype=torch.long))
            held_code_list.append(torch.tensor(row.get("held_before_code", [0] * 5), dtype=torch.long).clamp(0, 127))

            draw_id_raw = row.get("window_hand_id", [0] * 10)
            draw_code_raw = row.get("window_hand_code", [0] * 10)
            vis = row.get("window_hand_vis", [0.0] * 10)

            d_ids, d_codes = [], []
            for i in range(10):
                is_vis = (vis[i] > 0.5)
                d_ids.append(draw_id_raw[i] if is_vis else 0)
                d_codes.append(draw_code_raw[i] if is_vis else 0)

            draw_id_list.append(torch.tensor(d_ids, dtype=torch.long))
            draw_code_list.append(torch.tensor(d_codes, dtype=torch.long).clamp(0, 127))

            sel_id_list.append(torch.tensor(row.get("selected_chips_id", [0] * 5), dtype=torch.long))
            sel_code_list.append(torch.tensor(row.get("selected_chips_code", [0] * 5), dtype=torch.long).clamp(0, 127))

            sc = int(row.get("selected_cross", 0))
            sel_cross_list.append(torch.tensor(max(0, min(sc, 63)), dtype=torch.long))
            sel_beast_list.append(torch.tensor(1 if row.get("beast_selected") else 0, dtype=torch.long))

        return {
            "p_hp": torch.tensor(p_hp_list, dtype=torch.float32),
            "e_hp": torch.tensor(e_hp_list, dtype=torch.float32),
            "turn_idx": torch.tensor(turn_list, dtype=torch.float32),
            "grid_tile": torch.stack(grid_t_list),
            "grid_owner": torch.stack(grid_o_list),
            "folder_p": torch.stack(fp_list),
            "folder_e": torch.stack(fe_list),
            "cross_hist_p": torch.stack(cp_list),
            "cross_hist_e": torch.stack(ce_list),
            "held_id": torch.stack(held_id_list),
            "held_code": torch.stack(held_code_list),
            "draw_id": torch.stack(draw_id_list),
            "draw_code": torch.stack(draw_code_list),
            "sel_id": torch.stack(sel_id_list),
            "sel_code": torch.stack(sel_code_list),
            "sel_cross": torch.stack(sel_cross_list),
            "sel_beast": torch.stack(sel_beast_list),
        }


# =============================================================================
# Critic-driven candidates
# =============================================================================

@dataclass(frozen=True)
class _Candidate:
    sel_slots: Tuple[int, ...]   # indices in 0..9
    sel_ids: Tuple[int, ...]     # chip ids in order
    sel_codes: Tuple[int, ...]   # raw codes in order
    cross_id: int                # 0=no change, else 1..10
    beast: int                   # 0/1


# =============================================================================
# Planner
# =============================================================================

class PlanningAgentStrategy:
    """
    Critic-driven planning for Custom Screen.
      - Enumerates tens of thousands of legal chip sequences (ordered).
      - Cross choices (0 + available).
      - Beast choice (0/1) but NEVER alongside cross change in same commit.

    Navigation notes (your UI):
      - OK is idx 10
      - BEAST is idx 11
      - OK <-> BEAST is vertical: OK (10) is UP from BEAST, BEAST is DOWN from OK
      - CHIP grid is idx 0..9 (wrap hazards on LEFT at 0/5)
    """

    def __init__(
        self,
        model_path: str,
        chips_db_path: str,
        device: torch.device,
        key_bit_positions: Dict[str, int],
    ):
        self.device = device
        self.key_bits = dict(key_bit_positions)
        self.chips = ChipsDB(chips_db_path)

        # Logging knobs live in ChipsDB; mirror them here because the planner
        # references self.LOG_*.
        self.LOG_NAV = getattr(self.chips, "LOG_NAV", _env_flag("PLAN_LOG_NAV", default=False))
        self.LOG_ENUM = getattr(self.chips, "LOG_ENUM", _env_flag("PLAN_LOG_ENUM", default=False))
        self.LOG_TOPK = getattr(self.chips, "LOG_TOPK", _env_flag("PLAN_LOG_TOPK", default=False))


        self.critic: Optional[PlanningCriticRunner] = None
        try:
            self.critic = PlanningCriticRunner(model_path, device=str(device))
            print(f"[Plan] PlanningCritic loaded from {model_path} on {device}")
        except Exception as e:
            self.critic = None
            print(f"[Plan] ❌ Failed to load planning critic from {model_path}: {e}")

        self._st: Dict[int, Dict[str, Any]] = {}
        self.WINDOW_WARMUP_FRAMES = 180

        # Perf knobs
        self.MAX_CHAINS = 120_000          # safety (chains only)
        self.MAX_CANDIDATES = 350_000      # safety (chains * cross * beast)
        self.CRITIC_BATCH = 2048           # batch rows per critic pass

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    def reset_state(self, port: int) -> None:
        self._st[port] = {
            "action_queue": deque(),
            "current_targets": deque(),
            "frames_in_window": 0,
            "plan_generated": False,
            "last_inside_window": False,

            # stall breaker
            "nav_last_idx": None,
            "nav_stall": 0,
        }

    def _get(self, port: int) -> Dict[str, Any]:
        st = self._st.get(port)
        if st is None:
            self.reset_state(port)
            st = self._st[port]
        return st

    # -------------------------------------------------------------------------
    # Key masks
    # -------------------------------------------------------------------------

    def _int_to_bin16(self, mask: int) -> str:
        return format(int(mask) & 0xFFFF, "016b")

    def _get_key_mask(self, btn_name: str) -> int:
        ALIASES = {
            "A": "Z", "B": "X", "L": "A", "R": "S",
            "START": "RETURN", "SELECT": "BACKSPACE",
            "UP": "Up", "DOWN": "Down", "LEFT": "Left", "RIGHT": "Right",
        }
        physical_name = ALIASES.get(btn_name, btn_name)

        if physical_name in self.key_bits:
            return (1 << int(self.key_bits[physical_name]))

        pu = str(physical_name).upper()
        for k, v in self.key_bits.items():
            if str(k).upper() == pu:
                return (1 << int(v))

        bu = str(btn_name).upper()
        for k, v in self.key_bits.items():
            if str(k).upper() == bu:
                return (1 << int(v))

        return 0

    def _nav_snapshot(self, st: Dict[str, Any], game_state: dict) -> Dict[str, Any]:
        curr_idx_raw = game_state.get("selected_menu_index")
        curr_idx = None if curr_idx_raw is None else int(curr_idx_raw)
        snap = {
            "curr_idx": curr_idx,
            "curr_cross_idx": _as_int(game_state.get("selected_cross_index", 0), 0),
            "inside_cross_window": _as_bool(game_state.get("inside_cross_window", 0), False),
            "visible_count": max(0, min(10, _as_int(game_state.get("chip_visible_count", 5), 5))),
            "targets_head": (st["current_targets"][0] if st.get("current_targets") else None),
            "targets_len": (len(st["current_targets"]) if st.get("current_targets") else 0),
            "queue_len": (len(st["action_queue"]) if st.get("action_queue") else 0),
            "frames_in_window": st.get("frames_in_window", 0),
            "nav_last_idx": st.get("nav_last_idx"),
            "nav_stall": st.get("nav_stall", 0),
        }
        return snap

    def _create_response(self, mask: int, debug_msg: str, extra: Optional[Dict[str, Any]] = None) -> dict:
        d = {"plan_action": debug_msg}
        if extra:
            d.update(extra)
        return {
            "button_command": {"type": "key_press", "key": self._int_to_bin16(mask)},
            "ng_key_bin": "",
            "debug": d,
        }

    def _no_op(self, why: str = "no_op", extra: Optional[Dict[str, Any]] = None) -> dict:
        return self._create_response(0, why, extra=extra)

    def _press(
        self,
        st: Dict[str, Any],
        key_name: str,
        *,
        reason: str,
        game_state: dict,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        mask = self._get_key_mask(key_name)

        q: Deque[int] = st["action_queue"]
        for _ in range(5):
            q.append(mask)
        for _ in range(25):
            q.append(0)

        snap = self._nav_snapshot(st, game_state)
        if extra:
            snap = dict(snap)
            snap.update(extra)

        if self.LOG_NAV:
            print(f"[Plan][nav] key={key_name} reason={reason} snap={snap}")

        return self._create_response(mask, f"press_{key_name}", extra={"key": key_name, "reason": reason, "snap": snap})

    # -------------------------------------------------------------------------
    # Public entry
    # -------------------------------------------------------------------------

    def decide_action(self, port: int, game_state: dict) -> dict:
        st = self._get(port)

        inside_window = _as_bool(game_state.get("inside_window", 0), False)
        cust_gauge = _as_int(game_state.get("cust_gauge", 0), 0)
        planning_phase = inside_window and (cust_gauge == 0)

        if st["last_inside_window"] and not inside_window:
            self.reset_state(port)
            st = self._get(port)
            st["last_inside_window"] = inside_window
            return self._no_op("window_closed_reset")

        st["last_inside_window"] = inside_window

        if not planning_phase:
            st["frames_in_window"] = 0
            st["action_queue"].clear()
            st["current_targets"].clear()
            st["plan_generated"] = False
            st["nav_last_idx"] = None
            st["nav_stall"] = 0
            return self._no_op("not_planning_phase", extra={"inside_window": inside_window, "cust_gauge": cust_gauge})

        st["frames_in_window"] += 1
        if st["frames_in_window"] < self.WINDOW_WARMUP_FRAMES:
            return self._no_op("warmup")

        q: Deque[int] = st["action_queue"]
        if q:
            mask = q.popleft()
            return self._create_response(mask, "executing_queue")

        if st["current_targets"]:
            return self._navigate_to_next_target(st, game_state)

        if st["plan_generated"]:
            return self._no_op("plan_done")

        st["plan_generated"] = True
        return self._run_critic_and_plan(st, game_state)

    # -------------------------------------------------------------------------
    # Critic planning
    # -------------------------------------------------------------------------

    def _run_critic_and_plan(self, st: Dict[str, Any], game_state: dict) -> dict:
        base_row, base_dbg = self._build_base_row(game_state)
        candidates = self._enumerate_candidates(game_state)

        if not candidates:
            st["current_targets"].append({"type": "ok", "val": 0})
            return self._navigate_to_next_target(st, game_state)

        if self.critic is None:
            best = self._pick_heuristic_candidate(candidates)
            self._targets_from_candidate(st, game_state, best)
            print(f"[Plan][pick][heuristic] {self._format_plan_line(best)}")
            return self._navigate_to_next_target(st, game_state)

        # Evaluate in batches so we can handle 100k+ candidates safely.
        best_c: Optional[_Candidate] = None
        best_s: float = -1e30

        # maintain top-5 for logging
        topk = 5
        top: List[Tuple[float, _Candidate]] = []  # unsorted small list

        def _push_top(score: float, cand: _Candidate) -> None:
            nonlocal top
            top.append((score, cand))
            top.sort(key=lambda x: x[0], reverse=True)
            if len(top) > topk:
                top = top[:topk]

        it = iter(candidates)
        total = len(candidates)
        batches = (total + self.CRITIC_BATCH - 1) // self.CRITIC_BATCH

        for bi in range(batches):
            chunk = list(islice(it, self.CRITIC_BATCH))
            if not chunk:
                break

            rows: List[Dict[str, Any]] = []
            for c in chunk:
                r = dict(base_row)
                sel_id = list(c.sel_ids)[:5]
                sel_code = list(c.sel_codes)[:5]
                while len(sel_id) < 5:
                    sel_id.append(0)
                    sel_code.append(0)
                r["selected_chips_id"] = sel_id
                r["selected_chips_code"] = sel_code
                r["selected_cross"] = int(c.cross_id)
                r["beast_selected"] = bool(c.beast)
                rows.append(r)

            scores = self.critic.infer_rows(rows)

            for c, s in zip(chunk, scores):
                if s is None:
                    continue
                ss = float(s)
                if ss > best_s:
                    best_s = ss
                    best_c = c
                _push_top(ss, c)

        if best_c is None:
            best_c = self._pick_heuristic_candidate(candidates)
            self._targets_from_candidate(st, game_state, best_c)
            print(f"[Plan][pick][critic_fail] {self._format_plan_line(best_c)}")
            return self._navigate_to_next_target(st, game_state)

        self._targets_from_candidate(st, game_state, best_c)

        print(f"[Plan][pick][critic] score={best_s:.3f} {self._format_plan_line(best_c)}")
        if self.LOG_TOPK:
            top_dbg = [{"score": float(s), **self._cand_dbg(c)} for s, c in top]
            print(f"[Plan][topk] {top_dbg}")

        return self._navigate_to_next_target(st, game_state)

    def _cand_dbg(self, c: _Candidate) -> Dict[str, Any]:
        return {
            "slots": list(c.sel_slots),
            "ids": list(c.sel_ids),
            "codes": list(c.sel_codes),
            "cross_id": int(c.cross_id),
            "beast": int(c.beast),
        }
    
    def _cross_name(self, cross_id: int) -> str:
        return _CROSS_NAME_BY_ID.get(int(cross_id), f"Cross{int(cross_id)}")

    def _format_plan_line(self, c: _Candidate) -> str:
        cross_id = int(c.cross_id)
        cross_name = self._cross_name(cross_id)
        beast = int(c.beast)

        parts: List[str] = []
        for slot, cid, code in zip(c.sel_slots, c.sel_ids, c.sel_codes):
            info = self.chips.get(int(cid))
            nm = (info.name or f"ID:{int(cid)}").strip()
            ch = _code_to_char(int(code))
            # Keep it compact but unambiguous for validation
            parts.append(f"{int(slot)}:{nm}[{ch}]#{int(cid)}")

        chips_s = " ".join(parts) if parts else "(no chips)"
        cross_s = f"{cross_name}({cross_id})" if cross_id != 0 else "Normal(0)"
        return f"cross={cross_s} beast={beast} chips={chips_s}"


    def _build_base_row(self, game_state: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # scalars
        p_hp_open = _as_int(game_state.get("player_health", 0), 0)
        e_hp_open = _as_int(game_state.get("enemy_health", 0), 0)
        turn_idx_open = _as_int(
            game_state.get("turn_index") or game_state.get("turn_idx") or game_state.get("turn") or 0,
            0,
        )

        # grid
        grid_tile = (
            game_state.get("grid_tile_open")
            or game_state.get("grid_tile")
            or game_state.get("grid_state")
            or [0] * 18
        )
        grid_owner = (
            game_state.get("grid_owner_open")
            or game_state.get("grid_owner_state")
            or game_state.get("grid_owner")
            or [2] * 18
        )
        grid_tile = [int(_as_int(x, 0)) for x in _as_list(grid_tile)]
        grid_owner = [int(_as_int(x, 2)) for x in _as_list(grid_owner)]
        if len(grid_tile) != 18:
            grid_tile = (grid_tile + [0] * 18)[:18]
        if len(grid_owner) != 18:
            grid_owner = (grid_owner + [2] * 18)[:18]

        # masks
        fp = game_state.get("folder_used_mask_p_open") or game_state.get("folder_used_mask_p") or [0.0] * 30
        fe = game_state.get("folder_used_mask_e_open") or game_state.get("folder_used_mask_e") or [0.0] * 30
        cp = game_state.get("used_cross_mask_p_open") or game_state.get("used_cross_mask_p") or [0.0] * 11
        ce = game_state.get("used_cross_mask_e_open") or game_state.get("used_cross_mask_e") or [0.0] * 11

        fp = [1.0 if _as_bool(x, False) else 0.0 for x in _as_list(fp)]
        fe = [1.0 if _as_bool(x, False) else 0.0 for x in _as_list(fe)]
        cp = [1.0 if _as_bool(x, False) else 0.0 for x in _as_list(cp)]
        ce = [1.0 if _as_bool(x, False) else 0.0 for x in _as_list(ce)]

        fp = (fp + [0.0] * 30)[:30]
        fe = (fe + [0.0] * 30)[:30]
        cp = (cp + [0.0] * 11)[:11]
        ce = (ce + [0.0] * 11)[:11]

        # held-before (5)
        held_before_id = _as_list(game_state.get("held_before_id") or game_state.get("held_id") or [0] * 5)
        held_before_code = _as_list(game_state.get("held_before_code") or game_state.get("held_code") or [0] * 5)
        held_before_id = [int(_as_int(x, 0)) for x in held_before_id][:5]
        held_before_code = [int(_as_int(x, 0)) for x in held_before_code][:5]
        while len(held_before_id) < 5:
            held_before_id.append(0)
            held_before_code.append(0)

        # window hand (10) + vis
        chip_slots = _as_list(game_state.get("chip_slots", []))[:10]
        chip_codes = _as_list(game_state.get("chip_codes", []))[:10]
        visible_count = max(0, min(10, _as_int(game_state.get("chip_visible_count", 5), 5)))

        while len(chip_slots) < 10:
            chip_slots.append(255)
        while len(chip_codes) < 10:
            chip_codes.append(0)

        window_hand_vis = [1.0 if i < visible_count else 0.0 for i in range(10)]
        window_hand_id = [int(_as_int(x, 0)) for x in chip_slots[:10]]
        window_hand_code = [int(_as_int(x, 0)) for x in chip_codes[:10]]

        row = {
            "p_hp_open": p_hp_open,
            "e_hp_open": e_hp_open,
            "turn_index_open": turn_idx_open,
            "grid_tile_open": grid_tile,
            "grid_owner_open": grid_owner,
            "folder_used_mask_p_open": fp,
            "folder_used_mask_e_open": fe,
            "used_cross_mask_p_open": cp,
            "used_cross_mask_e_open": ce,
            "held_before_id": held_before_id,
            "held_before_code": held_before_code,
            "window_hand_id": window_hand_id,
            "window_hand_code": window_hand_code,
            "window_hand_vis": window_hand_vis,
            "selected_chips_id": [0] * 5,
            "selected_chips_code": [0] * 5,
            "selected_cross": 0,
            "beast_selected": False,
        }

        dbg = {"p_hp": p_hp_open, "e_hp": e_hp_open, "turn": turn_idx_open, "vis": visible_count}
        return row, dbg

    # -------------------------------------------------------------------------
    # Candidate enumeration (FULL permutations)
    # -------------------------------------------------------------------------

    def _get_game_version(self, game_state: dict) -> Optional[str]:
        gv = game_state.get("game_version")
        if isinstance(gv, str) and gv.strip():
            return gv.strip()

        rp = game_state.get("rom_path") or game_state.get("ROM_PATH")
        if isinstance(rp, str):
            s = rp.strip().lower()
            if s.startswith("bn6,"):
                try:
                    v = int(s.split(",", 1)[1])
                    return "Gregar" if v == 0 else "Falzar"
                except Exception:
                    pass
        return None

    def _available_cross_ids(self, game_state: dict) -> List[int]:
        used = [int(_as_int(x, -1)) for x in _as_list(game_state.get("player_used_crosses_list"))]
        used = [x for x in used if 0 <= x <= 10]

        gv = self._get_game_version(game_state)
        if gv in _VERSION_CROSS_IDXS:
            base = list(_VERSION_CROSS_IDXS[gv])
        else:
            base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        return [cid for cid in base if cid not in used]

    def _enumerate_candidates(self, game_state: dict) -> List[_Candidate]:
        chip_slots_raw = _as_list(game_state.get("chip_slots", []))[:10]
        chip_codes_raw = _as_list(game_state.get("chip_codes", []))[:10]
        visible_count = max(0, min(10, _as_int(game_state.get("chip_visible_count", 5), 5)))

        # normalize arrays to length 10
        while len(chip_slots_raw) < 10:
            chip_slots_raw.append(255)
        while len(chip_codes_raw) < 10:
            chip_codes_raw.append(0)

        hand_ids = [int(_as_int(x, 0)) for x in chip_slots_raw[:10]]
        hand_codes = [int(_as_int(x, 0)) for x in chip_codes_raw[:10]]

        # only visible & valid indices participate
        valid_indices = []
        for i in range(visible_count):
            if _chip_valid(hand_ids[i]):
                valid_indices.append(i)

        if not valid_indices:
            # still allow "do nothing"
            return [_Candidate(sel_slots=(), sel_ids=(), sel_codes=(), cross_id=0, beast=0)]

        # Full ordered chip sequences
        chains = _generate_chip_chains_ordered(
            hand_ids,
            hand_codes,
            valid_indices,
            max_len=MAX_CHIPS_PER_TURN,
            max_chains=self.MAX_CHAINS,
        )

        # Cross options: 0 + available (but cross selection is often disabled if beast already happened)
        is_beast_active = _as_bool(game_state.get("is_player_beasted_out", 0), False)
        is_beast_over = _as_bool(game_state.get("is_player_beasted_over", 0), False)
        cross_opts = [0]
        if not is_beast_active:
            # if already in beast, you typically cannot cross change; keep conservative.
            cross_opts += self._available_cross_ids(game_state)

        # Beast opts
        beast_opts = [0] if (is_beast_active or is_beast_over) else [0, 1]

        # Compose candidates with HARD constraint: no cross change + beast in same commit.
        out: List[_Candidate] = []
        filtered_cross_beast = 0
        filtered_slots = 0

        # Quick estimate log
        est = len(chains) * len(cross_opts) * len(beast_opts)
        if self.LOG_ENUM:
            print(
                f"[Plan][perm] vis={visible_count} valid={len(valid_indices)} chains={len(chains)} "
                f"cross_opts={len(cross_opts)} beast_opts={len(beast_opts)} est_candidates={est}"
            )

        for cross_id in cross_opts:
            for beast in beast_opts:
                cross_change = (int(cross_id) != 0)
                beast_pick = (int(beast) == 1)

                if cross_change and beast_pick:
                    filtered_cross_beast += len(chains)
                    continue

                max_allowed = MAX_CHIPS_IF_BEAST if beast_pick else MAX_CHIPS_PER_TURN

                for chain in chains:
                    if len(chain) > max_allowed:
                        filtered_slots += 1
                        continue

                    sel_slots = tuple(int(i) for i in chain)
                    sel_ids = tuple(int(hand_ids[i]) for i in chain)
                    sel_codes = tuple(int(hand_codes[i]) for i in chain)

                    out.append(_Candidate(sel_slots, sel_ids, sel_codes, int(cross_id), int(beast)))

                    if len(out) >= self.MAX_CANDIDATES:
                        print(f"[Plan][perm] Hit candidate safety limit {self.MAX_CANDIDATES}. Truncating.")
                        print(f"[Plan][perm] filtered cross+beast={filtered_cross_beast} filtered slots={filtered_slots}")
                        return out

        if self.LOG_ENUM:
            print(
                f"[Plan][perm] candidates={len(out)} "
                f"filtered cross+beast={filtered_cross_beast} filtered slots={filtered_slots}"
            )
        return out

    def _pick_heuristic_candidate(self, candidates: List[_Candidate]) -> _Candidate:
        best = candidates[0]
        best_s = -1e30
        for c in candidates:
            dmg = 0
            for cid in c.sel_ids:
                dmg += int(self.chips.get(int(cid)).damage)
            s = float(dmg) - 50.0 * float(c.beast) - 10.0 * float(1 if c.cross_id != 0 else 0)
            if s > best_s:
                best_s = s
                best = c
        return best

    def _targets_from_candidate(self, st: Dict[str, Any], game_state: dict, c: _Candidate) -> None:
        st["current_targets"].clear()

        # HARD GUARD: illegal plan (cross change + beast) should never exist, but refuse anyway.
        if int(c.cross_id) != 0 and int(c.beast) == 1:
            print(f"[Plan] ⚠️ Illegal best candidate (cross change + beast). Dropping beast. cross_id={c.cross_id}")
            c = _Candidate(c.sel_slots, c.sel_ids, c.sel_codes, int(c.cross_id), 0)

        # Cross: convert cross_id -> dynamic menu index based on remaining available crosses.
        if int(c.cross_id) != 0:
            used_list = [int(_as_int(x, -1)) for x in _as_list(game_state.get("player_used_crosses_list"))]
            used_list = [x for x in used_list if 0 <= x <= 10]

            gv = self._get_game_version(game_state)
            if gv in _VERSION_CROSS_IDXS:
                base = list(_VERSION_CROSS_IDXS[gv])
            else:
                base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            available = [cid for cid in base if cid not in used_list]
            if int(c.cross_id) in available:
                menu_idx = available.index(int(c.cross_id))
                st["current_targets"].append({"type": "cross", "val": int(menu_idx)})

        # Beast (only if not cross-changing)
        if int(c.beast) == 1:
            st["current_targets"].append({"type": "beast", "val": 0})

        # Chips in order
        for slot in c.sel_slots:
            st["current_targets"].append({"type": "chip", "val": int(slot)})

        # Confirm
        st["current_targets"].append({"type": "ok", "val": 0})

    # -------------------------------------------------------------------------
    # Navigation (includes wrap-safe chip motion + corrected BEAST geometry)
    # -------------------------------------------------------------------------

    def _chip_nav_step(self, curr_idx: int, tgt_idx: int, visible_count: int) -> Optional[str]:
        """
        Deterministic movement inside chip grid (0..9).
        Avoid LEFT from x==0 because your UI wraps to OK (10).
        """
        if visible_count <= 0:
            return None
        if tgt_idx < 0 or tgt_idx >= visible_count:
            return None
        if curr_idx < 0 or curr_idx > 9:
            return None

        curr_x, curr_y = curr_idx % 5, curr_idx // 5
        tgt_x, tgt_y = tgt_idx % 5, tgt_idx // 5

        if curr_y != tgt_y:
            direction = "DOWN" if tgt_y > curr_y else "UP"
            dest_idx = curr_idx + (5 if direction == "DOWN" else -5)
            if 0 <= dest_idx < visible_count:
                return direction

        if curr_x != tgt_x:
            direction = "RIGHT" if tgt_x > curr_x else "LEFT"
            if direction == "LEFT" and curr_x == 0:
                return None
            dest_idx = curr_idx + (1 if direction == "RIGHT" else -1)
            if 0 <= dest_idx < visible_count and (dest_idx // 5) == curr_y:
                return direction

        return None

    def _safe_chip_nudge(self, curr_idx: int, tgt_idx: int, visible_count: int) -> str:
        if visible_count > 5:
            tgt_y = tgt_idx // 5
            curr_y = curr_idx // 5
            if curr_y == 0 and tgt_y == 1 and (curr_idx + 5) < visible_count:
                return "DOWN"
            if curr_y == 1 and tgt_y == 0 and (curr_idx - 5) >= 0:
                return "UP"

        if (curr_idx % 5) == 0:
            return "RIGHT"
        if (curr_idx % 5) == 4:
            return "LEFT"
        return "RIGHT"

    def _navigate_to_next_target(self, st: Dict[str, Any], game_state: dict) -> dict:
        if not st["current_targets"]:
            return self._no_op("no_targets")

        target = st["current_targets"][0]

        curr_idx_raw = game_state.get("selected_menu_index")
        curr_cross_idx = _as_int(game_state.get("selected_cross_index", 0), 0)
        inside_cross_window = _as_bool(game_state.get("inside_cross_window", 0), False)

        if curr_idx_raw is None:
            return self._no_op("no_selected_menu_index")
        curr_idx = int(curr_idx_raw)

        visible_count = max(0, min(10, _as_int(game_state.get("chip_visible_count", 5), 5)))

        # Stall tracking
        if st["nav_last_idx"] == curr_idx:
            st["nav_stall"] += 1
        else:
            st["nav_last_idx"] = curr_idx
            st["nav_stall"] = 0

        # Stall breaker: recenter without causing 0<->10 ping-pong
        if st["nav_stall"] >= 10:
            st["nav_stall"] = 0
            if curr_idx == 10:
                return self._press(st, "LEFT", reason="recenter_from_ok", game_state=game_state)
            if curr_idx == 0:
                return self._press(st, "RIGHT", reason="recenter_from_left_edge", game_state=game_state)
            return self._press(st, "START", reason="recenter_via_start", game_state=game_state)

        # 1) OK
        if target["type"] == "ok":
            if curr_idx == 10:
                st["current_targets"].popleft()
                return self._press(st, "A", reason="select_ok", game_state=game_state)
            if inside_cross_window:
                return self._press(st, "DOWN", reason="exit_cross_window_to_ok_path", game_state=game_state)
            return self._press(st, "START", reason="navigate_to_ok", game_state=game_state)

        # 2) BEAST (FIXED: BEAST is DOWN from OK)
        if target["type"] == "beast":
            if inside_cross_window:
                return self._press(st, "DOWN", reason="exit_cross_window_for_beast", game_state=game_state)

            if curr_idx == 11:
                st["current_targets"].popleft()
                return self._press(st, "A", reason="select_beast", game_state=game_state)

            if curr_idx == 10:
                return self._press(st, "DOWN", reason="navigate_ok_to_beast", game_state=game_state)

            # go to OK first
            return self._press(st, "START", reason="navigate_to_ok_for_beast", game_state=game_state)

        # 3) CROSS (dynamic menu index)
        if target["type"] == "cross":
            desired_menu_idx = int(target["val"])

            used_list = [int(_as_int(x, -1)) for x in _as_list(game_state.get("player_used_crosses_list"))]
            used_list = [x for x in used_list if 0 <= x <= 10]

            gv = self._get_game_version(game_state)
            if gv in _VERSION_CROSS_IDXS:
                base = list(_VERSION_CROSS_IDXS[gv])
            else:
                base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            available_cross_ids = [cid for cid in base if cid not in used_list]
            avail_n = len(available_cross_ids)

            if desired_menu_idx < 0 or desired_menu_idx >= avail_n:
                st["current_targets"].popleft()
                return self._no_op("cross_idx_oob", extra={"desired": desired_menu_idx, "avail_n": avail_n})

            if not inside_cross_window:
                # enter cross window (your existing behavior)
                if curr_idx == 11:
                    return self._press(st, "UP", reason="navigate_up_to_cross_window", game_state=game_state)
                if curr_idx == 10:
                    return self._press(st, "LEFT", reason="navigate_left_to_cross_window", game_state=game_state)
                return self._press(st, "UP", reason="navigate_up_to_cross_window", game_state=game_state)

            if curr_cross_idx == desired_menu_idx:
                st["current_targets"].popleft()
                return self._press(st, "A", reason="select_cross", game_state=game_state)

            diff = desired_menu_idx - curr_cross_idx
            if diff > 0:
                return self._press(st, "DOWN", reason="navigate_down_to_cross", game_state=game_state)
            if diff < 0:
                return self._press(st, "UP", reason="navigate_up_to_cross", game_state=game_state)

        # 4) CHIP
        if target["type"] == "chip":
            tgt_idx = int(target["val"])

            if inside_cross_window:
                return self._press(st, "DOWN", reason="exit_cross_window_for_chip", game_state=game_state)

            if curr_idx == 11:
                return self._press(st, "UP", reason="leave_beast_to_ok_or_grid", game_state=game_state)
            if curr_idx == 10:
                return self._press(st, "LEFT", reason="leave_ok_to_grid", game_state=game_state)

            if curr_idx == tgt_idx:
                st["current_targets"].popleft()
                return self._press(st, "A", reason="select_chip", game_state=game_state)

            step = self._chip_nav_step(curr_idx, tgt_idx, visible_count)
            if step is not None:
                return self._press(st, step, reason="chip_step", game_state=game_state)

            nudge = self._safe_chip_nudge(curr_idx, tgt_idx, visible_count)
            return self._press(st, nudge, reason="chip_nudge", game_state=game_state)

        return self._no_op("unhandled_target", extra={"target": target})
