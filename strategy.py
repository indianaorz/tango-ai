import time, base64
from collections import defaultdict, deque
from io import BytesIO
import torch
from PIL import Image

import utils

class DRLAgentStrategy:
    """Actor-Critic wrapper with per-env observation state, cross-select menu flows, and sticky-hold combat logic."""

    def __init__(self,
                 key_bit_positions,
                 discrete_actions,
                 util_fns,
                 actor_critic_model,
                 device,
                 frame_h: int,
                 frame_w: int,
                 seq_len_frames: int,
                 *,
                 max_health_config: float,
                 max_charge_config: float,
                 max_cust_gauge_config: float,
                 cross_select_default: int = 0, use_images: bool = True):
        self.use_images = bool(use_images)
        self.key_bits        = key_bit_positions
        self.discrete_actions= discrete_actions
        self.bin16           = util_fns["int_to_binary_string"]
        self.map_discrete    = util_fns["map_discrete_action_to_buttons"]
        self.preprocess_frame= util_fns.get("preprocess_frame")

        self.model      = actor_critic_model
        self.dev        = device
        self.H, self.W  = frame_h, frame_w
        self.T          = seq_len_frames

        # normalization constants
        self.max_hp     = max_health_config
        self.max_charge = max_charge_config
        self.max_cust   = max_cust_gauge_config

        # per-env rolling buffers
        self._frames = defaultdict(lambda: deque(maxlen=self.T))  # [C,H,W]
        self._feats  = defaultdict(lambda: deque(maxlen=self.T))  # [F]

        # menu FSMs & holds
        self._skip_legacy = defaultdict(lambda: {'phase': 'INIT', 'ts': 0.0})

        # cross-select state
        self._cross_mode  = defaultdict(lambda: cross_select_default)  # int per port
        self._cross_seq   = defaultdict(lambda: {'active': False, 'steps': [], 'idx': 0, 'ts': 0.0, 'done_once': False})

        # held combat buttons (persist across frames outside menus)
        self._held_bits = defaultdict(int)

    # -------------------------- public API --------------------------- #
    def set_cross_select(self, port: int, mode: int) -> None:
        """Set cross selection mode for this port (0..5)."""
        self._cross_mode[port] = int(mode)
        cs = self._cross_seq[port]
        cs.update(active=False, steps=[], idx=0, ts=0.0, done_once=False)

    # -------------------------- helpers ------------------------------ #
    def _append_obs(self, port, pil_img, raw):
        if self.use_images:
            f_t = self.preprocess_frame(pil_img, self.H, self.W)  # [3,H,W] default
        else:
            # C=0 frame placeholder (valid zero-width channel tensor)
            f_t = torch.zeros((0, self.H, self.W), dtype=torch.float32)
        self._frames[port].append(f_t)

        pg = raw.get('player_grid_position', [1,1])
        eg = raw.get('enemy_grid_position',  [4,1])
        cust = raw.get('cust_gauge', raw.get('cust_gage', 0))

        feat = [
            raw.get('player_health',0)/self.max_hp,
            raw.get('enemy_health',0)/self.max_hp,
            (pg[0]-1)/5, (pg[1]-1)/2,
            (eg[0]-1)/5, (eg[1]-1)/2,
            raw.get('player_charge',0)/self.max_charge,
            raw.get('enemy_charge',0)/self.max_charge,
            cust/self.max_cust,
            1.0 if raw.get('is_player_beasted_out') else 0.0,
            1.0 if raw.get('is_enemy_beasted_out') else 0.0,
        ]
        self._feats[port].append(torch.tensor(feat, dtype=torch.float32))

        while len(self._frames[port]) < self.T:
            self._frames[port].appendleft(torch.zeros_like(f_t))
            self._feats [port].appendleft(torch.zeros_like(self._feats[port][-1]))

    def _get_seq_tensors(self, port):
        frames = torch.stack(list(self._frames[port]), 0).unsqueeze(0).to(self.dev) # [1,T,C,H,W]
        feats  = torch.stack(list(self._feats [port]), 0).unsqueeze(0).to(self.dev) # [1,T,F]
        return frames, feats

    # ------------------ legacy RETURN <-> Z loop --------------------- #
    def _legacy_menu_skip(self, port):
        st  = self._skip_legacy[port]
        now = time.monotonic()  # monotonic > time.time for timers

        # durations tuned to work at 5 Hz too
        HOLD_RETURN_S = 0.60
        PAUSE_S       = 0.25

        if st['phase'] == 'INIT':
            st.update(phase='HOLD_RETURN', ts=now)
            return 1 << self.key_bits['RETURN']  # press immediately on entry

        if st['phase'] == 'HOLD_RETURN':
            if (now - st['ts']) < HOLD_RETURN_S:
                return 1 << self.key_bits['RETURN']  # keep holding across frames
            st.update(phase='PAUSE1', ts=now)
            return 0

        if st['phase'] == 'PAUSE1':
            if (now - st['ts']) < PAUSE_S:
                return 0
            st.update(phase='TAP_Z', ts=now)
            return 1 << self.key_bits['Z']  # TAP immediately on entry

        if st['phase'] == 'TAP_Z':
            # one-frame tap already sent; go to pause
            st.update(phase='PAUSE2', ts=now)
            return 0

        if st['phase'] == 'PAUSE2':
            if (now - st['ts']) < PAUSE_S:
                return 0
            st.update(phase='HOLD_RETURN', ts=now)
            return 1 << self.key_bits['RETURN']  # press immediately on entry

        st.update(phase='INIT', ts=now)
        return 0


    # --------------------- cross-select runner ----------------------- #
    def _build_cross_steps(self, mode: int):
        """
        Build sequence as one-frame taps followed by 2s releases.
        Mode mapping (>=1):
          1: UP → Z → START → Z
          2: UP → DOWN → Z → START → Z
          3: UP → DOWN×2 → Z → START → Z
          4: UP → DOWN×3 → Z → START → Z
          5: UP → DOWN×4 → Z → START → Z
        """
        WAIT = 2.0  # seconds of release after each tap
        steps = []

        # small settle before the first action
        steps += [('PAUSE', WAIT)]

        # UP tap
        steps += [('UP', 0.0), ('PAUSE', WAIT)]

        # DOWN taps (mode-1 times)
        downs = max(0, mode - 1)
        for _ in range(downs):
            steps += [('DOWN', 0.0), ('PAUSE', WAIT)]

        # Z, RETURN, Z — each with a release window after
        steps += [('Z', 0.0),      ('PAUSE', WAIT)]
        steps += [('RETURN', 0.0), ('PAUSE', WAIT)]
        steps += [('Z', 0.0),      ('PAUSE', WAIT)]

        # final settle
        steps += [('PAUSE', WAIT)]
        return steps

    def _ensure_cross_seq(self, port):
        cs = self._cross_seq[port]
        if not cs['active'] and not cs['done_once']:
            mode = int(self._cross_mode[port])
            if mode <= 0:
                return
            cs['steps'] = self._build_cross_steps(mode)
            cs['idx']   = 0
            cs['ts']    = 0.0
            cs['active']= True

    def _run_cross_seq(self, port):
        """
        Execute one frame of the cross sequence.
        Key steps emit a press for exactly one frame, then immediately advance
        to a PAUSE step (release-all) that lasts for its duration.
        """
        cs = self._cross_seq[port]
        if not cs['active']:
            return 0

        now = time.time()
        if cs['ts'] == 0.0:
            cs['ts'] = now

        while cs['idx'] < len(cs['steps']):
            name, dur = cs['steps'][cs['idx']]

            if name == 'PAUSE':
                # stay released for 'dur' seconds
                if (now - cs['ts']) < dur:
                    return 0  # release all
                # finished this pause → next step
                cs['idx'] += 1
                cs['ts']   = now
                continue

            # name is a KEY tap (UP/DOWN/Z/RETURN) → emit once, then move to next step immediately
            kb = self.key_bits
            mask = 0
            if name == 'UP':      mask = 1 << kb['UP']
            elif name == 'DOWN':  mask = 1 << kb['DOWN']
            elif name == 'Z':     mask = 1 << kb['Z']
            elif name == 'RETURN':mask = 1 << kb['RETURN']

            # advance to next step (which should be a PAUSE) and reset timer so next frame is release
            cs['idx'] += 1
            cs['ts']   = now
            return mask  # one-frame press

        # finished
        cs['active']    = False
        cs['done_once'] = True
        return 0

    def _menu_controller(self, port):
        """
        Inside the custom window:
        - Run cross sequence exclusively (including pauses) if configured and not yet done.
        - Otherwise, use legacy skip.
        """
        self._held_bits[port] = 0  # clear combat holds in menus
        self._ensure_cross_seq(port)

        cs = self._cross_seq[port]
        mode = int(self._cross_mode[port])

        if mode > 0 and (cs['active'] or not cs['done_once']):
            return self._run_cross_seq(port)

        return self._legacy_menu_skip(port)

    # ------------------------- main step ----------------------------- #
    def decide_action(self, port, game_state):
        inside_menu = bool(game_state.get("inside_window", False))

        if inside_menu:
            key_bits = self._menu_controller(port)
            return {
                "button_command": {"type": "key_press", "key": self.bin16(key_bits)},
                "action_idx": None, "log_prob": None, "value": None,
                "stacked_frames_tensor": None, "game_features_tensor": None,
            }

        # leaving menu: reset menu FSMs for next time
        if self._skip_legacy[port].get('phase') != 'INIT':
            self._skip_legacy[port] = {'phase': 'INIT', 'ts': 0.0}
        self._cross_seq[port].update(active=False, idx=0, ts=0.0, steps=[], done_once=False)

        # 1) encode observation
        pil = None
        b64 = game_state.get("image")
        if b64:
            try:
                pil = Image.open(BytesIO(base64.b64decode(b64)))
            except Exception:
                pass
        self._append_obs(port, pil, game_state)
        seq_f, seq_feat = self._get_seq_tensors(port)

        # 2) model forward for acting — IMPORTANT: do NOT change train/eval here.
        with torch.no_grad():
            a_idx, log_p, _, value, _ = self.model.get_action_and_value(seq_f, seq_feat)

        action_name = self.discrete_actions[a_idx.item()]
        frame_mask  = int(self.map_discrete(a_idx.item(), self.discrete_actions, self.key_bits), 2)

        # 3) sticky holds — explicit HOLD_* sets, explicit RELEASE_* clears
        X_bit = 1 << self.key_bits["X"]
        Z_bit = 1 << self.key_bits["Z"]

        if action_name == "HOLD_X":
            self._held_bits[port] |= X_bit
        elif action_name == "HOLD_Z":
            self._held_bits[port] |= Z_bit
        elif action_name == "RELEASE_X":
            self._held_bits[port] &= ~X_bit
        elif action_name == "RELEASE_Z":
            self._held_bits[port] &= ~Z_bit

        final_mask = frame_mask | self._held_bits[port]

        # Map the actually executed mask back to the closest discrete action
        exec_idx_int = self.preprocess_frame.__self__.utils.bitmask_to_action_index(final_mask) \
            if hasattr(self.preprocess_frame, "__self__") else utils.bitmask_to_action_index(final_mask)

        exec_idx_t = torch.tensor(exec_idx_int, device=self.dev)

        # If the executed mask doesn't match the originally sampled action, recompute logπ for exec action
        if exec_idx_int != a_idx.item():
            with torch.no_grad():
                _, log_p, _, value, _ = self.model.get_action_and_value(seq_f, seq_feat, exec_idx_t)
            a_idx = exec_idx_t  # make sure buffer stores the executed action index

        return {
            "button_command": {"type":"key_press","key": self.bin16(final_mask)},
            "action_idx": a_idx,
            "log_prob":   log_p,
            "value":      value,
            "stacked_frames_tensor": seq_f,
            "game_features_tensor":  seq_feat,
        }

    def reset_state(self, port):
        self._frames[port].clear()
        self._feats [port].clear()
        self._skip_legacy[port] = {'phase':'INIT','ts':0.0}
        self._held_bits[port]   = 0
        self._cross_seq[port].update(active=False, steps=[], idx=0, ts=0.0, done_once=False)

    def encode_for_policy(self, port: int, game_state: dict):
        """
        Public: update per-port rolling buffers from a raw game_state and
        return [1,T,C,H,W] frames + [1,T,F] features tensors on self.dev.
        """
        pil = None
        b64 = game_state.get("image")
        if b64:
            try:
                pil = Image.open(BytesIO(base64.b64decode(b64)))
            except Exception:
                pil = None
        self._append_obs(port, pil, game_state)
        return self._get_seq_tensors(port)

# -----------------------------------------------------------------------------
# Base Strategy ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class ActionStrategy:
    """Common interface all concrete strategies must expose."""

    def __init__(self,
                 key_bit_positions: dict[str, int],
                 discrete_actions: list[str],
                 util_fns: dict):
        # button‑bit mapping & action‑list come from config
        self.key_bits        = key_bit_positions        # e.g. {'UP':6, ...}
        self.discrete_actions= discrete_actions          # e.g. ["NO_OP","UP", ...]
        # helper lambdas passed in from utils.py so we keep strategy self‑contained
        self.bin16           = util_fns["int_to_binary_string"]
        self.map_discrete    = util_fns["map_discrete_action_to_buttons"]
        self.preprocess_frame= util_fns.get("preprocess_frame")  # DRL only
        self.rand_skip_act   = util_fns.get("generate_random_action_for_skip_strategy")

    # interface -----------------------------------------------------------------
    def decide_action(self, port: int, game_state: dict):  # -> dict (see callers)
        raise NotImplementedError

    def reset_state(self, port: int):                      # optional
        pass

# -----------------------------------------------------------------------------
# Skip‑and‑Random Strategy (simple hand‑crafted baseline) -----------------------
# -----------------------------------------------------------------------------

class SkipAndRandomStrategy(ActionStrategy):
    """Spams RETURN/Z in chip‑select windows and else presses random movement."""

    def __init__(self,
                 key_bit_positions,
                 discrete_actions,
                 util_fns,
                 random_action_keys):
        super().__init__(key_bit_positions, discrete_actions, util_fns)
        self.random_keys    = random_action_keys
        # per‑port FSM state → (state_code, timestamp)
        self._state         = defaultdict(lambda: (0, 0.))

    # ------------------------------------------------------------------
    def _fsm_in_window(self, port):
        code, ts = self._state[port]
        now      = time.time()
        key_bits = 0
        if code == 0:                       # just entered window -> wait 0.5 s
            self._state[port] = (1, now)
        elif code == 1 and now-ts >= 2:   # send RETURN, begin spamming
            key_bits          = 1 << self.key_bits['RETURN']
            self._state[port] = (2, now)
        elif code == 2:                     # keep RETURN for 1 s then Z
            if now-ts >= 1.0:
                key_bits          = 1 << self.key_bits['Z']
                self._state[port] = (3, 0.)
            else:
                key_bits          = 1 << self.key_bits['RETURN']
        # state 3 == finished, send NO_OP
        return key_bits

    # ------------------------------------------------------------------
    def decide_action(self, port, game_state):
        inside_win = bool(game_state.get('inside_window', False))
        if inside_win:
            key_bits = self._fsm_in_window(port)
        else:
            # reset FSM if we left the window
            if self._state[port][0] != 0:
                self._state[port] = (0, 0.)
            key_bits = int(self.rand_skip_act(self.key_bits, self.random_keys), 2)

        return {
            'button_command': {'type': 'key_press', 'key': self.bin16(key_bits)},
            'action_idx': None, 'log_prob': None, 'value': None,
            'stacked_frames_tensor': None, 'game_features_tensor': None
        }

    def reset_state(self, port):
        self._state[port] = (0, 0.)

# -----------------------------------------------------------------------------
# Scripted Opponents
#   • ScriptedStandAndShootStrategy: never moves; taps X on a cadence.
#   • ScriptedWanderStrategy       : randomly walks; never shoots.
#
# Both reuse a robust in-window skipper so battles start reliably.
# -----------------------------------------------------------------------------

from collections import defaultdict
import time
import random

class _MenuSkipController:
    def __init__(self, key_bits: dict[str,int]):
        self.key_bits = key_bits
        self._state   = defaultdict(lambda: {"phase":"INIT","t":0.0})

    def step(self, port: int) -> int:
        kb  = self.key_bits
        st  = self._state[port]
        now = time.monotonic()

        HOLD_RETURN_S = 0.60
        PAUSE_S       = 0.25

        if st["phase"] == "INIT":
            st.update(phase="HOLD_RETURN", t=now)
            return 1 << kb["RETURN"]

        if st["phase"] == "HOLD_RETURN":
            if (now - st["t"]) < HOLD_RETURN_S:
                return 1 << kb["RETURN"]
            st.update(phase="PAUSE1", t=now)
            return 0

        if st["phase"] == "PAUSE1":
            if (now - st["t"]) < PAUSE_S:
                return 0
            st.update(phase="TAP_Z", t=now)
            return 1 << kb["Z"]  # one-frame tap

        if st["phase"] == "TAP_Z":
            st.update(phase="PAUSE2", t=now)
            return 0

        if st["phase"] == "PAUSE2":
            if (now - st["t"]) < PAUSE_S:
                return 0
            st.update(phase="HOLD_RETURN", t=now)
            return 1 << kb["RETURN"]

        st.update(phase="INIT", t=now)
        return 0


class ScriptedStandAndShootStrategy(ActionStrategy):
    """
    Deterministic bot: stands still and taps X at a fixed cadence.
    - Inside chip-select window: run the shared menu skipper.
    - In battle: send a one-frame X press every FIRE_EVERY_MS.
    """
    def __init__(self,
                 key_bit_positions: dict[str,int],
                 discrete_actions: list[str],
                 util_fns: dict,
                 *,
                 fire_every_ms: int = 500,
                 seed: int | None = None):
        super().__init__(key_bit_positions, discrete_actions, util_fns)
        self._menu = _MenuSkipController(key_bit_positions)
        self._fire_interval = max(100, int(fire_every_ms)) / 1000.0
        self._last_fire = defaultdict(lambda: 0.0)
        if seed is not None:
            random.seed(seed)

    def decide_action(self, port: int, game_state: dict):
        if bool(game_state.get("inside_window", False)):
            mask = self._menu.step(port)
            return {"button_command": {"type":"key_press","key": self.bin16(mask)},
                    "action_idx": None, "log_prob": None, "value": None,
                    "stacked_frames_tensor": None, "game_features_tensor": None}

        # In battle: tap X at cadence; no movement.
        now = time.monotonic()
        mask = 0
        if now - self._last_fire[port] >= self._fire_interval:
            self._last_fire[port] = now
            mask |= 1 << self.key_bits["X"]  # one-frame tap
        return {"button_command": {"type":"key_press","key": self.bin16(mask)},
                "action_idx": None, "log_prob": None, "value": None,
                "stacked_frames_tensor": None, "game_features_tensor": None}

    def reset_state(self, port: int):
        self._last_fire[port] = 0.0


class ScriptedWanderStrategy(ActionStrategy):
    """
    Stochastic bot: randomly walks; never shoots.
    - Inside chip-select window: shared menu skipper.
    - In battle: choose a direction and hold for HOLD_MS; occasionally NO_OP.
    Deterministic with a fixed seed for reproducibility.
    """
    def __init__(self,
                 key_bit_positions: dict[str,int],
                 discrete_actions: list[str],
                 util_fns: dict,
                 *,
                 hold_ms: int = 350,
                 noop_chance: float = 0.15,
                 seed: int = 1337):
        super().__init__(key_bit_positions, discrete_actions, util_fns)
        self._menu = _MenuSkipController(key_bit_positions)
        self._hold_s = max(50, int(hold_ms)) / 1000.0
        self._noop_p = min(0.9, max(0.0, noop_chance))
        self._dir_by_port = defaultdict(lambda: {"dir": None, "t": 0.0})
        self._rng = random.Random(seed)

        self._dir_keys = ["UP","DOWN","LEFT","RIGHT"]

    def _sample_dir_or_none(self):
        if self._rng.random() < self._noop_p:
            return None
        return self._rng.choice(self._dir_keys)

    def decide_action(self, port: int, game_state: dict):
        if bool(game_state.get("inside_window", False)):
            mask = self._menu.step(port)
            return {"button_command": {"type":"key_press","key": self.bin16(mask)},
                    "action_idx": None, "log_prob": None, "value": None,
                    "stacked_frames_tensor": None, "game_features_tensor": None}

        st  = self._dir_by_port[port]
        now = time.monotonic()
        if st["dir"] is None or (now - st["t"]) >= self._hold_s:
            st["dir"] = self._sample_dir_or_none()
            st["t"]   = now

        mask = 0
        if st["dir"] is not None and st["dir"] in self.key_bits:
            mask |= 1 << self.key_bits[st["dir"]]
        return {"button_command": {"type":"key_press","key": self.bin16(mask)},
                "action_idx": None, "log_prob": None, "value": None,
                "stacked_frames_tensor": None, "game_features_tensor": None}

    def reset_state(self, port: int):
        self._dir_by_port[port] = {"dir": None, "t": 0.0}


# scripted_more.py
import time, random
from collections import defaultdict
from strategy import ActionStrategy, _MenuSkipController

# 1) Strafe & Shoot (oscillate rows; fire on cadence or on alignment)
class ScriptedStrafeAndShoot(ActionStrategy):
    def __init__(self, key_bit_positions, discrete_actions, util_fns,
                 *, swap_every_ms=500, fire_every_ms=600, align_only=False, seed=123):
        super().__init__(key_bit_positions, discrete_actions, util_fns)
        self._menu = _MenuSkipController(key_bit_positions)
        self._swap_s = max(120, int(swap_every_ms)) / 1000.0
        self._fire_s = max(120, int(fire_every_ms)) / 1000.0
        self._align_only = bool(align_only)
        self._last_swap = defaultdict(lambda: 0.0)
        self._last_fire = defaultdict(lambda: 0.0)
        self._dir = defaultdict(lambda: "UP")   # current strafe direction
        random.seed(seed)

    def decide_action(self, port, game_state):
        if bool(game_state.get("inside_window", False)):
            return {"button_command":{"type":"key_press","key": self.bin16(self._menu.step(port))},
                    "action_idx":None,"log_prob":None,"value":None,"stacked_frames_tensor":None,"game_features_tensor":None}

        now = time.monotonic()
        mask = 0

        # Strafe toggle
        if now - self._last_swap[port] >= self._swap_s:
            self._last_swap[port] = now
            self._dir[port] = "DOWN" if self._dir[port] == "UP" else "UP"
        dir_bit = self.key_bits[self._dir[port]]
        mask |= (1 << dir_bit)

        # Fire logic
        aligned = False
        try:
            pr = game_state.get("player_grid_position",[1,1])[1]
            er = game_state.get("enemy_grid_position",[4,1])[1]
            aligned = (pr == er)
        except Exception:
            pass

        if now - self._last_fire[port] >= self._fire_s and (aligned or not self._align_only):
            self._last_fire[port] = now
            mask |= (1 << self.key_bits["X"])  # one-frame tap

        return {"button_command":{"type":"key_press","key": self.bin16(mask)},
                "action_idx":None,"log_prob":None,"value":None,"stacked_frames_tensor":None,"game_features_tensor":None}

    def reset_state(self, port:int):
        self._last_swap[port] = 0.0
        self._last_fire[port] = 0.0
        self._dir[port] = "UP"


# 2) Random Walk + Shoot (random move holds; periodic shots)
class ScriptedRandomWalkAndShoot(ActionStrategy):
    def __init__(self, key_bit_positions, discrete_actions, util_fns,
                 *, hold_ms=350, noop_chance=0.10, fire_every_ms=500, seed=1337):
        super().__init__(key_bit_positions, discrete_actions, util_fns)
        self._menu = _MenuSkipController(key_bit_positions)
        self._hold_s = max(60, int(hold_ms)) / 1000.0
        self._noop_p = min(0.9, max(0.0, noop_chance))
        self._fire_s = max(120, int(fire_every_ms)) / 1000.0
        self._state = defaultdict(lambda: {"dir": None, "t": 0.0})
        self._last_fire = defaultdict(lambda: 0.0)
        self._rng = random.Random(seed)
        self._dirs = ["UP","DOWN","LEFT","RIGHT"]

    def decide_action(self, port, game_state):
        if bool(game_state.get("inside_window", False)):
            return {"button_command":{"type":"key_press","key": self.bin16(self._menu.step(port))},
                    "action_idx":None,"log_prob":None,"value":None,"stacked_frames_tensor":None,"game_features_tensor":None}

        st  = self._state[port]
        now = time.monotonic()

        # Sample/refresh direction
        if st["dir"] is None or (now - st["t"]) >= self._hold_s:
            st["dir"] = None if self._rng.random() < self._noop_p else self._rng.choice(self._dirs)
            st["t"]   = now

        mask = 0
        if st["dir"] and st["dir"] in self.key_bits:
            mask |= 1 << self.key_bits[st["dir"]]

        # Periodic fire
        if now - self._last_fire[port] >= self._fire_s:
            self._last_fire[port] = now
            mask |= 1 << self.key_bits["X"]

        return {"button_command":{"type":"key_press","key": self.bin16(mask)},
                "action_idx":None,"log_prob":None,"value":None,"stacked_frames_tensor":None,"game_features_tensor":None}

    def reset_state(self, port:int):
        self._state[port] = {"dir": None, "t": 0.0}
        self._last_fire[port] = 0.0


# 3) Charge-then-Release (stand or strafe while charging; release on align or timeout)
# scripted_more.py
import time, random
from collections import defaultdict
from strategy import ActionStrategy, _MenuSkipController

class ScriptedChargeAndRelease(ActionStrategy):
    """
    Hold X to charge, optionally move, then release (stop holding) when:
      • player_charge >= charge_level AND (aligned OR timeout)
    After release, restart charging.

    Args:
      move_mode: "still" | "strafe" | "stand" (alias of "still")
      charge_level: 1..3 (your config.CHARGE_MAX_LEVEL)
      align_release: if True, prefer to release only when rows align
      max_charge_ms: safety timeout to release even if not aligned
      strafe_ms: toggle UP/DOWN every strafe_ms when move_mode="strafe"
    """
    def __init__(self, key_bit_positions, discrete_actions, util_fns,
                 *, move_mode="still", charge_level=3, align_release=True,
                 max_charge_ms=3500, strafe_ms=500, seed=7):
        super().__init__(key_bit_positions, discrete_actions, util_fns)
        self._menu = _MenuSkipController(key_bit_positions)

        # Normalized/validated args
        self._lvl           = max(1, int(charge_level))
        self._align_release = bool(align_release)
        self._max_charge_s  = max(300, int(max_charge_ms)) / 1000.0
        self._move_mode     = "still" if move_mode in ("still", "stand") else move_mode
        self._strafe_s      = max(120, int(strafe_ms)) / 1000.0
        random.seed(seed)

        # Per-port state
        self._state        = defaultdict(lambda: "CHARGING")     # "CHARGING" | "RELEASING"
        self._start_t      = defaultdict(lambda: 0.0)            # when current charge cycle began
        self._release_t    = defaultdict(lambda: 0.0)            # when we released
        self._dir          = defaultdict(lambda: "UP")
        self._last_swap    = defaultdict(lambda: 0.0)

    def _aligned_rows(self, gs) -> bool:
        try:
            pr = gs.get("player_grid_position", [1,1])[1]
            er = gs.get("enemy_grid_position",  [4,1])[1]
            return pr == er
        except Exception:
            return False

    def _movement_mask(self, port, now) -> int:
        if self._move_mode != "strafe":
            return 0  # stand still
        if now - self._last_swap[port] >= self._strafe_s:
            self._last_swap[port] = now
            self._dir[port] = "DOWN" if self._dir[port] == "UP" else "UP"
        return (1 << self.key_bits[self._dir[port]])

    def decide_action(self, port, game_state):
        # 1) Menu skip unchanged
        if bool(game_state.get("inside_window", False)):
            mask = self._menu.step(port)
            return {"button_command": {"type":"key_press","key": self.bin16(mask)},
                    "action_idx": None, "log_prob": None, "value": None,
                    "stacked_frames_tensor": None, "game_features_tensor": None}

        now = time.monotonic()
        if self._start_t[port] == 0.0:
            self._start_t[port] = now

        mask = self._movement_mask(port, now)

        # --- core logic ---
        # Use our own player's charge for this window
        cur_lvl = int(game_state.get("player_charge", 0))
        aligned = self._aligned_rows(game_state)
        timeout = (now - self._start_t[port]) >= self._max_charge_s

        state = self._state[port]

        if state == "CHARGING":
            # hold X every frame to build charge
            mask |= (1 << self.key_bits["X"])

            # release condition: reached level AND (aligned or timeout)
            if cur_lvl >= self._lvl and ((self._align_release and aligned) or timeout):
                self._state[port]     = "RELEASING"
                self._release_t[port] = now
                # IMPORTANT: do NOT press X on the release frame — just stop holding.
                # So we *do not* include X in mask this frame; the switch to RELEASING
                # means next frame we won't hold X, causing the charge shot to fire.
                mask &= ~(1 << self.key_bits["X"])

        elif state == "RELEASING":
            # One frame (or a short window) of not holding X to let the charge shot fire
            # Keep not holding X for ~100ms to be safe at low polling rates
            if (now - self._release_t[port]) >= 0.10:
                self._state[port]  = "CHARGING"
                self._start_t[port]= now
            # no X bit while releasing → mask unchanged (movement only)

        return {"button_command": {"type":"key_press","key": self.bin16(mask)},
                "action_idx": None, "log_prob": None, "value": None,
                "stacked_frames_tensor": None, "game_features_tensor": None}

    def reset_state(self, port:int):
        self._state[port]      = "CHARGING"
        self._start_t[port]    = 0.0
        self._release_t[port]  = 0.0
        self._last_swap[port]  = 0.0
        self._dir[port]        = "UP"
