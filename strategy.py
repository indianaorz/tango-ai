import time, base64
from collections import defaultdict, deque
from io import BytesIO
import torch
from PIL import Image


class DRLAgentStrategy:
    """Actor-Critic wrapper with per-env observation state + robust menu-skip FSM."""

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
                 max_cust_gauge_config: float):

        self.key_bits         = key_bit_positions
        self.discrete_actions = discrete_actions
        self.bin16            = util_fns["int_to_binary_string"]
        self.map_discrete     = util_fns["map_discrete_action_to_buttons"]
        self.preprocess_frame = util_fns.get("preprocess_frame")

        self.model      = actor_critic_model
        self.dev        = device
        self.H, self.W  = frame_h, frame_w
        self.T          = seq_len_frames

        # normalisation constants
        self.max_hp     = max_health_config
        self.max_charge = max_charge_config
        self.max_cust   = max_cust_gauge_config

        # per-env rolling buffers
        self._frames = defaultdict(lambda: deque(maxlen=self.T))  # [C,H,W] tensors
        self._feats  = defaultdict(lambda: deque(maxlen=self.T))  # 1-D tensors

        # per-env menu FSM + held bits
        self._skip       = defaultdict(lambda: {'phase': 'INIT', 'ts': 0.0})
        self._held_bits  = defaultdict(int)

    # ──────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────
    def _append_obs(self, port, pil_img, raw):
        f_t = self.preprocess_frame(pil_img, self.H, self.W)  # [C,H,W], float 0-1
        self._frames[port].append(f_t)

        # scalar features (keep order stable)
        pg = raw.get('player_grid_position', [1, 1])
        eg = raw.get('enemy_grid_position',  [4, 1])
        cust = raw.get('cust_gauge', raw.get('cust_gage', 0))

        feat = [
            raw.get('player_health', 0) / self.max_hp,
            raw.get('enemy_health',  0) / self.max_hp,
            (pg[0] - 1) / 5, (pg[1] - 1) / 2,
            (eg[0] - 1) / 5, (eg[1] - 1) / 2,
            raw.get('player_charge', 0) / self.max_charge,
            raw.get('enemy_charge',  0) / self.max_charge,
            cust / self.max_cust,
            1.0 if raw.get('is_player_beasted_out') else 0.0,
            1.0 if raw.get('is_enemy_beasted_out') else 0.0,
        ]
        self._feats[port].append(torch.tensor(feat, dtype=torch.float32))

        # left-pad early timesteps
        while len(self._frames[port]) < self.T:
            self._frames[port].appendleft(torch.zeros_like(f_t))
            self._feats [port].appendleft(torch.zeros_like(self._feats[port][-1]))

    def _get_seq_tensors(self, port):
        frames = torch.stack(list(self._frames[port]), 0).unsqueeze(0).to(self.dev)  # [1,T,C,H,W]
        feats  = torch.stack(list(self._feats [port]), 0).unsqueeze(0).to(self.dev)  # [1,T,F]
        return frames, feats

    # ──────────────────────────────────────────────────────────────────
    # robust menu skip (Return → pause → Z → pause → loop)
    # ──────────────────────────────────────────────────────────────────
    def _ensure_skip_shape(self, port):
        """Migrate any legacy _skip[port] dicts to the new {'phase','ts'} shape."""
        st = self._skip[port]
        if 'phase' not in st or 'ts' not in st:
            self._skip[port] = {'phase': 'INIT', 'ts': 0.0}
        return self._skip[port]

    def _skip_menu(self, port):
        st  = self._ensure_skip_shape(port)
        now = time.time()

        def elapsed(sec): return (now - st['ts']) >= sec

        phase = st['phase']

        if phase == 'INIT':
            if st['ts'] == 0.0:
                st['ts'] = now
                return 0
            if elapsed(0.30):                    # settle 300 ms
                st['phase'] = 'HOLD_RETURN'; st['ts'] = now
            return 0

        if phase == 'HOLD_RETURN':
            if not elapsed(0.22):                # hold Return ~220 ms
                return 1 << self.key_bits['RETURN']
            st['phase'] = 'PAUSE1'; st['ts'] = now
            return 0

        if phase == 'PAUSE1':
            if not elapsed(0.10):                # gap
                return 0
            st['phase'] = 'TAP_Z'; st['ts'] = now
            return 0

        if phase == 'TAP_Z':
            if not elapsed(0.12):                # hold Z ~120 ms
                return 1 << self.key_bits['Z']
            st['phase'] = 'PAUSE2'; st['ts'] = now
            return 0

        if phase == 'PAUSE2':
            if not elapsed(0.10):                # gap
                return 0
            st['phase'] = 'HOLD_RETURN'; st['ts'] = now  # loop while still in window
            return 0

        # fallback
        st['phase'] = 'INIT'; st['ts'] = now
        return 0

    # ──────────────────────────────────────────────────────────────────
    # policy step
    # ──────────────────────────────────────────────────────────────────
    def decide_action(self, port, game_state):
        in_menu = bool(game_state.get("inside_window", False))

        if in_menu:
            # do NOT carry battle holds into menus
            self._held_bits[port] = 0
            # drive the menu-skip FSM
            key_bits = self._skip_menu(port)
            return {
                "button_command": {"type": "key_press", "key": self.bin16(key_bits)},
                "action_idx": None, "log_prob": None, "value": None,
                "stacked_frames_tensor": None, "game_features_tensor": None,
            }

        # leaving menu → reset FSM cleanly
        st = self._ensure_skip_shape(port)
        if st['phase'] != 'INIT':
            self._skip[port] = {'phase': 'INIT', 'ts': 0.0}

        # encode observation
        pil = None
        b64 = game_state.get("image")
        if b64:
            try:
                pil = Image.open(BytesIO(base64.b64decode(b64)))
            except Exception:
                pil = None

        self._append_obs(port, pil, game_state)
        seq_f, seq_feat = self._get_seq_tensors(port)

        # model forward (stochastic during training)
        self.model.eval()
        with torch.no_grad():
            a_idx, log_p, _, value, _ = self.model.get_action_and_value(seq_f, seq_feat)

        action_name = self.discrete_actions[a_idx.item()]
        frame_mask  = int(self.map_discrete(a_idx.item(), self.discrete_actions, self.key_bits), 2)

        # sticky holds (explicit RELEASE_* clears; taps don’t affect the latch)
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
        # NOTE: choosing "X" or "Z" (tap) does not toggle the latch

        final_mask = frame_mask | self._held_bits[port]

        return {
            "button_command": {"type": "key_press", "key": self.bin16(final_mask)},
            "action_idx": a_idx,
            "log_prob":   log_p,
            "value":      value,
            "stacked_frames_tensor": seq_f,
            "game_features_tensor":  seq_feat,
        }

    def reset_state(self, port):
        self._frames[port].clear()
        self._feats [port].clear()
        self._skip  [port] = {'phase': 'INIT', 'ts': 0.0}   # ← consistent shape
        self._held_bits[port] = 0

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
        elif code == 1 and now-ts >= 0.5:   # send RETURN, begin spamming
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

