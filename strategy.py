import time, base64
from collections import defaultdict, deque
from io import BytesIO
import torch
from PIL import Image

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
                 cross_select_default: int = 0):
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
        f_t = self.preprocess_frame(pil_img, self.H, self.W)  # [C,H,W], float 0-1
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
        now = time.time()

        def elapsed(sec): return (now - st['ts']) >= sec
        if st['phase'] == 'INIT':
            if st['ts'] == 0.0:
                st['ts'] = now
                return 0
            if elapsed(0.30):
                st['phase'] = 'HOLD_RETURN'; st['ts'] = now
            return 0

        if st['phase'] == 'HOLD_RETURN':
            if not elapsed(0.22):
                return 1 << self.key_bits['RETURN']
            st['phase'] = 'PAUSE1'; st['ts'] = now
            return 0

        if st['phase'] == 'PAUSE1':
            if not elapsed(0.10):
                return 0
            st['phase'] = 'TAP_Z'; st['ts'] = now
            return 0

        if st['phase'] == 'TAP_Z':
            if not elapsed(0.12):
                return 1 << self.key_bits['Z']
            st['phase'] = 'PAUSE2'; st['ts'] = now
            return 0

        if st['phase'] == 'PAUSE2':
            if not elapsed(0.10):
                return 0
            st['phase'] = 'HOLD_RETURN'; st['ts'] = now
            return 0

        st['phase'] = 'INIT'; st['ts'] = now
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

