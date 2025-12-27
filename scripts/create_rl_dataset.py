# scripts/create_rl_dataset.py
from __future__ import annotations

import os
import json
import numpy as np

# --- CONFIG ---
INPUT_DIR = "data/dataset"
OUTPUT_DIR = "data/nitrogen_rl"

FPS = 60
LOOKAHEAD_FRAMES = 4 * FPS      # 4.0 seconds
PRE_EVENT_FRAMES = 18           # start labeling 18 frames BEFORE execution
CHARGE_THRESHOLD = 2            # Max charge level
NO_CHIP = 65535                 # observed sentinel

# Output files
WEIGHTS_OUT = "frame_weights.jsonl"
EVENTS_OUT = "rl_events.jsonl"

# -----------------------------------------------------------------------------
# Form mapping (normalized to match your UI crossName indexing)
# -----------------------------------------------------------------------------
FORM_NAMES = [
    "Normal", "Fire", "Elec", "Slash", "Erase", "Charge",
    "Aqua", "Thawk", "Tengu", "Grnd", "Dust",
]


def _normalize_game_emotion_to_cross(game_emotion: int) -> tuple[int, bool]:
    """
    Normalize `*_game_emotion` into:
      - cross_id: 0..10 (matches UI crossName indexing)
      - is_beast: best-effort boolean

    Common encodings:
      - 0..10: cross id (Normal..Dust)
      - 12..22: beast variant (base cross + 12)
      - 11: sometimes appears as "Normal beast"/special beast state -> map to Normal+Beast
    """
    ge = int(game_emotion)

    if 0 <= ge <= 10:
        return ge, False

    if 12 <= ge <= 22:
        base = ge - 12
        if 0 <= base <= 10:
            return base, True

    if ge == 11:
        return 0, True

    return 0, False


def _emotion_to_form_label(game_emotion: int, *, beast_mode: int | None = None) -> str:
    """
    Return a string label consistent with the UI naming.
    If beast_mode is available, it takes precedence over the derived beast flag.
    """
    cross_id, derived_beast = _normalize_game_emotion_to_cross(game_emotion)
    name = FORM_NAMES[cross_id] if 0 <= cross_id < len(FORM_NAMES) else "Normal"

    is_beast = bool(beast_mode) if beast_mode is not None else derived_beast
    return f"{name}{'_BEAST' if is_beast else ''}"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _combine_weight(prev: float, new: float) -> float:
    """
    Combine overlapping weights in a stable way:
      - rewards (>1): keep the max
      - penalties (<1): keep the min
      - ignore ~1.0
    """
    if abs(new - 1.0) < 1e-9:
        return prev
    if new > 1.0 and prev > 1.0:
        return max(prev, new)
    if new < 1.0 and prev < 1.0:
        return min(prev, new)
    if new > 1.0:
        return max(prev, new)
    return min(prev, new)


def _charge_release_events(charge: np.ndarray) -> np.ndarray:
    """
    True at i when prev >= threshold and curr == 0.
    """
    n = int(charge.shape[0])
    out = np.zeros(n, dtype=bool)
    if n <= 1:
        return out
    prev = charge[:-1]
    curr = charge[1:]
    out[1:] = (prev >= CHARGE_THRESHOLD) & (curr == 0)
    return out


def _chip_use_events(chip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Your telemetry semantics:
      - chip[i] is the chip "on deck" (will fire on next A)
      - NO_CHIP means nothing on deck
      - A "use" is when the on-deck chip changes away from a previous non-NO_CHIP chip.

    Returns:
      used_mask[i] True when a chip was consumed at i (i is the change frame)
      used_chip_id[i] the chip id that was consumed (prev value), else NO_CHIP
    """
    n = int(chip.shape[0])
    used = np.zeros(n, dtype=bool)
    used_id = np.full(n, NO_CHIP, dtype=np.int32)
    if n <= 1:
        return used, used_id

    prev = chip[:-1]
    curr = chip[1:]

    # "use" if previous was a real chip and current differs (including curr==NO_CHIP).
    m = (prev != NO_CHIP) & (curr != prev)

    used[1:] = m
    used_id[1:] = np.where(m, prev, NO_CHIP).astype(np.int32)
    return used, used_id


def _apply_event(
    *,
    replay_name: str,
    frame_to_weight: dict,
    events_out: list,
    actor: str,              # "player" | "enemy"
    kind: str,               # "chip" | "charge"
    label: str,              # "CHIP:123" | "CROSS:Charge" | etc
    start_frame: int,
    event_frame: int,
    end_frame: int,
    damage_dealt: int,
    damage_taken: int,
):
    """
    Decide outcome + weight, write:
      - per-frame weights for [start_frame..event_frame] (inclusive)
      - one event record in rl_events.jsonl
    """
    start_frame = int(max(0, start_frame))
    event_frame = int(max(0, event_frame))
    end_frame = int(max(event_frame, end_frame))

    if actor == "player":
        miss = (damage_dealt <= 0)
        got_hit = (damage_taken > 0)

        if miss:
            weight = 0.10
            outcome = "bad"
        elif got_hit:
            weight = 0.50
            outcome = "bad"
        else:
            if kind == "charge":
                weight = 3.00
            else:
                weight = 1.0 + (damage_dealt / 100.0)
                weight = min(weight, 5.0)
            outcome = "good"

    elif actor == "enemy":
        dodged = (damage_taken == 0)
        if dodged:
            weight = 2.00
            outcome = "good"
        else:
            weight = 0.50
            outcome = "bad"

    else:
        raise ValueError(f"Unknown actor={actor!r}")

    for fidx in range(start_frame, event_frame + 1):
        prev = float(frame_to_weight.get(fidx, 1.0))
        frame_to_weight[fidx] = float(_combine_weight(prev, weight))

    if actor == "player":
        if damage_dealt <= 0:
            reason = "miss"
        elif damage_taken > 0:
            reason = "trade"
        else:
            reason = "clean"
    else:
        reason = "dodged" if damage_taken == 0 else "hit"

    events_out.append({
        "replay": replay_name,
        "actor": actor,
        "kind": kind,
        "label": label,
        "outcome": outcome,
        "reason": reason,
        "start_frame": start_frame,
        "event_frame": event_frame,
        "end_frame": end_frame,
        "weight": round(float(weight), 3),
        "damage_dealt": int(damage_dealt),
        "damage_taken": int(damage_taken),
    })


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------
def process_replay(replay_path: str):
    frames = []
    try:
        with open(replay_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frames.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return [], {}

    if not frames:
        return [], {}

    total = len(frames)

    # HP / Charge (fallback-safe)
    e_hp = np.array([_safe_int(frames[i].get("enemy_health", 1000), 1000) for i in range(total)], dtype=np.int32)
    p_hp = np.array([_safe_int(frames[i].get("player_health", 1000), 1000) for i in range(total)], dtype=np.int32)
    p_charge = np.array([_safe_int(frames[i].get("player_charge", 0), 0) for i in range(total)], dtype=np.int32)
    e_charge = np.array([_safe_int(frames[i].get("enemy_charge", 0), 0) for i in range(total)], dtype=np.int32)

    # On-deck chip (65535 when none)
    p_chip = np.array([_safe_int(frames[i].get("player_chip", NO_CHIP), NO_CHIP) for i in range(total)], dtype=np.int32)
    e_chip = np.array([_safe_int(frames[i].get("enemy_chip", NO_CHIP), NO_CHIP) for i in range(total)], dtype=np.int32)

    # Current cross/form (game_emotion encoding normalized later)
    p_form = np.array([_safe_int(frames[i].get("player_game_emotion", 0), 0) for i in range(total)], dtype=np.int32)
    e_form = np.array([_safe_int(frames[i].get("enemy_game_emotion", 0), 0) for i in range(total)], dtype=np.int32)

    # Beast overlay (use if present; 0 if missing)
    p_beast = np.array([_safe_int(frames[i].get("beast_mode", 0), 0) for i in range(total)], dtype=np.int32)
    e_beast = np.array([_safe_int(frames[i].get("beast_mode", 0), 0) for i in range(total)], dtype=np.int32)

    # Damage deltas from HP
    e_dmg = np.maximum(0, e_hp[:-1] - e_hp[1:])
    p_dmg = np.maximum(0, p_hp[:-1] - p_hp[1:])
    e_dmg = np.append(e_dmg, 0).astype(np.int32)
    p_dmg = np.append(p_dmg, 0).astype(np.int32)

    # Events
    player_chip_use, player_chip_used_id = _chip_use_events(p_chip)
    enemy_chip_use, enemy_chip_used_id = _chip_use_events(e_chip)

    player_charge_release = _charge_release_events(p_charge)
    enemy_charge_release  = _charge_release_events(e_charge)

    # Build per-frame weight map (sparse updates)
    frame_to_weight = {}
    rl_events = []

    replay_name = os.path.basename(os.path.dirname(replay_path))

    for i in range(1, total):  # start at 1 so edge logic is safe
        start = max(0, i - PRE_EVENT_FRAMES)
        end = min(total, i + LOOKAHEAD_FRAMES)

        # --- Player chip use (chip consumed at i; label uses prev chip id) ---
        if player_chip_use[i]:
            chip_id = int(player_chip_used_id[i])
            if chip_id != NO_CHIP:
                label = f"CHIP:{chip_id}"
                dealt = int(np.sum(e_dmg[i:end]))
                taken = int(np.sum(p_dmg[i:end]))
                _apply_event(
                    replay_name=replay_name,
                    frame_to_weight=frame_to_weight,
                    events_out=rl_events,
                    actor="player",
                    kind="chip",
                    label=label,
                    start_frame=start,
                    event_frame=i,
                    end_frame=end - 1,
                    damage_dealt=dealt,
                    damage_taken=taken,
                )

        # --- Player charge shot release (label by normalized form + beast overlay) ---
        if player_charge_release[i]:
            form_label = _emotion_to_form_label(int(p_form[i]), beast_mode=int(p_beast[i]))
            label = f"CROSS:{form_label}"
            dealt = int(np.sum(e_dmg[i:end]))
            taken = int(np.sum(p_dmg[i:end]))
            _apply_event(
                replay_name=replay_name,
                frame_to_weight=frame_to_weight,
                events_out=rl_events,
                actor="player",
                kind="charge",
                label=label,
                start_frame=start,
                event_frame=i,
                end_frame=end - 1,
                damage_dealt=dealt,
                damage_taken=taken,
            )

        # --- Enemy chip use (enemy chip consumed; from OUR POV: reward if we dodged) ---
        if enemy_chip_use[i]:
            chip_id = int(enemy_chip_used_id[i])
            if chip_id != NO_CHIP:
                label = f"CHIP:{chip_id}"
                dealt = int(np.sum(e_dmg[i:end]))
                taken = int(np.sum(p_dmg[i:end]))
                _apply_event(
                    replay_name=replay_name,
                    frame_to_weight=frame_to_weight,
                    events_out=rl_events,
                    actor="enemy",
                    kind="chip",
                    label=label,
                    start_frame=start,
                    event_frame=i,
                    end_frame=end - 1,
                    damage_dealt=dealt,
                    damage_taken=taken,
                )

        # --- Enemy charge shot release (label by normalized enemy form + beast overlay) ---
        if enemy_charge_release[i]:
            form_label = _emotion_to_form_label(int(e_form[i]), beast_mode=int(e_beast[i]))
            label = f"CROSS:{form_label}"
            dealt = int(np.sum(e_dmg[i:end]))
            taken = int(np.sum(p_dmg[i:end]))
            _apply_event(
                replay_name=replay_name,
                frame_to_weight=frame_to_weight,
                events_out=rl_events,
                actor="enemy",
                kind="charge",
                label=label,
                start_frame=start,
                event_frame=i,
                end_frame=end - 1,
                damage_dealt=dealt,
                damage_taken=taken,
            )

    # Convert sparse weights to jsonl records
    weights_records = []
    for fidx in sorted(frame_to_weight.keys()):
        w = float(frame_to_weight[fidx])
        if abs(w - 1.0) < 1e-9:
            continue
        weights_records.append({
            "key": f"{replay_name}/{int(fidx)}",
            "val": round(w, 3),
        })

    return rl_events, weights_records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_events = []
    all_weights = []

    from tqdm import tqdm

    print(f"Scanning {INPUT_DIR}...")

    action_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file == "actions.jsonl":
                action_files.append(os.path.join(root, file))

    print(f"Found {len(action_files)} replays with actions.jsonl")

    # Optional counters for sanity
    c = {
        ("player", "chip", "good"): 0,
        ("player", "chip", "bad"): 0,
        ("player", "charge", "good"): 0,
        ("player", "charge", "bad"): 0,
        ("enemy", "chip", "good"): 0,
        ("enemy", "chip", "bad"): 0,
        ("enemy", "charge", "good"): 0,
        ("enemy", "charge", "bad"): 0,
    }

    for path in tqdm(action_files, desc="Processing replays", unit="replay"):
        evts, wrecs = process_replay(path)
        all_events.extend(evts)
        all_weights.extend(wrecs)

        for e in evts:
            key = (e["actor"], e["kind"], e["outcome"])
            if key in c:
                c[key] += 1

    print("Event summary:")
    for k in sorted(c.keys()):
        print(f"  {k}: {c[k]}")

    # Save weights (training)
    weights_path = os.path.join(OUTPUT_DIR, WEIGHTS_OUT)
    print(f"Writing {len(all_weights)} weight records to {weights_path}...")
    with open(weights_path, "w") as f:
        for rec in all_weights:
            f.write(json.dumps(rec) + "\n")

    # Save events (inspection)
    events_path = os.path.join(OUTPUT_DIR, EVENTS_OUT)
    print(f"Writing {len(all_events)} event records to {events_path}...")
    with open(events_path, "w") as f:
        for evt in all_events:
            f.write(json.dumps(evt) + "\n")

    print("✅ Done!")


if __name__ == "__main__":
    main()
