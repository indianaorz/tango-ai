# convert_dataset.py
from __future__ import annotations

import json
import argparse
from collections import OrderedDict
from typing import Dict, Tuple

from action_schema import (
    GBA_BITS,
    GBA_TO_NITROGEN,
    NITROGEN_TEMPLATE,
)

# Fields that represent the "Player" and "Enemy" relative to the capture
SWAP_PAIRS = [
    ("player_health", "enemy_health"),
    ("player_pos", "enemy_pos"),
    ("player_charge", "enemy_charge"),
    ("player_chip", "enemy_chip"),
]

def parse_input_bitmask(bitmask_int: int) -> OrderedDict:
    """
    Produces a Nitrogen-shaped action row:
      - axes are present (list-wrapped), left at 0.0
      - only GBA buttons are ever set to 1.0
      - all other Nitrogen buttons remain 0.0
    """
    action = NITROGEN_TEMPLATE.copy()

    if not isinstance(bitmask_int, int):
        bitmask_int = 0

    for bit, gba_btn in GBA_BITS.items():
        if (bitmask_int >> bit) & 1:
            nitro_key = GBA_TO_NITROGEN.get(gba_btn)
            if nitro_key is not None:
                action[nitro_key] = 1.0

    return action

def extract_state_v1(obj: dict, swap: bool) -> dict:
    state_wrapper = obj.get("state", {})
    st = state_wrapper.get("V1", state_wrapper) if isinstance(state_wrapper, dict) else {}
    if not st:
        return {}

    if swap:
        st = st.copy()
        for k1, k2 in SWAP_PAIRS:
            st[k1], st[k2] = st.get(k2), st.get(k1)

    return st

def _update_press_stats(stats: Dict[str, int], row: dict) -> None:
    # Count presses for only keys that exist in the row template.
    for k, v in row.items():
        if k in NITROGEN_TEMPLATE and isinstance(v, (int, float)) and float(v) > 0.5:
            stats[k] = stats.get(k, 0) + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--swap-players", action="store_true", help="Swap P1/P2 telemetry data")
    parser.add_argument("--print-stats", action="store_true", help="Print button press-rate sanity stats")
    args = parser.parse_args()

    count = 0
    prev_hp = {"p": None, "e": None}
    press_stats: Dict[str, int] = {}

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # 1) Inputs
                joyflags = data.get("input", 0)
                if isinstance(joyflags, dict):
                    joyflags = joyflags.get("local", 0)

                final_row = parse_input_bitmask(int(joyflags or 0))

                # 2) Metadata
                final_row["frame_idx"] = data.get("frame", count)
                final_row["tick"] = data.get("tick", count)

                # 3) State Extraction & Swap
                st = extract_state_v1(data, args.swap_players)
                for k, v in st.items():
                    final_row[k] = v

                # 4) Metrics
                p_hp = final_row.get("player_health")
                e_hp = final_row.get("enemy_health")
                curr_p = p_hp if isinstance(p_hp, (int, float)) else prev_hp["p"]
                curr_e = e_hp if isinstance(e_hp, (int, float)) else prev_hp["e"]

                if curr_p is not None and prev_hp["p"] is not None:
                    diff = prev_hp["p"] - curr_p
                    final_row["player_damage_taken"] = diff if 0 < diff < 1000 else 0
                else:
                    final_row["player_damage_taken"] = 0

                if curr_e is not None and prev_hp["e"] is not None:
                    diff = prev_hp["e"] - curr_e
                    final_row["enemy_damage_taken"] = diff if 0 < diff < 1000 else 0
                else:
                    final_row["enemy_damage_taken"] = 0

                prev_hp["p"] = curr_p
                prev_hp["e"] = curr_e

                if args.print_stats:
                    _update_press_stats(press_stats, final_row)

                fout.write(json.dumps(final_row) + "\n")
                count += 1

            except Exception:
                # Intentionally skip bad lines to keep conversion resilient.
                continue

    print(f"Converted {count} frames.")

    if args.print_stats and count > 0:
        # Basic “is something stuck?” sanity
        keys = [k for k in NITROGEN_TEMPLATE.keys() if isinstance(NITROGEN_TEMPLATE[k], (int, float))]
        pairs = []
        for k in keys:
            c = press_stats.get(k, 0)
            pairs.append((k, c / float(count)))
        pairs.sort(key=lambda kv: kv[1], reverse=True)

        print("\n=== Button press-rate (top 12) ===")
        for k, r in pairs[:12]:
            print(f"{k:16s} {r*100:6.2f}%")

        # Heuristic warnings for obvious bitmask mismatch
        for k, r in pairs:
            if r > 0.95:
                print(f"⚠️ WARNING: '{k}' is pressed {r*100:.1f}% of frames. Bitmask mapping may be wrong or stuck input.")

if __name__ == "__main__":
    main()
