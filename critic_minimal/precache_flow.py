# critic_minimal/precache_flow.py
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from critic_minimal.features import (
    BUTTON_KEYS,
    aggregate_action,
    bellman_hp_return_sampled,
    extract_flow_features,
    _as_int,
    _as_float,
)

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def _safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            frames.append(json.loads(line))
    return frames


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def process_replay(args: Tuple[str, str, float, float, int]) -> int:
    folder, out_dir, gamma, scale, pred_len = args
    folder_p = Path(folder)
    save_path = Path(out_dir) / f"{folder_p.name}.pt"

    try:
        frames = _safe_read_jsonl(folder_p / "actions.jsonl")
    except Exception:
        return 0

    n = len(frames)
    # Need enough room to form (t .. t+pred_len) windows; plus a little slack.
    if n < (pred_len + 10):
        return 0

    # 1) Battle mask (cust_gauge > 0 with a small pre-roll)
    cust_values = [_as_int(f.get("cust_gauge"), 0) for f in frames]
    battle_mask = [False] * n
    for i in range(n):
        if cust_values[i] > 0:
            battle_mask[i] = True
            if i > 0 and cust_values[i - 1] == 0:
                for j in range(max(0, i - 4), i):
                    battle_mask[j] = True

    if not any(battle_mask):
        return 0

    # 2) Bellman returns (conditioning signal; not the sampler weights)
    indices = list(range(n))
    returns = bellman_hp_return_sampled(
        frames,
        indices,
        battle_mask,
        gamma=gamma,
        reward_scale=scale,
        positive_only=False,
        ema_alpha=0.0,
    )

    # 3) Extract features + raw per-frame action vectors
    feats_list: List[List[float]] = []
    actions_list: List[torch.Tensor] = []
    for i in range(n):
        feats_list.append(extract_flow_features(frames[i]))
        actions_list.append(aggregate_action(frames, i, hold=1))

    state_tensor = torch.tensor(feats_list, dtype=torch.float16)              # [N, feat]
    action_tensor = torch.stack(actions_list).to(torch.float16)              # [N, act]
    return_tensor = torch.tensor(returns, dtype=torch.float16)               # [N]
    mask_tensor = torch.tensor(battle_mask, dtype=torch.bool)                # [N]

    # 4) Build event-weighted sampler weights
    acts_f = action_tensor.float()  # [N, 10]

    # 4a) Sliding action variance over next pred_len frames
    # unfold length = N - pred_len + 1 (windows fully inside)
    if n >= pred_len:
        # [N - pred_len + 1, pred_len, 10]
        unfolded = acts_f.unfold(0, pred_len, 1)
        local_std = unfolded.std(dim=1).mean(dim=1)  # [N - pred_len + 1]
        # pad to [N] by adding (pred_len - 1) zeros to the tail
        if pred_len > 1:
            act_variance = torch.cat([local_std, torch.zeros(pred_len - 1, dtype=torch.float32)])
        else:
            act_variance = local_std
        # act_variance length should be N
        if act_variance.numel() != n:
            # Very defensive; should never happen.
            act_variance = torch.zeros(n, dtype=torch.float32)
    else:
        act_variance = torch.zeros(n, dtype=torch.float32)

    # 4b) Immediate damage magnitude (per-frame, NOT discounted)
    p_hp = torch.zeros(n, dtype=torch.float32)
    e_hp = torch.zeros(n, dtype=torch.float32)
    lp, le = 0.0, 0.0
    for i in range(n):
        lp = _as_float(frames[i].get("player_health"), lp)
        le = _as_float(frames[i].get("enemy_health"), le)
        p_hp[i] = float(lp)
        e_hp[i] = float(le)

    # damage deltas live on edges i -> i+1, so length (n-1)
    dp = (p_hp[:-1] - p_hp[1:]).clamp(min=0.0)  # damage taken at i
    de = (e_hp[:-1] - e_hp[1:]).clamp(min=0.0)  # damage dealt at i

    step_event = torch.zeros(n, dtype=torch.float32)
    step_event[:-1] = de + dp

    # 4c) Future dealt damage within next pred_len frames (horizon payoff)
    # Align to frame i in [0..n-1]. Use de (length n-1) windows.
    future_dealt = torch.zeros(n, dtype=torch.float32)
    if de.numel() >= pred_len:
        # de_unf length = (n-1) - pred_len + 1 = n - pred_len
        # shape: [n - pred_len, pred_len]
        de_unf = de.unfold(0, pred_len, 1)
        # fill indices [0 .. n - pred_len - 1]
        future_dealt[: (n - pred_len)] = de_unf.max(dim=1).values

    # 4d) Penalize "only SOUTH" unless it actually cashes out soon
    try:
        idx_south = BUTTON_KEYS.index("SOUTH")
    except ValueError:
        idx_south = 9
    only_south = (acts_f[:, idx_south] > 0.5) & (acts_f.sum(dim=1) <= 1.0)

    # Weight formula (bounded)
    # - step_event: immediate “something happened”
    # - future_dealt: pushes toward setups that cash out within horizon
    # - act_variance: encourages movement / input diversity
    w = torch.ones(n, dtype=torch.float32)
    w = w + 0.35 * torch.tanh(step_event / 40.0)
    w = w + 0.65 * torch.tanh(future_dealt / 40.0)
    w = w + 1.25 * act_variance

    # Suppress pure SOUTH loops unless damage appears within horizon
    w = torch.where(only_south & (future_dealt < 1.0), w * 0.15, w)

    # Apply battle mask
    w[~mask_tensor] = 0.0

    # Clip to avoid sampler collapse
    w = w.clamp(min=0.0, max=6.0)

    payload = {
        "states": state_tensor,
        "actions": action_tensor,
        "returns": return_tensor,
        "mask": mask_tensor,
        "weights": w.to(torch.float16),
        "meta": {
            "pred_len": int(pred_len),
            "gamma": float(gamma),
            "reward_scale": float(scale),
        },
    }

    try:
        torch.save(payload, save_path)
    except Exception:
        return 0

    return n


def _worker(args: Tuple[str, str, float, float, int]) -> int:
    return process_replay(args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/dataset")
    ap.add_argument("--out_dir", default="data/cache_flow_v5_eventweighted")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pred_len", type=int, default=18, help="future action horizon (frames) stored/weighted")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    pred_len = int(args.pred_len)
    if pred_len <= 0:
        raise SystemExit("--pred_len must be > 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    replays = [d for d in Path(args.dataset_dir).iterdir() if d.is_dir()]
    jobs: List[Tuple[str, str, float, float, int]] = [
        (str(r), str(out_dir), float(args.gamma), float(args.scale), pred_len) for r in replays
    ]

    print(f"Precaching {len(jobs)} replays (event-weighted)... pred_len={pred_len}")

    with mp.Pool(args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs)):
            pass

    print("Precache Complete.")


if __name__ == "__main__":
    main()
