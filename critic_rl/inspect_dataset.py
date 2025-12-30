# scripts/inspect_dataset.py
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch

from critic_rl.dataset import (
    ACTION_DIM,
    BUTTON_KEYS,
    InMemoryCriticRLTDDataset,
)

# ---------------------------------------------------------------------------
# Expectations (for verification / human-readable reporting)
# ---------------------------------------------------------------------------

# IMPORTANT UPDATE:
# - "scalars" is now STATE-ONLY and is 16-dim.
# - Buttons moved to a separate "action" vector (len(BUTTON_KEYS) == ACTION_DIM).
SCALAR_MAP: Dict[int, Tuple[str, float]] = {
    0: ("Player HP", 1000.0),
    1: ("Enemy HP", 1000.0),
    2: ("Player Charge", 2.0),
    3: ("Enemy Charge", 2.0),
    4: ("Cust Gauge", 100.0),
    5: ("Inside Window", 1.0),
    6: ("Turn Index", 50.0),
    7: ("Player X", 750.0),
    8: ("Player Y", 750.0),
    9: ("Enemy X", 750.0),
    10: ("Enemy Y", 750.0),
    11: ("Window Happened", 1.0),
    12: ("Window Selected", 1.0),
    13: ("Window Beast", 1.0),
    14: ("Player Cross Ent", 1.0),
    15: ("Enemy Cross Ent", 1.0),
}

ACTION_MAP: Dict[int, str] = {i: k for i, k in enumerate(BUTTON_KEYS)}

BEAST_FEAT_MAP = {
    0: "P Active",
    1: "P Turns Since",
    2: "P Ever",
    3: "E Active",
    4: "E Turns Since",
    5: "E Ever",
}


def print_section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f" {title}")
    print(f"{'=' * 78}")


def _select_valid(t: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Select valid timesteps.
    - If t is [N,T] -> returns [M]
    - If t is [N,T,D...] -> returns [M, D...]
    """
    if t.ndim < 2:
        raise ValueError(f"Expected tensor with >=2 dims [N,T,...], got {tuple(t.shape)}")
    if valid.shape != t.shape[:2]:
        raise ValueError(f"valid shape {tuple(valid.shape)} does not match prefix of {tuple(t.shape)}")

    if t.ndim == 2:
        return t[valid]
    return t[valid]


def analyze_tensor(name: str, t: torch.Tensor, valid: torch.Tensor, *, is_discrete: bool = False) -> None:
    data = _select_valid(t, valid)

    if data.numel() == 0:
        print(f"{name:<28} | [EMPTY]")
        return

    d_type = str(data.dtype).replace("torch.", "")
    shape_str = f"{list(t.shape[2:])}" if t.ndim > 2 else "[]"

    # Bool mean/std crashes unless cast
    if data.dtype == torch.bool:
        data_f = data.float()
        vmin = data_f.min().item()
        vmax = data_f.max().item()
        vmean = data_f.mean().item()
        vstd = data_f.std().item()
        print(
            f"{name:<28} | bool       | Tail:{shape_str:<12} | "
            f"Min:{vmin:<7.2f} Max:{vmax:<7.2f} Mean:{vmean:<7.2f} Std:{vstd:<7.2f}"
        )
        return

    if is_discrete or data.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        uniq = torch.unique(data)
        num_uniq = int(uniq.numel())
        vmin = int(data.min().item())
        vmax = int(data.max().item())
        sample = uniq[:8].tolist()
        sample_str = str(sample) + ("..." if num_uniq > 8 else "")
        print(
            f"{name:<28} | {d_type:<9} | Tail:{shape_str:<12} | "
            f"Range:[{vmin},{vmax}] | Unique:{num_uniq:<6} | Samples:{sample_str}"
        )
        return

    status = "OK"
    if torch.isnan(data).any():
        status = "!! NAN !!"
    elif torch.isinf(data).any():
        status = "!! INF !!"

    vmin = data.min().item()
    vmax = data.max().item()
    vmean = data.mean().item()
    vstd = data.std().item()
    print(
        f"{name:<28} | {d_type:<9} | Tail:{shape_str:<12} | "
        f"Min:{vmin:<9.4f} Max:{vmax:<9.4f} Mean:{vmean:<9.4f} Std:{vstd:<9.4f} | {status}"
    )


def _print_action_breakdown(ds: InMemoryCriticRLTDDataset, valid: torch.Tensor) -> None:
    if "action" not in ds.x:
        print("!! MISSING: action (buttons vector) - Rebuild Cache !!")
        return

    a = ds.x["action"]  # [N,T,A]
    if a.ndim != 3 or a.shape[2] != ACTION_DIM:
        print(f"!! action shape unexpected: {tuple(a.shape)} expected [N,T,{ACTION_DIM}]")
        return

    flat_a = _select_valid(a, valid)  # [M,A]
    print(f"Action dim: {ACTION_DIM}  (BUTTON_KEYS={len(BUTTON_KEYS)})")
    print(f"{'Idx':<4} {'Button':<18} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'NonZero%':<10}")

    for i in range(ACTION_DIM):
        col = flat_a[:, i].float()
        vmin = col.min().item()
        vmax = col.max().item()
        vmean = col.mean().item()
        vstd = col.std().item()
        nonzero = (col.abs() > 1e-6).float().mean().item() * 100.0
        nm = ACTION_MAP.get(i, f"BTN[{i}]")
        print(f"{i:<4} {nm:<18} {vmin:<10.4f} {vmax:<10.4f} {vmean:<10.4f} {vstd:<10.4f} {nonzero:<10.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--folder_len", type=int, default=30)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--require_cust_gt0", action="store_true", default=True)
    ap.add_argument("--val_ratio", type=float, default=0.0)
    args = ap.parse_args()

    print(f"--- Loading InMemory cache: {args.cache_dir} | split={args.split} ---")

    try:
        ds = InMemoryCriticRLTDDataset(
            args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            split=str(args.split),
            val_ratio=float(args.val_ratio),
        )
    except Exception as e:
        print(f"FATAL: Could not load dataset. {e}")
        return

    valid = ds.valid  # [N,T] bool

    print_section("1) REWARDS (targets)")
    r = ds.r  # [N,T] float32
    r_flat = _select_valid(r, valid).float()

    abs_mean = r_flat.abs().mean().item()
    zero_frac = (r_flat.abs() <= 1e-9).float().mean().item() * 100.0
    qs = torch.quantile(r_flat, torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], device=r_flat.device)).tolist()
    print(f"AbsMean: {abs_mean:.6f} | sparsity(==0): {zero_frac:.2f}%")
    print(f"Quantiles: q00={qs[0]:.3f} q01={qs[1]:.3f} q10={qs[2]:.3f} q50={qs[3]:.3f} q90={qs[4]:.3f} q99={qs[5]:.3f} q100={qs[6]:.3f}")

    d_flat = _select_valid(ds.done, valid).float()
    print(f"Done rate over VALID steps: {d_flat.mean().item() * 100.0:.2f}%")


    raw_rmse = (r_flat**2).mean().sqrt()
    sym_r = torch.sign(r_flat) * torch.log1p(torch.abs(r_flat))
    sym_rmse = (sym_r**2).mean().sqrt()

    print(f"Count: {int(r_flat.numel()):,}")
    print(f"Min:   {r_flat.min().item():.3f}")
    print(f"Max:   {r_flat.max().item():.3f}")
    print(f"Mean:  {r_flat.mean().item():.6f}")
    print("-" * 56)
    print(f"Zero-Pred RMSE (Linear): {raw_rmse.item():.6f}")
    print(f"Zero-Pred RMSE (SymLog): {sym_rmse.item():.6f}  (sanity baseline)")

    print_section("2) DISCRETE INPUTS (IDs / indices)")

    analyze_tensor("grid_tile", ds.x["grid_tile"], valid, is_discrete=True)
    analyze_tensor("grid_owner", ds.x["grid_owner"], valid, is_discrete=True)

    analyze_tensor("hand_id", ds.x["hand_id"], valid, is_discrete=True)
    analyze_tensor("held_id", ds.x["held_id"], valid, is_discrete=True)
    analyze_tensor("folder_id_p", ds.x["folder_id_p"], valid, is_discrete=True)
    analyze_tensor("folder_id_e", ds.x["folder_id_e"], valid, is_discrete=True)

    # Option A/C spatial indices
    if "p_grid_idx" in ds.x:
        analyze_tensor("p_grid_idx", ds.x["p_grid_idx"], valid, is_discrete=True)
        analyze_tensor("e_grid_idx", ds.x["e_grid_idx"], valid, is_discrete=True)
        analyze_tensor("rel_pe_idx", ds.x["rel_pe_idx"], valid, is_discrete=True)
    else:
        print("!! MISSING: p_grid_idx/e_grid_idx/rel_pe_idx (Option A/C) - Rebuild Cache !!")

    # Option B board feature
    if "board_feat" in ds.x:
        analyze_tensor("board_feat", ds.x["board_feat"], valid, is_discrete=False)
    else:
        print("!! MISSING: board_feat (Option B) - Rebuild Cache !!")

    # On-deck / last-used
    if "current_chip_p" in ds.x:
        analyze_tensor("current_chip_p", ds.x["current_chip_p"], valid, is_discrete=True)
        analyze_tensor("current_chip_e", ds.x["current_chip_e"], valid, is_discrete=True)
    else:
        print("!! MISSING: current_chip_p/current_chip_e (On Deck) - Rebuild Cache !!")

    if "last_used_id_p" in ds.x:
        analyze_tensor("last_used_id_p", ds.x["last_used_id_p"], valid, is_discrete=True)
        analyze_tensor("last_used_id_e", ds.x["last_used_id_e"], valid, is_discrete=True)
    else:
        print("!! MISSING: last_used_id_p/last_used_id_e (Just Fired) - Rebuild Cache !!")

    analyze_tensor("hand_code", ds.x["hand_code"], valid, is_discrete=True)
    analyze_tensor("held_code", ds.x["held_code"], valid, is_discrete=True)

    analyze_tensor("active_cross_idx_p", ds.x["active_cross_idx_p"], valid, is_discrete=True)
    analyze_tensor("active_cross_idx_e", ds.x["active_cross_idx_e"], valid, is_discrete=True)

    print_section("3) CONTINUOUS / MASK INPUTS")

    analyze_tensor("hand_vis", ds.x["hand_vis"], valid)
    analyze_tensor("held_mask", ds.x["held_mask"], valid)  # bool
    analyze_tensor("folder_used_p", ds.x["folder_used_p"], valid)
    analyze_tensor("folder_used_e", ds.x["folder_used_e"], valid)
    analyze_tensor("folder_mask_p", ds.x["folder_mask_p"], valid)  # bool
    analyze_tensor("folder_mask_e", ds.x["folder_mask_e"], valid)  # bool
    analyze_tensor("used_cross", ds.x["used_cross"], valid)

    if "grid_p_occ" in ds.x:
        analyze_tensor("grid_p_occ", ds.x["grid_p_occ"], valid)
        analyze_tensor("grid_e_occ", ds.x["grid_e_occ"], valid)
    else:
        print("!! MISSING: grid_p_occ/grid_e_occ (Option A occupancy) - Rebuild Cache !!")

    print("\n--- Beast Features (6-dim) Breakdown ---")
    bf = ds.x["beast_feats"]  # [N,T,6]
    flat_bf = _select_valid(bf, valid)  # [M,6]
    for i in range(6):
        nm = BEAST_FEAT_MAP.get(i, f"Feat {i}")
        col = flat_bf[:, i]
        print(
            f"  {i}: {nm:<15} | Min:{col.min().item():.3f} "
            f"Max:{col.max().item():.3f} Mean:{col.mean().item():.3f}"
        )

    print_section("4) STATE SCALARS (16-dim, STATE ONLY)")
    if "scalars" not in ds.x:
        print("!! MISSING: scalars - Rebuild Cache !!")
    else:
        scalars = ds.x["scalars"]  # [N,T,16]
        if scalars.ndim != 3 or scalars.shape[2] != 16:
            print(f"!! scalars shape unexpected: {tuple(scalars.shape)} expected [N,T,16]")
        flat_sc = _select_valid(scalars, valid)  # [M,16]

        print(f"{'Idx':<4} {'Name':<22} {'NormFactor':<11} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'Status'}")
        print("-" * 102)

        for i in range(int(flat_sc.shape[1])):
            name, factor = SCALAR_MAP.get(i, ("(Unknown)", 1.0))
            vals = flat_sc[:, i]
            vmin = vals.min().item()
            vmax = vals.max().item()
            vmean = vals.mean().item()
            vstd = vals.std().item()

            status = "OK"
            if torch.isnan(vals).any():
                status = "!! NAN !!"
            elif torch.isinf(vals).any():
                status = "!! INF !!"
            elif vmax > 2.0 and "(Unknown)" not in name:
                status = "(!High)"
            elif vstd == 0.0:
                status = "(Constant)"

            print(f"{i:<4} {name:<22} {factor:<11.1f} {vmin:<10.4f} {vmax:<10.4f} {vmean:<10.4f} {vstd:<10.4f} {status}")

    print_section("5) ACTION VECTOR (buttons, separate from scalars)")
    _print_action_breakdown(ds, valid)

    print("\nDone.")


if __name__ == "__main__":
    main()
