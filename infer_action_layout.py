#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import torch

from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.shared import BUTTON_ACTION_TOKENS as NG_BUTTON_TOKENS


# -----------------------------------------------------------------------------
# Tokenizer loading
# -----------------------------------------------------------------------------

def _coerce_tokenizer_cfg(tok_cfg: Any) -> NitrogenTokenizerConfig:
    if isinstance(tok_cfg, NitrogenTokenizerConfig):
        return tok_cfg
    if isinstance(tok_cfg, dict):
        return NitrogenTokenizerConfig.model_validate(tok_cfg)
    raise TypeError(f"Unsupported tokenizer_cfg type: {type(tok_cfg)}")


def load_tokenizer_from_ckpt(ckpt_path: str) -> NitrogenTokenizer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint not a dict: {type(ckpt)}")

    ckpt_cfg = ckpt.get("ckpt_config")
    if not isinstance(ckpt_cfg, dict):
        raise RuntimeError("Checkpoint missing ckpt_config dict.")

    tok_cfg_raw = ckpt_cfg.get("tokenizer_cfg")
    if tok_cfg_raw is None:
        raise RuntimeError("Checkpoint missing ckpt_config.tokenizer_cfg.")

    tok_cfg = _coerce_tokenizer_cfg(tok_cfg_raw)
    tok = NitrogenTokenizer(tok_cfg)
    tok.train()  # ensure training-time encode path (this is where actions are produced)
    return tok


# -----------------------------------------------------------------------------
# Tensor helpers (tokenizer may return numpy arrays)
# -----------------------------------------------------------------------------

def _as_torch(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    try:
        import numpy as np  # noqa: F401
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return torch.tensor(x)
    raise TypeError(f"Unsupported tensor-like type: {type(x)}")


def _ensure_batched_3d(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize actions-like tensors into [B,T,D].
    Common cases:
      - [T,D] -> [1,T,D]
      - [D]   -> [1,1,D]
    """
    if t.ndim == 3:
        return t
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 1:
        return t.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unexpected ndim={t.ndim} for actions tensor")


def _get_actions_first_step(enc: Dict[str, Any]) -> torch.Tensor:
    a = enc.get("actions", None)
    if a is None:
        raise RuntimeError("Tokenizer output missing 'actions' key.")
    a = _as_torch(a).to(dtype=torch.float32)
    a = _ensure_batched_3d(a)
    return a[0, 0]  # [25]


# -----------------------------------------------------------------------------
# Sample builder
# -----------------------------------------------------------------------------

def _make_base_sample(*, vh: int, ah: int, H: int = 256, W: int = 256) -> Dict[str, Any]:
    frames = torch.zeros((1, vh, 3, H, W), dtype=torch.float32)  # [-1,1] safe
    j_left = torch.zeros((1, ah, 2), dtype=torch.float32)        # expected input is [-1,1]
    j_right = torch.zeros((1, ah, 2), dtype=torch.float32)       # expected input is [-1,1]
    buttons = torch.zeros((1, ah, 21), dtype=torch.float32)      # 0/1
    dropped = torch.zeros((1, vh), dtype=torch.bool)

    return {
        "frames": frames,
        "j_left": j_left,
        "j_right": j_right,
        "buttons": buttons,
        "dropped_frames": dropped,
        "game": "bn6",
    }


# -----------------------------------------------------------------------------
# Layout inference
# -----------------------------------------------------------------------------

def _layout_slices_from_tokenizer(tok: NitrogenTokenizer):
    """
    Your pasted tokenizer packs as:
      action = concat([buttons(21), j_left(2), j_right(2)])  when old_layout=False (default)

    It also supports an "old_layout" mode used by unpack_actions:
      if old_layout=True, it expects [j_left(2), j_right(2), buttons(21)]

    We derive slices here so probes never assume an axis/buttons split that isn't true.
    """
    old_layout = bool(getattr(tok, "old_layout", False))
    if old_layout:
        # [j_left(2), j_right(2), buttons(21)]
        btn_slice = slice(4, 25)
        jl_slice = slice(0, 2)
        jr_slice = slice(2, 4)
        layout_name = "[j_left, j_right, buttons]"
    else:
        # [buttons(21), j_left(2), j_right(2)]
        btn_slice = slice(0, 21)
        jl_slice = slice(21, 23)
        jr_slice = slice(23, 25)
        layout_name = "[buttons, j_left, j_right]"
    return old_layout, btn_slice, jl_slice, jr_slice, layout_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="weights/ng.pt (for tokenizer_cfg)")
    ap.add_argument("--vh", type=int, default=None, help="override vision_horizon (else tokenizer value)")
    ap.add_argument("--ah", type=int, default=None, help="override action_horizon (else tokenizer value)")
    ap.add_argument("--probe-step", type=int, default=0, help="which action-horizon step to probe (default 0)")
    args = ap.parse_args()

    ng_btn = list(NG_BUTTON_TOKENS)
    if len(ng_btn) != 21:
        raise RuntimeError(f"Expected 21 NG buttons, got {len(ng_btn)}")

    tok = load_tokenizer_from_ckpt(args.ckpt)

    vh = int(args.vh) if args.vh is not None else int(getattr(tok, "vision_horizon", 1))
    ah = int(args.ah) if args.ah is not None else int(getattr(tok, "action_horizon", 18))
    step = int(args.probe_step)
    if step < 0 or step >= ah:
        raise ValueError(f"--probe-step must be in [0, {ah-1}], got {step}")

    old_layout, btn_slice, jl_slice, jr_slice, layout_name = _layout_slices_from_tokenizer(tok)

    print("✅ NG_BUTTON_TOKENS (canonical input order):")
    print("   ", ng_btn)
    print(f"\nTokenizer horizons: vision_horizon={vh} action_horizon={ah}")
    print(f"Tokenizer old_layout={old_layout} => enc.actions layout {layout_name}")
    print(f"Probing at horizon step t={step}")

    # Baseline
    base = _make_base_sample(vh=vh, ah=ah)
    enc0 = tok.encode(base)

    # enc['actions'] is [T,25] or [1,T,25]; normalize and pick step
    a0_full = _as_torch(enc0["actions"]).to(dtype=torch.float32)
    a0_full = _ensure_batched_3d(a0_full)  # [1,T,25]
    a0 = a0_full[0, step]                  # [25]

    jl0 = a0[jl_slice].detach().cpu().tolist()
    jr0 = a0[jr_slice].detach().cpu().tolist()
    btn0 = a0[btn_slice].detach().cpu()

    print("\n=== BASELINE enc.actions (zero input) ===")
    print("j_left  =", jl0)
    print("j_right =", jr0)
    print("buttons slice stats: min/max/mean =",
          float(btn0.min().item()), float(btn0.max().item()), float(btn0.mean().item()))

    # Probe mapping
    mapping_in_to_out: Dict[int, int] = {}
    collisions: Dict[int, List[int]] = {}

    for i, name in enumerate(ng_btn):
        s = _make_base_sample(vh=vh, ah=ah)
        s["buttons"][0, step, i] = 1.0  # press button i at probed step only
        enc = tok.encode(s)

        a_full = _as_torch(enc["actions"]).to(dtype=torch.float32)
        a_full = _ensure_batched_3d(a_full)
        a = a_full[0, step]  # [25]

        btn_part = a[btn_slice].detach().cpu()  # correct 21 dims
        out_j = int(torch.argmax(btn_part).item())
        out_val = float(btn_part[out_j].item())

        mapping_in_to_out[i] = out_j
        collisions.setdefault(out_j, []).append(i)

        # If encode is producing one-hot-ish buttons, this should be ~1.0.
        # If it isn't, warn (but still record mapping).
        if out_val < 0.9:
            print(f"⚠️ weak mapping? input={name} -> out_slot={out_j} val={out_val:.3f}")

    # Invert mapping -> derived output order
    out_order: List[Optional[str]] = [None] * 21
    for in_i, out_j in mapping_in_to_out.items():
        if 0 <= out_j < 21:
            out_order[out_j] = ng_btn[in_i]

    print("\n=== DERIVED enc.actions BUTTON ORDER (output slot -> input name) ===")
    for j, nm in enumerate(out_order):
        print(f"{j:2d}: {nm}")

    missing = [j for j, nm in enumerate(out_order) if nm is None]
    if missing:
        print("\n❌ Could not fill all output slots (None at):", missing)
    else:
        print("\n✅ All 21 output slots mapped.")

    multi = {j: ins for j, ins in collisions.items() if len(ins) > 1}
    if multi:
        print("\n❌ Collisions detected (multiple inputs map to same output slot):")
        for j, ins in sorted(multi.items()):
            print(f"  out_slot {j}: {[ng_btn[i] for i in ins]}")
    else:
        print("\n✅ No collisions; mapping is 1-to-1.")

    print("\nNEXT STEP:")
    print("- In viewer/play: do NOT assume '[axes(4), buttons(21)]'.")
    print("- Use tokenizer.decode()/unpack_actions (or these derived slices) to interpret model outputs.")
    print("- Keep cache-side 'buttons' in NG_BUTTON_TOKENS order; tokenizer will pack consistently.")


if __name__ == "__main__":
    main()
