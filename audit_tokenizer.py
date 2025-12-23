#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.shared import BUTTON_ACTION_TOKENS as NG_BUTTON_TOKENS


def _coerce_tokenizer_cfg(tok_cfg: Any) -> NitrogenTokenizerConfig:
    """
    ckpt_config.tokenizer_cfg is sometimes stored as a raw dict.
    NitrogenTokenizer expects a NitrogenTokenizerConfig instance.
    """
    if isinstance(tok_cfg, NitrogenTokenizerConfig):
        return tok_cfg
    if isinstance(tok_cfg, dict):
        return NitrogenTokenizerConfig.model_validate(tok_cfg)
    raise TypeError(f"Unsupported tokenizer_cfg type: {type(tok_cfg)}")



def _window_indices_end(idx: int, T: int, n: int) -> List[int]:
    idx = max(0, min(idx, n - 1))
    start = idx - (T - 1)
    out = []
    for t in range(T):
        j = start + t
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        out.append(j)
    return out


def _future_indices_start(idx: int, T: int, n: int) -> List[int]:
    idx = max(0, min(idx, n - 1))
    out = []
    for t in range(T):
        j = idx + t
        if j >= n:
            j = n - 1
        out.append(j)
    return out


def load_ng_checkpoint_faithful(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint not a dict: {type(ckpt)}")
    if "ckpt_config" not in ckpt or not isinstance(ckpt["ckpt_config"], dict):
        raise RuntimeError("Checkpoint missing ckpt_config dict.")
    tok_cfg = ckpt["ckpt_config"].get("tokenizer_cfg", None)
    if tok_cfg is None:
        raise RuntimeError("Checkpoint missing ckpt_config.tokenizer_cfg.")
    return {"ckpt": ckpt, "tokenizer_cfg": tok_cfg}


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True, help="path to cached .pt file")
    ap.add_argument("--ckpt", type=str, default="weights/ng.pt", help="base NitroGen ckpt (for tokenizer_cfg)")
    ap.add_argument("--anchor", type=int, default=1090, help="anchor frame index to inspect")
    ap.add_argument("--action-offset", type=int, default=0, help="future offset for action plan")
    args = ap.parse_args()

    btn_tokens = list(NG_BUTTON_TOKENS)
    print("✅ NG button token order:")
    print("   ", btn_tokens)

    ck = load_ng_checkpoint_faithful(args.ckpt)
    tok_cfg = _coerce_tokenizer_cfg(ck["tokenizer_cfg"])
    tokenizer = NitrogenTokenizer(tok_cfg)

    tokenizer.train()

    ah = int(getattr(tokenizer, "action_horizon", 18))
    vh = int(getattr(tokenizer, "vision_horizon", 1))
    print(f"\nTokenizer horizons: vision_horizon={vh} action_horizon={ah}")

    cache_path = Path(args.cache)
    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        data = torch.load(cache_path, map_location="cpu", weights_only=True)

    frames_u8 = data["frames"]   # [N,3,H,W] uint8
    actions = data["actions"]    # [N,25] float
    n = int(actions.shape[0])
    print(f"\nCache: {cache_path.name} N={n}")

    anchor = max(0, min(int(args.anchor), n - 1))
    ids_vis = _window_indices_end(anchor, vh, n)
    ids_act = _future_indices_start(anchor + int(args.action_offset), ah, n)

    act_win = actions[ids_act].float()          # [T,25] in cache-space (axes 0, buttons 0/1)
    j_left = act_win[:, 0:2].unsqueeze(0)       # [1,T,2]
    j_right = act_win[:, 2:4].unsqueeze(0)      # [1,T,2]
    buttons01 = act_win[:, 4:].unsqueeze(0)     # [1,T,21]

    # frames: [1,V,3,H,W] in [-1,1] like training
    fr = frames_u8[ids_vis].float().div(255.0).mul(2.0).sub(1.0).unsqueeze(0)

    sample = {
        "frames": fr,
        "j_left": j_left,
        "j_right": j_right,
        "buttons": buttons01,
        "dropped_frames": torch.zeros((1, vh), dtype=torch.bool),
        "game": "bn6",
    }

    enc = tokenizer.encode(sample)
    print("enc keys =", sorted(list(enc.keys())))
    print("tokenizer.training =", getattr(tokenizer, "training", None))
    print("tok_cfg.training =", getattr(tok_cfg, "training", None))


    def _as_tensor(v: Any, *, dtype: torch.dtype = torch.float32) -> Any:
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.to(dtype=dtype)
        # common: list-of-lists or numpy arrays
        try:
            t = torch.as_tensor(v)
            if t.dtype != dtype:
                t = t.to(dtype=dtype)
            return t
        except Exception:
            return v  # leave as-is for debugging prints

    def _ensure_batched_3d(t: Any, *, name: str) -> Any:
        """
        Normalize common tokenizer outputs:
        - [T,D]   -> [1,T,D]
        - [D]     -> [1,1,D]
        - [B,T,D] -> unchanged
        """
        if not torch.is_tensor(t):
            return t
        if t.ndim == 2:
            return t.unsqueeze(0)
        if t.ndim == 1:
            return t.unsqueeze(0).unsqueeze(0)
        return t

    # Pull things we care about (coerced + normalized)
    enc_actions = _ensure_batched_3d(_as_tensor(enc.get("actions", None)), name="actions")
    enc_buttons = _ensure_batched_3d(_as_tensor(enc.get("buttons", None)), name="buttons")
    enc_jl      = _ensure_batched_3d(_as_tensor(enc.get("j_left", None)),  name="j_left")
    enc_jr      = _ensure_batched_3d(_as_tensor(enc.get("j_right", None)), name="j_right")



    def show(name: str, t: torch.Tensor, k: int = 12) -> None:
        tt = t.detach().cpu()
        print(f"{name}: shape={tuple(tt.shape)} dtype={tt.dtype}")
        if tt.ndim >= 3:
            vals = tt[0, 0, :k].tolist()
            print(f"  first step first{k} =", vals)
        elif tt.ndim == 2:
            vals = tt[0, :k].tolist()
            print(f"  first row first{k} =", vals)
        else:
            vals = tt[:k].tolist()
            print(f"  first{k} =", vals)


    print("\n=== INPUT (cache-space) first step ===")
    print("j_left[0,0]  =", j_left[0, 0].tolist())
    print("j_right[0,0] =", j_right[0, 0].tolist())
    print("buttons01 pressed =", [btn_tokens[i] for i, v in enumerate(buttons01[0, 0].tolist()) if v > 0.5])

    print("\n=== TOKENIZER OUTPUT ===")
    if torch.is_tensor(enc_jl):
        show("enc.j_left", enc_jl)
    if torch.is_tensor(enc_jr):
        show("enc.j_right", enc_jr)
    if torch.is_tensor(enc_buttons):
        # unique values for buttons (should usually remain {0,1})
        uniq = torch.unique(enc_buttons.detach().cpu())
        print("enc.buttons unique =", sorted(float(x) for x in uniq.tolist()))
        print("enc.buttons pressed =", [btn_tokens[i] for i, v in enumerate(enc_buttons[0, 0].tolist()) if v > 0.5])
    else:
        print("enc.buttons missing")

    if not torch.is_tensor(enc_actions):
        print("enc['actions'] present but not a torch.Tensor")
        print("type(enc['actions']) =", type(enc.get("actions", None)))
        raise SystemExit("❌ enc['actions'] not tensor; coercion failed.")


    print("\nenc.actions first step (first 10 dims) =", enc_actions[0, 0, :10].detach().cpu().tolist())
    print("enc.actions axis part [0:4] =", enc_actions[0, 0, 0:4].detach().cpu().tolist())
    print("enc.actions buttons part max =", float(enc_actions[0, :, 4:].max().item()),
          "min =", float(enc_actions[0, :, 4:].min().item()))

    # Compare enc.actions to a few candidate constructions
    # (this tells us exactly what transform is being applied)
    act_cache = act_win.unsqueeze(0)  # [1,T,25] 0/1 buttons

    cand_raw = torch.cat([j_left, j_right, buttons01], dim=-1)  # [1,T,25]
    cand_center = torch.cat([j_left, j_right, buttons01 - 0.5], dim=-1)
    cand_pm1 = torch.cat([j_left, j_right, buttons01 * 2.0 - 1.0], dim=-1)

    print("\n=== RECONSTRUCTION ERROR (enc.actions vs candidates) ===")
    print("max|enc - raw(0/1)|      =", _max_abs_err(enc_actions, cand_raw))
    print("max|enc - centered(±0.5)|=", _max_abs_err(enc_actions, cand_center))
    print("max|enc - pm1(±1)|       =", _max_abs_err(enc_actions, cand_pm1))

    # Also show if axes are shifted by a constant
    delta = (enc_actions - cand_center).detach().cpu()
    ax_delta = delta[0, :, 0:4]
    print("\nAxis delta stats vs centered candidate:")
    print("mean delta axes =", ax_delta.mean(dim=0).tolist())
    print("max abs delta axes =", ax_delta.abs().max(dim=0).values.tolist())

    print("\n✅ Done. This tells you whether tokenizer is centering buttons (0/1 -> ±0.5), "
          "and whether it offsets/remaps any axes.")


if __name__ == "__main__":
    main()
