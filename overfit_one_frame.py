# overfit_one_frame.py
# -----------------------------------------------------------------------------
# Overfit a single cached frame/window to validate end-to-end alignment.
#
# Target:
#   file = data/dataset_cached/<FILENAME>.pt
#   anchor idx = 3206  (vision window ends here)
#   plan = actions[idx + ACTION_OFFSET : idx + ACTION_OFFSET + action_horizon]
#
# This uses the SAME:
#   - model checkpoint loading (faithful)
#   - NitrogenTokenizer.encode(per-sample)
#   - cache->(j_left/j_right/buttons) split
#   - frame normalization to [-1,1]
#
# Usage:
#   python overfit_one_frame.py \
#     --pt data/dataset_cached/20230929001213-ummm-bn6-vs-DthKrdMnSP-round1-p1.pt \
#     --idx 3206 \
#     --ng weights/ng.pt \
#     --outdir checkpoints_overfit \
#     --steps 5000 \
#     --lr 1e-4
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from nitrogen.mm_tokenizers import NitrogenTokenizer
from nitrogen.shared import BUTTON_ACTION_TOKENS as NG_BUTTON_TOKENS

from action_schema import BUTTON_TOKENS, ACTION_DIM, GBA_UI_BUTTONS
import numpy as np
import torch
from typing import Any, Dict

# Keys that must have a leading batch dim for NitroGen.forward
_BATCH_2D = {"actions", "actions_mask"}                # [T,25] -> [1,T,25]
_BATCH_1D = {"sa_token_ids", "vl_token_ids", "vl_attn_mask"}  # [L] -> [1,L]
_BATCH_SCALARS = {"embodiment_id", "game_ids", "has_detection_target", "has_real_action"}  # [] -> [1]

def _to_torch_and_device(enc: Dict[str, Any], *, device: torch.device) -> Dict[str, Any]:
    """
    Convert tokenizer.encode output to torch tensors on device.
    Keeps non-tensors (e.g. 'game' string) as-is.
    """
    out: Dict[str, Any] = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            t = v
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        else:
            out[k] = v
            continue

        # Normalize old uint8 masks -> bool
        if t.dtype == torch.uint8:
            t = t.bool()

        out[k] = t.to(device=device, non_blocking=True)

    return out


def _ensure_batched(model_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make NitroGen.forward happy by ensuring a leading batch dimension exists
    for the fields that are commonly returned unbatched by tokenizer.encode().
    """
    out = dict(model_input)

    # [T,25] -> [1,T,25]
    for k in _BATCH_2D:
        t = out.get(k, None)
        if torch.is_tensor(t) and t.ndim == 2:
            out[k] = t.unsqueeze(0)

    # [L] -> [1,L]
    for k in _BATCH_1D:
        t = out.get(k, None)
        if torch.is_tensor(t) and t.ndim == 1:
            out[k] = t.unsqueeze(0)

    # [] -> [1]
    for k in _BATCH_SCALARS:
        t = out.get(k, None)
        if torch.is_tensor(t) and t.ndim == 0:
            out[k] = t.unsqueeze(0)

    # (optional) if your tokenizer ever returns dropped_* as [V] instead of [1,V]
    for k in ("dropped_frames", "dropped_images"):
        t = out.get(k, None)
        if torch.is_tensor(t) and t.ndim == 1:
            out[k] = t.unsqueeze(0)

    return out

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

class NgPolicyError(RuntimeError):
    pass


def _try_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception:
        return None


def _get_attr(mod: Any, name: str) -> Any:
    return getattr(mod, name, None) if mod is not None else None


def _first_non_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _set_seed(seed: int) -> None:
    if seed <= 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _window_indices_end(idx: int, T: int, n: int) -> List[int]:
    idx = max(0, min(int(idx), int(n) - 1))
    start = idx - (T - 1)
    out: List[int] = []
    for t in range(T):
        j = start + t
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        out.append(int(j))
    return out


def _future_indices_start(idx: int, T: int, n: int) -> List[int]:
    idx = max(0, min(int(idx), int(n) - 1))
    out: List[int] = []
    for t in range(T):
        j = idx + t
        if j >= n:
            j = n - 1
        out.append(int(j))
    return out


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _extract_loss(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    if hasattr(outputs, "loss"):
        loss = getattr(outputs, "loss")
        if torch.is_tensor(loss):
            return loss
    if isinstance(outputs, (tuple, list)) and outputs:
        if torch.is_tensor(outputs[0]):
            return outputs[0]
        if isinstance(outputs[0], dict) and "loss" in outputs[0]:
            return outputs[0]["loss"]
    raise RuntimeError(f"Could not extract loss from outputs type={type(outputs)}")


def _to_dict(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {"repr": repr(x)}


# -----------------------------------------------------------------------------
# Faithful checkpoint load (same as train.py)
# -----------------------------------------------------------------------------

class NgLoaded:
    def __init__(self, *, model: nn.Module, ckpt_config: Any, tokenizer_cfg: Any, device: torch.device):
        self.model = model
        self.ckpt_config = ckpt_config
        self.tokenizer_cfg = tokenizer_cfg
        self.device = device


def load_ng_checkpoint_faithful(ckpt_path: str, device: torch.device) -> NgLoaded:
    if not ckpt_path:
        raise NgPolicyError("ckpt_path is empty.")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise NgPolicyError(f"Checkpoint is not a dict. Got: {type(ckpt)}")

    raw_config = ckpt.get("ckpt_config", None)
    if not isinstance(raw_config, dict):
        raise NgPolicyError("Checkpoint missing ckpt_config dict; cannot load faithfully.")

    cfg_mod = _first_non_none(
        _try_import("nitrogen.cfg"),
        _try_import("nitrogen.config"),
        _try_import("config"),
    )
    CkptConfig = _get_attr(cfg_mod, "CkptConfig")
    if CkptConfig is None:
        raise NgPolicyError("Could not import CkptConfig; cannot validate checkpoint config faithfully.")

    ckpt_config = CkptConfig.model_validate(raw_config)
    model_cfg = getattr(ckpt_config, "model_cfg", None)
    if model_cfg is None:
        raise NgPolicyError("Validated ckpt_config has no model_cfg.")

    tokenizer_cfg = raw_config.get("tokenizer_cfg", None)
    tok_mod = _try_import("nitrogen.mm_tokenizers")
    NitrogenTokenizerConfig = _get_attr(tok_mod, "NitrogenTokenizerConfig")
    if tokenizer_cfg is not None and NitrogenTokenizerConfig is not None:
        try:
            tokenizer_cfg = NitrogenTokenizerConfig.model_validate(tokenizer_cfg)
        except Exception:
            pass

    mod = _try_import("nitrogen.flow_matching_transformer.nitrogen")
    NitroGen = _get_attr(mod, "NitroGen")
    if NitroGen is None:
        raise NgPolicyError("Could not import NitroGen.")

    try:
        model = NitroGen(config=model_cfg)
    except TypeError:
        model = NitroGen(config=model_cfg, game_mapping=None)

    state = ckpt.get("model", None)
    if state is None:
        raise NgPolicyError("Checkpoint missing 'model' state_dict key.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise NgPolicyError(f"Unexpected keys in state_dict (first 30): {unexpected[:30]}")
    if missing:
        raise NgPolicyError(f"Missing keys in state_dict (first 30): {missing[:30]}")

    model.to(device)
    return NgLoaded(model=model, ckpt_config=ckpt_config, tokenizer_cfg=tokenizer_cfg, device=device)


# -----------------------------------------------------------------------------
# Build the single sample from cached .pt
# -----------------------------------------------------------------------------

def build_single_sample(
    *,
    pt_path: Path,
    anchor_idx: int,
    vision_horizon: int,
    action_horizon: int,
    action_offset: int,
) -> Dict[str, Any]:
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    frames_u8: torch.Tensor = data["frames"]      # [N,3,H,W] uint8
    actions: torch.Tensor = data["actions"].float()  # [N,ACTION_DIM]

    n = int(frames_u8.shape[0])
    if n <= 0:
        raise RuntimeError("Empty cached pt file.")

    if actions.ndim != 2 or int(actions.shape[1]) != int(ACTION_DIM):
        raise RuntimeError(f"actions expected [N,{ACTION_DIM}], got {tuple(actions.shape)}")

    V = int(vision_horizon)
    T = int(action_horizon)

    ids_vis = _window_indices_end(anchor_idx, V, n)

    start_act = int(anchor_idx) + int(action_offset)
    ids_act = _future_indices_start(start_act, T, n)

    act_win = actions[ids_act]  # [T,ACTION_DIM]
    j_left = act_win[:, 0:2]    # [T,2] in [-1,1]
    j_right = act_win[:, 2:4]   # [T,2] in [-1,1]
    buttons = act_win[:, 4:]    # [T,21] in {0,1}

    fr = frames_u8[ids_vis].float().div(255.0).mul(2.0).sub(1.0)  # [-1,1] [V,3,H,W]

    return {
        "frames": fr,                           # [V,3,H,W]
        "j_left": j_left,                       # [T,2]
        "j_right": j_right,                     # [T,2]
        "buttons": buttons,                     # [T,21]
        "dropped_frames": torch.zeros(V, dtype=torch.bool),
        "game": "bn6",
        "__meta": {
            "pt": str(pt_path),
            "n": n,
            "anchor": int(anchor_idx),
            "ids_vis": ids_vis,
            "ids_act": ids_act,
            "action_offset": int(action_offset),
        },
    }


# -----------------------------------------------------------------------------
# Metrics (uses tokenizer-space model_input["actions"] to avoid layout guessing)
# -----------------------------------------------------------------------------

def compute_metrics_from_tokenizer_actions(
    model_input: Dict[str, Any],
    *,
    press_threshold: float,
) -> Dict[str, float]:
    """
    model_input["actions"] is what the tokenizer produced for GT:
      shape [B,T,25]
      layout: buttons first (len BUTTON_TOKENS), then j_left(2), j_right(2)
    We only compute button metrics here.
    """
    actions = model_input.get("actions", None)
    if not torch.is_tensor(actions):
        return {}

    # [B,T,25] -> [T,21] (B is 1)
    btn = actions[0, :, : len(BUTTON_TOKENS)].float()

    gt = (btn > press_threshold).to(torch.bool)  # [T,21]
    # Also useful: what fraction of buttons are pressed at all
    gt_press_rate = float(gt.float().mean().item())

    # For “confidence gap”, we can look at raw values too
    # (Tokenizer typically stores buttons as floats 0..1 or logits depending on variant,
    # but in your current runs it’s 0..1. We guard anyway.)
    raw = btn
    if raw.min().item() < -0.05 or raw.max().item() > 1.05:
        prob = torch.sigmoid(raw)
    else:
        prob = raw.clamp(0.0, 1.0)

    pos = prob[gt]
    neg = prob[~gt]
    pos_mean = float(pos.mean().item()) if pos.numel() else 0.0
    neg_mean = float(neg.mean().item()) if neg.numel() else 0.0

    return {
        "gt_press_rate": gt_press_rate,
        "gt_pos_prob_mean": pos_mean,
        "gt_neg_prob_mean": neg_mean,
    }

def _batch_to_torch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Convert tokenizer.encode output to torch tensors where applicable.
    NitroGen.forward expects tensors for actions/actions_mask/images/token ids/etc.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=True)
            continue

        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            # preserve bool masks
            if t.dtype == torch.uint8:
                # older codepaths sometimes emit uint8 masks; treat as bool
                t = t.bool()
            out[k] = t.to(device=device, non_blocking=True)
            continue

        # Some tokenizer fields are plain python lists (e.g. game names); keep them.
        out[k] = v

    return out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=== Overfit One Frame ===")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True)
    ap.add_argument("--idx", type=int, default=3206)
    ap.add_argument("--ng", type=str, default="weights/ng.pt")
    ap.add_argument("--outdir", type=str, default="checkpoints_overfit")
    ap.add_argument("--logdir", type=str, default="logs/overfit_one_frame")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--press_th", type=float, default=0.5)
    ap.add_argument("--action_offset", type=int, default=int(os.getenv("ACTION_OFFSET", "0")))
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--freeze_vision", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(args.seed)

    # Canonical token order check
    if list(NG_BUTTON_TOKENS) != list(BUTTON_TOKENS):
        raise RuntimeError("BUTTON_TOKENS mismatch vs nitrogen.shared.BUTTON_ACTION_TOKENS")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    # Load checkpoint + tokenizer
    loaded = load_ng_checkpoint_faithful(args.ng, device=device)
    if loaded.tokenizer_cfg is None:
        raise RuntimeError("Base checkpoint missing tokenizer_cfg")

    tokenizer = NitrogenTokenizer(loaded.tokenizer_cfg)
    tokenizer.train()

    model = loaded.model
    model.train()

    if args.freeze_vision:
        frozen = 0
        total = 0
        for name, p in model.named_parameters():
            total += 1
            if ("vision" in name) or ("vision_tower" in name) or ("siglip" in name):
                p.requires_grad = False
                frozen += 1
        print(f"🔒 Froze {frozen}/{total} params by name match for vision tower.")

    action_horizon = int(getattr(tokenizer, "action_horizon", 18))
    vision_horizon = int(getattr(tokenizer, "vision_horizon", 1))
    old_layout = getattr(tokenizer, "old_layout", None)

    print("\n=== TOKENIZER ===")
    print(f"vision_horizon={vision_horizon} action_horizon={action_horizon} old_layout={old_layout}")
    print(f"ACTION_OFFSET={args.action_offset}")
    print("==============\n")

    # Build the single sample
    pt_path = Path(args.pt)
    sample = build_single_sample(
        pt_path=pt_path,
        anchor_idx=args.idx,
        vision_horizon=vision_horizon,
        action_horizon=action_horizon,
        action_offset=args.action_offset,
    )
    print("Sample meta:", sample["__meta"])

    # Encode ONCE (targets fixed)
    enc = tokenizer.encode(
        {
            "frames": sample["frames"].unsqueeze(0),                 # [1,V,3,H,W]
            "j_left": sample["j_left"].unsqueeze(0),                 # [1,T,2]
            "j_right": sample["j_right"].unsqueeze(0),               # [1,T,2]
            "buttons": sample["buttons"].unsqueeze(0),               # [1,T,21]
            "dropped_frames": sample["dropped_frames"].unsqueeze(0), # [1,V]
            "game": sample["game"],
        }
    )

    model_input = _to_torch_and_device(enc, device=device)
    model_input = _ensure_batched(model_input)


    # Quick sanity (prints once)
    print("=== MODEL_INPUT TYPES ===")
    for k in sorted(model_input.keys()):
        v = model_input[k]
        if torch.is_tensor(v):
            print(f"{k:24s} Tensor {tuple(v.shape)} {v.dtype} {v.device}")
        else:
            print(f"{k:24s} {type(v)}")
    print("=========================\n")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    gba_keys = list(GBA_UI_BUTTONS)
    print("Overfitting target buttons:", gba_keys)

    use_amp = (device.type == "cuda")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model(model_input)  # <-- NO kwargs fallback
                loss = _extract_loss(outputs)
        else:
            outputs = model(model_input)      # <-- NO kwargs fallback
            loss = _extract_loss(outputs)

        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        writer.add_scalar("overfit/loss", loss_val, step)

        with torch.no_grad():
            m = compute_metrics_from_tokenizer_actions(model_input, press_threshold=args.press_th)
            for kk, vv in m.items():
                writer.add_scalar(f"overfit/{kk}", float(vv), step)

        if step % 25 == 0 or step == 1:
            print(
                f"step {step:5d} | loss={loss_val:.6f} | "
                f"gt_press_rate={m.get('gt_press_rate', 0.0):.3f} | "
                f"pos_mean={m.get('gt_pos_prob_mean', 0.0):.3f} | "
                f"neg_mean={m.get('gt_neg_prob_mean', 0.0):.3f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = outdir / f"overfit_{pt_path.stem}_idx{args.idx}_step{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "ckpt_config": _to_dict(loaded.ckpt_config),
                    "tokenizer_cfg": _to_dict(loaded.tokenizer_cfg),
                    "meta": {
                        "pt": str(pt_path),
                        "idx": int(args.idx),
                        "action_offset": int(args.action_offset),
                        "vision_horizon": int(vision_horizon),
                        "action_horizon": int(action_horizon),
                    },
                },
                ckpt_path,
            )
            print(f"💾 saved {ckpt_path}")

    writer.close()
    print("✅ done")
if __name__ == "__main__":
    main()
