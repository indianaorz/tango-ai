# train.py
# -----------------------------------------------------------------------------
# Tango NitroGen training (cached dataset) — faithful loader + correct collation
#
# Fixes:
# - Faithful checkpoint loading (validated ckpt_config, no mutation)
# - Collation tensorizes tokenizer outputs (fixes "actions is list" crash)
# - Frames normalized to [-1,1] once BEFORE tokenizer.encode() (as you requested)
# - bf16 autocast, no GradScaler
# -----------------------------------------------------------------------------

from __future__ import annotations

import bisect
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nitrogen.mm_tokenizers import NitrogenTokenizer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": int(os.getenv("BATCH_SIZE", "64")),
    "lr": float(os.getenv("LR", "1e-4")),
    "epochs": int(os.getenv("EPOCHS", "10")),
    "save_every": int(os.getenv("SAVE_EVERY", "5000")),
    "dataset_dir": os.getenv("DATASET_DIR", "data/dataset_cached"),
    "log_dir": os.getenv("LOG_DIR", "logs/tango_cached"),
    "checkpoints_dir": os.getenv("CKPT_DIR", "checkpoints"),
    "resume_path": os.getenv("RESUME_PATH", ""),  # optional
    "num_workers": int(os.getenv("NUM_WORKERS", "0")),
    "pin_memory": bool(int(os.getenv("PIN_MEMORY", "1"))),
    "tune_vision_tower": bool(int(os.getenv("TUNE_VISION_TOWER", "0"))),  # False => freeze params by name
    "max_keep_ckpts": int(os.getenv("MAX_KEEP_CKPTS", "3")),
    "seed": int(os.getenv("SEED", "0")),  # 0 => no fixed seed
}

NG_CKPT_PATH = os.getenv("NG_CKPT_PATH", "weights/ng.pt")


# -----------------------------------------------------------------------------
# Helpers
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


def _tensor_stats(t: torch.Tensor) -> str:
    tt = t.detach().float().cpu()
    return (
        f"min={tt.min().item():.4f} max={tt.max().item():.4f} "
        f"mean={tt.mean().item():.4f} std={tt.std(unbiased=False).item():.4f}"
    )


def _cleanup_old_checkpoints(ckpt_dir: Path, max_keep: int) -> None:
    if max_keep <= 0:
        return
    ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1,
    )
    if len(ckpts) <= max_keep:
        return
    for old in ckpts[:-max_keep]:
        try:
            print(f"🗑️ Deleting old checkpoint: {old}")
            old.unlink()
        except Exception as e:
            print(f"⚠️ Failed to delete {old}: {e}")


def _window_indices(idx: int, T: int, n: int) -> List[int]:
    idx = max(0, min(idx, n - 1))
    start = idx - (T - 1)
    out: List[int] = []
    for t in range(T):
        j = start + t
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        out.append(j)
    return out


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


def _is_numeric_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)


def _tensorize_value(v: Any, *, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    """
    Best-effort conversion:
      - Tensor => tensor
      - np.ndarray => tensor
      - list/tuple of numbers => tensor
      - list/tuple of list/tuple of numbers => tensor
    Returns None if not convertible.
    """
    if torch.is_tensor(v):
        return v
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return None
        return torch.from_numpy(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return torch.tensor([], dtype=dtype)
        # list of tensors
        if all(torch.is_tensor(x) for x in v):
            try:
                return torch.cat(list(v), dim=0)
            except Exception:
                try:
                    return torch.stack(list(v), dim=0)
                except Exception:
                    return None
        # list of numeric scalars
        if all(_is_numeric_scalar(x) for x in v):
            return torch.tensor(v, dtype=dtype)
        # list of list of numeric scalars
        if all(isinstance(x, (list, tuple, np.ndarray)) for x in v):
            try:
                vv = []
                for x in v:
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    if not isinstance(x, (list, tuple)):
                        return None
                    if not all(_is_numeric_scalar(y) for y in x):
                        return None
                    vv.append(list(x))
                return torch.tensor(vv, dtype=dtype)
            except Exception:
                return None
    return None


# -----------------------------------------------------------------------------
# Faithful loader (no mutation, validated config, strict key checks)
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
    if device is None:
        raise NgPolicyError("device is None.")

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

    # tokenizer cfg (validated if schema is present)
    tokenizer_cfg = raw_config.get("tokenizer_cfg", None)
    tok_mod = _try_import("nitrogen.mm_tokenizers")
    NitrogenTokenizerConfig = _get_attr(tok_mod, "NitrogenTokenizerConfig")
    if tokenizer_cfg is not None and NitrogenTokenizerConfig is not None:
        try:
            tokenizer_cfg = NitrogenTokenizerConfig.model_validate(tokenizer_cfg)
        except Exception:
            # if validation fails, keep raw dict so we can still construct tokenizer
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

    # Fail fast by default: you WANT to know if you’re not loading the same model.
    if unexpected:
        raise NgPolicyError(f"Unexpected keys in state_dict (first 30): {unexpected[:30]}")
    if missing:
        raise NgPolicyError(f"Missing keys in state_dict (first 30): {missing[:30]}")

    model.to(device)
    return NgLoaded(model=model, ckpt_config=ckpt_config, tokenizer_cfg=tokenizer_cfg, device=device)


# -----------------------------------------------------------------------------
# Cached Dataset
# -----------------------------------------------------------------------------

class CachedTangoDataset(Dataset):
    """
    Cached *.pt files:
      frames:  uint8 [N,3,H,W]
      actions: float [N,25]
    Returns:
      frames:  float32 [-1,1] [3,H,W]   (vision_horizon=1, last frame only)
      j_left:  float32 [T,2]
      j_right: float32 [T,2]
      buttons: float32 [T,21]
      dropped_frames: bool [1]
      game: str
    """

    def __init__(self, root_dir: str, *, action_horizon: int):
        self.root = Path(root_dir)
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.root}. Did you run precache_dataset.py?")

        if action_horizon <= 0:
            raise ValueError("action_horizon must be > 0")
        self.action_horizon = int(action_horizon)

        self.cumulative_sizes: List[int] = []
        self.file_lengths: List[int] = []
        total_frames = 0

        print(f"🔍 Scanning cached dataset at {self.root}...")
        for pt_file in tqdm(self.files, desc="Indexing"):
            try:
                data = torch.load(pt_file, weights_only=True, map_location="cpu")
                n = int(data["frames"].shape[0])
                total_frames += n
                self.cumulative_sizes.append(total_frames)
                self.file_lengths.append(n)
            except Exception as e:
                print(f"⚠️ Error reading {pt_file}: {e}")

        if not self.cumulative_sizes or self.cumulative_sizes[-1] <= 0:
            raise RuntimeError("Indexed dataset but found no frames.")

        print(f"✅ Indexed {len(self.files)} files, {total_frames:,} total frames.")

        self.cache_idx: int = -1
        self.cache_data: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def _load_file(self, file_idx: int) -> None:
        try:
            self.cache_data = torch.load(self.files[file_idx], weights_only=True, map_location="cpu", mmap=True)
        except TypeError:
            self.cache_data = torch.load(self.files[file_idx], weights_only=True, map_location="cpu")
        self.cache_idx = file_idx

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        file_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        if file_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_sizes[file_idx - 1]

        if self.cache_idx != file_idx or self.cache_data is None:
            self._load_file(file_idx)

        assert self.cache_data is not None
        actions_all = self.cache_data["actions"]  # [N,25]
        frames_all = self.cache_data["frames"]    # [N,3,H,W] uint8
        n = int(frames_all.shape[0])

        ids = _window_indices(int(local_idx), self.action_horizon, n)

        # action window [T,25]
        act_win = actions_all[ids].float()
        j_left = act_win[:, 0:2]     # [T,2]
        j_right = act_win[:, 2:4]    # [T,2]
        buttons = act_win[:, 4:]     # [T,21]

        # vision horizon = 1 (last frame in window)
        vis_idx = ids[-1]
        frame_u8 = frames_all[vis_idx]          # [3,H,W] uint8
        frame = frame_u8.float().div(255.0)     # [0,1]
        frame = frame.mul(2.0).sub(1.0)         # [-1,1]

        return {
            "frames": frame,
            "j_left": j_left,
            "j_right": j_right,
            "buttons": buttons,
            "dropped_frames": torch.zeros(1, dtype=torch.bool),
            "game": "bn6",
        }


# -----------------------------------------------------------------------------
# Collation: tokenize per-sample, then collate encoded dict.
# Critical: tensorize numeric lists/arrays (fixes "actions is list").
# -----------------------------------------------------------------------------

def _collate_encoded(encoded_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = encoded_list[0].keys()

    for k in keys:
        vals = [e[k] for e in encoded_list]
        v0 = vals[0]

        # Fast path: tensors
        if torch.is_tensor(v0):
            try:
                batched = torch.cat(vals, dim=0)  # common: each item already has batch dim 1
            except Exception:
                batched = torch.stack(vals, dim=0)
            out[k] = batched.to(device=device, non_blocking=True)
            continue

        # Numeric/list/ndarray path: tensorize if possible
        t0 = _tensorize_value(v0)
        if t0 is not None:
            t_vals: List[torch.Tensor] = []
            ok = True
            for v in vals:
                tv = _tensorize_value(v)
                if tv is None:
                    ok = False
                    break
                t_vals.append(tv)

            if ok:
                # Heuristic:
                # - if each tv is shaped like [1, ...], cat on dim 0
                # - else stack
                try:
                    if t_vals[0].ndim >= 1 and t_vals[0].shape[0] == 1:
                        batched = torch.cat(t_vals, dim=0)
                    else:
                        batched = torch.stack(t_vals, dim=0)
                except Exception:
                    batched = torch.stack(t_vals, dim=0)

                # IMPORTANT: NitroGen forward expects actions as float tensor
                if k == "actions":
                    batched = batched.to(dtype=torch.float32)

                out[k] = batched.to(device=device, non_blocking=True)
                continue

        # Fallback: keep python objects (strings, etc.)
        out[k] = vals

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    _set_seed(CONFIG["seed"])

    device = torch.device(CONFIG["device"])
    ckpt_dir = Path(CONFIG["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"🧠 Loading base checkpoint (faithful): {NG_CKPT_PATH}")
    loaded = load_ng_checkpoint_faithful(NG_CKPT_PATH, device=device)

    if loaded.tokenizer_cfg is None:
        raise RuntimeError("Base checkpoint missing tokenizer_cfg; cannot train safely.")

    tokenizer = NitrogenTokenizer(loaded.tokenizer_cfg)
    tokenizer.train()

    model = loaded.model
    model.train()

    # Optional: freeze vision tower params (config flags may still print True; params control training)
    if not CONFIG["tune_vision_tower"]:
        frozen = 0
        total = 0
        for name, p in model.named_parameters():
            total += 1
            if ("vision" in name) or ("vision_tower" in name) or ("siglip" in name):
                p.requires_grad = False
                frozen += 1
        print(f"🔒 Froze {frozen}/{total} params by name match for vision tower.")

    action_horizon = int(getattr(tokenizer, "action_horizon", 18))
    print(f"action_horizon={action_horizon}")

    writer = SummaryWriter(CONFIG["log_dir"])

    dataset = CachedTangoDataset(CONFIG["dataset_dir"], action_horizon=action_horizon)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=(CONFIG["num_workers"] > 0),
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG["lr"])

    # Resume (optional)
    global_step = 0
    resume_path = CONFIG["resume_path"].strip()
    if resume_path and Path(resume_path).exists():
        print(f"🔄 Resuming from {resume_path}...")
        ck = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        global_step = int(ck.get("step", 0))
        print(f"✅ Resumed at step {global_step}")

    did_probe = False

    print(f"🔥 Starting Training on {len(dataset)} frames...")
    for epoch in range(CONFIG["epochs"]):
        print(f"--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        pbar = tqdm(loader)

        for batch in pbar:
            encoded_samples: List[Dict[str, Any]] = []
            bs = int(batch["frames"].shape[0])

            for i in range(bs):
                frames_btchw = batch["frames"][i].unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
                j_left = batch["j_left"][i].unsqueeze(0)                     # [1,T,2]
                j_right = batch["j_right"][i].unsqueeze(0)                   # [1,T,2]
                buttons = batch["buttons"][i].unsqueeze(0)                   # [1,T,21]
                dropped = batch["dropped_frames"][i].unsqueeze(0)            # [1,1]
                game = batch["game"][i] if isinstance(batch["game"], list) else batch["game"]

                sample = {
                    "frames": frames_btchw,
                    "j_left": j_left,
                    "j_right": j_right,
                    "buttons": buttons,
                    "dropped_frames": dropped,
                    "game": game,
                }

                try:
                    enc = tokenizer.encode(sample)
                except ValueError as e:
                    print(f"⚠️ Tokenizer error: {e}")
                    enc = None

                if enc is not None:
                    encoded_samples.append(enc)

            if not encoded_samples:
                continue

            model_input = _collate_encoded(encoded_samples, device=device)

            # Sanity: ensure actions exists and is tensor (prevents your crash)
            if "actions" in model_input and not torch.is_tensor(model_input["actions"]):
                raise RuntimeError(f"model_input['actions'] is not a tensor, got {type(model_input['actions'])}")
            if "actions" not in model_input:
                # Some tokenizer layouts might name it differently; surface it immediately.
                raise RuntimeError(f"Tokenizer output missing 'actions' key. Keys: {sorted(list(model_input.keys()))[:64]}")

            if not did_probe:
                did_probe = True
                x_in = batch["frames"][0]
                print(f"\n[probe] dataset frames range: {_tensor_stats(x_in)}")
                pf = model_input.get("frames", None)
                pv = model_input.get("pixel_values", None)
                if torch.is_tensor(pv):
                    print(f"[probe] tokenizer pixel_values range: {_tensor_stats(pv)}")
                if torch.is_tensor(pf):
                    print(f"[probe] tokenizer frames range: {_tensor_stats(pf)}")
                a = model_input["actions"]
                print(f"[probe] tokenizer actions: shape={tuple(a.shape)} dtype={a.dtype} device={a.device}")
                print()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(model_input)
                loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            global_step += 1
            loss_val = float(loss.detach().item())
            pbar.set_description(f"Loss: {loss_val:.4f}")
            writer.add_scalar("Training/Loss", loss_val, global_step)

            if global_step % CONFIG["save_every"] == 0:
                save_path = ckpt_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "ckpt_config": _to_dict(loaded.ckpt_config),
                    },
                    save_path,
                )
                print(f"💾 Saved checkpoint: {save_path}")
                _cleanup_old_checkpoints(ckpt_dir, CONFIG["max_keep_ckpts"])

    print("✅ Training Complete.")
    writer.close()


if __name__ == "__main__":
    main()
