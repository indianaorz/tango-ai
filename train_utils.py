# train_utils.py
from __future__ import annotations

import bisect
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

from action_schema import BUTTON_TOKENS, ACTION_DIM

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class NgPolicyError(RuntimeError):
    pass

def _try_import(path: str):
    try: return importlib.import_module(path)
    except: return None

def _get_attr(mod: Any, name: str) -> Any:
    return getattr(mod, name, None) if mod is not None else None

def _first_non_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None: return v
    return None

def set_seed(seed: int) -> None:
    if seed <= 0: return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tensor_stats(t: torch.Tensor) -> str:
    tt = t.detach().float().cpu()
    return f"min={tt.min().item():.4f} max={tt.max().item():.4f} mean={tt.mean().item():.4f} std={tt.std(unbiased=False).item():.4f}"

def cleanup_old_checkpoints(ckpt_dir: Path, max_keep: int) -> None:
    if max_keep <= 0: return
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
    if len(ckpts) <= max_keep: return
    for old in ckpts[:-max_keep]:
        try: old.unlink()
        except Exception as e: print(f"⚠️ Failed to delete {old}: {e}")

def window_indices_end(idx: int, T: int, n: int) -> List[int]:
    idx = max(0, min(idx, n - 1))
    start = idx - (T - 1)
    return [max(0, min(start + t, n - 1)) for t in range(T)]

def to_dict(x: Any) -> Any:
    if x is None: return None
    if isinstance(x, dict): return x
    if hasattr(x, "model_dump"): return x.model_dump()
    if hasattr(x, "__dict__"): return dict(x.__dict__)
    return {"repr": repr(x)}

# -----------------------------------------------------------------------------
# Collation Logic
# -----------------------------------------------------------------------------
def _is_numeric_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)

def _tensorize_value(v: Any, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if torch.is_tensor(v): return v
    if isinstance(v, np.ndarray):
        if v.dtype == object: return None
        return torch.from_numpy(v)
    if isinstance(v, (list, tuple)):
        if not v: return torch.tensor([], dtype=dtype)
        if all(_is_numeric_scalar(x) for x in v): return torch.tensor(v, dtype=dtype)
        try:
            if isinstance(v[0], (np.ndarray, list)):
                return torch.tensor(np.array(v), dtype=dtype)
        except: return None
    return None

def collate_encoded(encoded_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not encoded_list: return out
    keys = encoded_list[0].keys()
    for k in keys:
        vals = [e[k] for e in encoded_list]
        v0 = vals[0]
        if torch.is_tensor(v0):
            try: batched = torch.cat(vals, dim=0)
            except: batched = torch.stack(vals, dim=0)
            out[k] = batched.to(device=device, non_blocking=True)
            continue
        t0 = _tensorize_value(v0)
        if t0 is not None:
            t_vals = []
            ok = True
            for v in vals:
                tv = _tensorize_value(v)
                if tv is None: ok = False; break
                t_vals.append(tv)
            if ok:
                batched = torch.stack(t_vals, dim=0)
                out[k] = batched.to(device=device, non_blocking=True)
                continue
        out[k] = vals
    return out

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
class NgLoaded:
    def __init__(self, *, model: nn.Module, ckpt_config: Any, tokenizer_cfg: Any, device: torch.device):
        self.model = model
        self.ckpt_config = ckpt_config
        self.tokenizer_cfg = tokenizer_cfg
        self.device = device

def load_ng_checkpoint_faithful(ckpt_path: str, device: torch.device) -> NgLoaded:
    if not ckpt_path: raise NgPolicyError("ckpt_path is empty.")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    raw_config = ckpt.get("ckpt_config", None)
    if not isinstance(raw_config, dict): raise NgPolicyError("Checkpoint missing ckpt_config dict.")

    cfg_mod = _first_non_none(_try_import("nitrogen.cfg"), _try_import("nitrogen.config"), _try_import("config"))
    CkptConfig = _get_attr(cfg_mod, "CkptConfig")
    if CkptConfig is None: raise NgPolicyError("Could not import CkptConfig.")

    ckpt_config = CkptConfig.model_validate(raw_config)
    model_cfg = getattr(ckpt_config, "model_cfg", None)
    if model_cfg is None: raise NgPolicyError("Validated ckpt_config has no model_cfg.")

    tokenizer_cfg = raw_config.get("tokenizer_cfg", None)
    tok_mod = _try_import("nitrogen.mm_tokenizers")
    NitrogenTokenizerConfig = _get_attr(tok_mod, "NitrogenTokenizerConfig")
    if tokenizer_cfg and NitrogenTokenizerConfig:
        try: tokenizer_cfg = NitrogenTokenizerConfig.model_validate(tokenizer_cfg)
        except: pass

    mod = _try_import("nitrogen.flow_matching_transformer.nitrogen")
    NitroGen = _get_attr(mod, "NitroGen")
    if NitroGen is None: raise NgPolicyError("Could not import NitroGen.")

    try: model = NitroGen(config=model_cfg)
    except TypeError: model = NitroGen(config=model_cfg, game_mapping=None)

    state = ckpt.get("model", None)
    if state is None: raise NgPolicyError("Checkpoint missing 'model' state_dict.")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected: print(f"⚠️ Unexpected keys: {unexpected[:5]}...")
    if missing: print(f"⚠️ Missing keys: {missing[:5]}...")

    model.to(device)
    return NgLoaded(model=model, ckpt_config=ckpt_config, tokenizer_cfg=tokenizer_cfg, device=device)

# -----------------------------------------------------------------------------
# Dataset (Updated with Masking)
# -----------------------------------------------------------------------------
class CachedSplitDataset(Dataset):
    def __init__(self, root_dir: str, vision_horizon: int, balance_sampling: bool, active_ratio: float, press_threshold: float, base_seed: int, mask_path: str = None):
        self.root = Path(root_dir)
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files: raise FileNotFoundError(f"No .pt files found in {self.root}")

        self.vision_horizon = int(vision_horizon)
        self.balance_sampling = bool(balance_sampling)
        self.active_ratio = float(active_ratio)
        self.press_threshold = float(press_threshold)
        self.base_seed = int(base_seed) if base_seed > 0 else 0
        self._rng = None

        # --- MASK LOADING ---
        self.keep_mask = None
        if mask_path and os.path.exists(mask_path):
            print(f"🎭 Loading Noise Mask: {mask_path}")
            try:
                img = Image.open(mask_path).convert("RGBA")
                # Force resize to 256x256 (Training Res) to match frames
                img = img.resize((256, 256), Image.NEAREST)
                # Extract Alpha Channel [H, W]
                # Logic: User said "pixel that isn't alpha should be considered black"
                # This implies Opaque (Alpha > 0) -> Noise -> Black it out.
                # Transparent (Alpha == 0) -> Signal -> Keep it.
                alpha_tensor = TF.to_tensor(img)[3, :, :] # [256, 256] in [0, 1]
                
                # Boolean mask: True where we keep the pixel (Alpha == 0)
                # We add dimensions [1, 1, H, W] for easy broadcasting later
                self.keep_mask = (alpha_tensor == 0.0).bool().unsqueeze(0).unsqueeze(0)
                print("✅ Mask loaded and active.")
            except Exception as e:
                print(f"❌ Failed to load mask: {e}")

        # --- Indexing Logic ---
        self.cumulative_sizes = []
        self.file_lengths = []
        total = 0
        index_path = self.root / "dataset_index.json"
        index_valid = False
        
        if index_path.exists():
            try:
                with open(index_path, "r") as f: meta = json.load(f)
                if meta.get("filenames") == [f.name for f in self.files]:
                    self.cumulative_sizes = meta["cumulative_sizes"]
                    self.file_lengths = meta["file_lengths"]
                    total = self.cumulative_sizes[-1]
                    print(f"⚡ Loaded index for {self.root} ({total:,} frames).")
                    index_valid = True
            except: pass
        
        if not index_valid:
            print(f"🔍 Indexing {len(self.files)} files in {self.root}...")
            file_names = []
            valid_files = []
            for f in tqdm(self.files, desc="Indexing"):
                try:
                    d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
                    n = int(d["frames"].shape[0])
                    if n > 0:
                        total += n
                        self.cumulative_sizes.append(total)
                        self.file_lengths.append(n)
                        file_names.append(f.name)
                        valid_files.append(f)
                except Exception as e: print(f"⚠️ Skipping {f.name}: {e}")
            self.files = valid_files
            if total > 0:
                try:
                    with open(index_path, "w") as f:
                        json.dump({"filenames": file_names, "cumulative_sizes": self.cumulative_sizes, "file_lengths": self.file_lengths}, f)
                except: pass
        
        if total == 0: raise RuntimeError(f"No valid frames found in {self.root}")

        self.cache_idx = -1
        self.cache_data = None
        self._active_idx = None
        self._idle_idx = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _init_rng(self):
        if self._rng: return
        wi = get_worker_info()
        seed = self.base_seed + (1009 * wi.id if wi else 0)
        self._rng = np.random.default_rng(seed)

    def _load_file(self, idx):
        self.cache_data = torch.load(self.files[idx], map_location="cpu", weights_only=True, mmap=True)
        self.cache_idx = idx
        if self.balance_sampling:
            acts = self.cache_data["actions"]
            if acts.ndim == 3:
                btns = acts[:, 0, 4:]
                active = (btns > self.press_threshold).any(dim=1)
                self._active_idx = torch.where(active)[0]
                self._idle_idx = torch.where(~active)[0]

    def _choose_anchor(self, default, n):
        if not self.balance_sampling: return default
        self._init_rng()
        if self._active_idx is None: return default
        n_act, n_idl = len(self._active_idx), len(self._idle_idx)
        if n_act == 0: return default
        use_active = (self._rng.random() < self.active_ratio)
        pool = self._active_idx if use_active else self._idle_idx
        if len(pool) == 0: pool = self._active_idx
        return int(pool[self._rng.integers(0, len(pool))])

    def __getitem__(self, global_idx):
        file_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        if file_idx == 0: local_idx = global_idx
        else: local_idx = global_idx - self.cumulative_sizes[file_idx-1]

        if self.cache_idx != file_idx: self._load_file(file_idx)

        n = int(self.cache_data["frames"].shape[0])
        anchor = self._choose_anchor(local_idx, n)

        V = self.vision_horizon
        vis_ids = window_indices_end(anchor, V, n)
        
        frames_u8 = self.cache_data["frames"][vis_ids].clone()
        frames = frames_u8.float().div(255.0).mul(2.0).sub(1.0)

        # --- APPLY MASK ---
        if self.keep_mask is not None:
            # Mask is [1, 1, 256, 256]. Frames is [V, 3, 256, 256].
            # Broadcasting handles the V and C dimensions.
            # ~keep_mask selects opaque pixels. We set them to -1.0 (Black).
            frames.masked_fill_(~self.keep_mask, -1.0)

        actions_window = self.cache_data["actions"][anchor].float().clone()
        
        return {
            "frames": frames,
            "j_left": actions_window[:, 0:2],
            "j_right": actions_window[:, 2:4],
            "buttons": actions_window[:, 4:],
            "dropped_frames": torch.zeros(V, dtype=torch.bool),
            "game": "bn6"
        }