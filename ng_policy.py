from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# JSON Summary Helper
# -----------------------------------------------------------------------------

def _summarize_tensor(x: Any, *, max_items: int = 16) -> Any:
    """
    JSON-safe summary for debugging.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
        flat = t.reshape(-1)
        n = int(flat.numel())
        k = min(max_items, n)

        sample = []
        if k > 0:
            sample = flat[:k].to("cpu", non_blocking=True).float().tolist()

        stats = {}
        if n > 0 and t.dtype.is_floating_point:
            tcpu = flat.to("cpu", non_blocking=True).float()
            stats = {
                "min": float(tcpu.min().item()),
                "max": float(tcpu.max().item()),
                "mean": float(tcpu.mean().item()),
                "std": float(tcpu.std(unbiased=False).item()),
            }
        elif n > 0 and (t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)):
            tcpu = flat.to("cpu", non_blocking=True)
            if tcpu.dtype == torch.bool:
                stats = {"true": int(tcpu.sum().item()), "count": n}
            else:
                stats = {
                    "min": int(tcpu.min().item()),
                    "max": int(tcpu.max().item()),
                }

        return {
            "type": "tensor",
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "device": str(t.device),
            "numel": n,
            "sample": sample,
            "stats": stats,
        }

    if isinstance(x, (list, tuple)):
        out = []
        for it in x[:max_items]:
            out.append(_summarize_tensor(it, max_items=max_items))
        return {"type": "list", "len": len(x), "items": out}

    if isinstance(x, dict):
        keys = list(x.keys())
        keys_sorted = sorted(keys)[:64]
        return {str(k): _summarize_tensor(x[k], max_items=max_items) for k in keys_sorted}

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return {"type": "repr", "value": repr(x)[:400]}


class NgPolicyError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Dynamic Import Helpers
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Checkpoint Loading with Patching
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NgLoaded:
    model: nn.Module
    ckpt_config: Any
    device: torch.device

# Minimal fallback container if CkptConfig validation fails completely
class DummyConfig:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg

def load_ng_checkpoint(ckpt_path: str, device: torch.device) -> NgLoaded:
    if not ckpt_path:
        raise NgPolicyError("ckpt_path is empty.")
    if device is None:
        raise NgPolicyError("device is None.")

    # Try to silence pydantic protected namespace warnings
    warnings.filterwarnings(
        "ignore",
        message=r'.*protected namespace "model_".*',
        category=UserWarning,
        module=r"pydantic\..*",
    )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if not isinstance(ckpt, dict):
        raise NgPolicyError(f"Checkpoint is not a dict. Got: {type(ckpt)}")
    if "model" not in ckpt or "ckpt_config" not in ckpt:
        raise NgPolicyError(f"Checkpoint missing required keys. Found: {list(ckpt.keys())}")

    raw_config = ckpt["ckpt_config"]

    # =========================================================================
    # CONFIG PATCHING
    # =========================================================================
    # The installed Nitrogen library may require fields (experiment_name, modality_cfg)
    # that our custom training script did not save. We inject placeholders here.
    
    if isinstance(raw_config, dict):
        # 1. Patch top-level required fields
        if "experiment_name" not in raw_config:
            raw_config["experiment_name"] = "inference_patch"
        
        if "modality_cfg" not in raw_config:
            # Provide a dummy modality config to satisfy validation
            raw_config["modality_cfg"] = {
                "modality_type": "image", 
                "image_size": 256,
                "num_channels": 3
            }

        # 2. Patch model_cfg required fields
        if "model_cfg" in raw_config:
            mcfg = raw_config["model_cfg"]
            if isinstance(mcfg, dict):
                # 'num_inference_timesteps' is required by some Nitrogen versions
                if "num_inference_timesteps" not in mcfg or mcfg.get("num_inference_timesteps") is None:
                    mcfg["num_inference_timesteps"] = 10 

    # =========================================================================
    # CONFIG LOADING
    # =========================================================================
    
    cfg_mod = _first_non_none(
        _try_import("nitrogen.cfg"),
        _try_import("nitrogen.config"),
        _try_import("nitrogen.configs"),
        _try_import("cfg"),
        _try_import("config"),
    )
    if cfg_mod is None:
        raise NgPolicyError("Could not import Nitrogen config module.")

    CkptConfig = _get_attr(cfg_mod, "CkptConfig")
    NitroGen_Config = _get_attr(cfg_mod, "NitroGen_Config") # Try to get the inner config class

    ckpt_config = None
    model_cfg = None

    # Attempt 1: Full Experiment Config Validation
    if CkptConfig is not None:
        try:
            ckpt_config = CkptConfig.model_validate(raw_config)
            model_cfg = getattr(ckpt_config, "model_cfg", None)
        except Exception as e:
            print(f"⚠️ CkptConfig validation failed: {e}")
            print("⚠️ Attempting fallback to raw NitroGen_Config...")

    # Attempt 2: Direct Model Config Validation (Fallback)
    if model_cfg is None and "model_cfg" in raw_config:
        # If we couldn't validate the whole experiment, try just the model part
        raw_model_cfg = raw_config["model_cfg"]
        
        # If NitroGen_Config is available, try to validate against it
        if NitroGen_Config is not None:
            try:
                model_cfg = NitroGen_Config.model_validate(raw_model_cfg)
                ckpt_config = DummyConfig(model_cfg) # Wrap it so downstream code works
            except Exception as e:
                print(f"⚠️ NitroGen_Config validation failed: {e}")
        
        # Last Resort: Pass the raw dict (NitroGen might accept it if it's not typed strictly)
        if model_cfg is None:
             print("⚠️ Using raw dict for model config (Risky).")
             model_cfg = raw_model_cfg
             ckpt_config = DummyConfig(model_cfg)

    if model_cfg is None:
         raise NgPolicyError("Failed to resolve model_cfg from checkpoint.")

    # =========================================================================
    # MODEL INSTANTIATION
    # =========================================================================

    mod = _try_import("nitrogen.flow_matching_transformer.nitrogen")
    NitroGen = _get_attr(mod, "NitroGen")
    if NitroGen is None:
        raise NgPolicyError("Could not import NitroGen class.")

    # Optional game mapping
    game_mapping = None
    mapping_mod = _first_non_none(
        _try_import("nitrogen.envs.mapping"),
        _try_import("nitrogen.mapping"),
    )
    if mapping_mod is not None:
        fn = _get_attr(mapping_mod, "default_game_mapping")
        if callable(fn):
            try:
                game_mapping = fn()
            except Exception:
                game_mapping = None

    try:
        model = NitroGen(config=model_cfg, game_mapping=game_mapping)
    except TypeError:
        model = NitroGen(config=model_cfg)

    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    if len(missing) > 0:
        print(f"Check point loaded with {len(missing)} missing keys (likely optimizer/head).")
    
    model.to(device).eval()
    return NgLoaded(model=model, ckpt_config=ckpt_config, device=device)


# -----------------------------------------------------------------------------
# Policy Classes (Unchanged)
# -----------------------------------------------------------------------------

# These token IDs come directly from the NitroGen file you pasted.
_PAD_TOKEN = 0
_IMG_TOKEN = 1
_ACT_TOKEN = 4
_GAME_ID_TOKEN = 6

class NgDiscreteActionScorer(nn.Module):
    def __init__(self, action_prototypes: torch.Tensor, temperature: float = 1.0):
        super().__init__()
        self.register_buffer("protos", action_prototypes.float(), persistent=False)
        self.temperature = float(temperature)

    def forward(self, action_vec: torch.Tensor) -> torch.Tensor:
        diff = action_vec.unsqueeze(1) - self.protos.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)
        return -dist2 / self.temperature

class NgNitroGenPolicy(nn.Module):
    def __init__(
        self,
        ng: NgLoaded,
        *,
        default_game_id: Optional[int] = None,
        discrete_action_prototypes: Optional[torch.Tensor] = None,
        discrete_temperature: float = 1.0,
    ):
        super().__init__()
        self.ng = ng.model
        self.cfg = ng.ckpt_config.model_cfg
        self.default_game_id = "bn6"

        # Robustly handle different config object types (Pydantic vs SimpleNamespace vs Dict)
        self.action_dim = int(getattr(self.cfg, "action_dim", self.cfg.get("action_dim", 25) if isinstance(self.cfg, dict) else 25))
        self.action_horizon = int(getattr(self.cfg, "action_horizon", self.cfg.get("action_horizon", 1) if isinstance(self.cfg, dict) else 1))
        self.max_seq_len = int(getattr(self.cfg, "max_seq_len", self.cfg.get("max_seq_len", 1024) if isinstance(self.cfg, dict) else 1024))

        self.tokens_per_image = self._infer_tokens_per_image()

        self.discrete = None
        if discrete_action_prototypes is not None:
            self.discrete = NgDiscreteActionScorer(discrete_action_prototypes, temperature=discrete_temperature)

    @torch.no_grad()
    def _infer_tokens_per_image(self) -> int:
        ve = getattr(self.ng, "vision_encoder", None)
        if ve is None: return 256 # Fallback guess if missing
        
        image_size = 256
        ve_cfg = getattr(ve, "config", None)
        if ve_cfg is not None:
            image_size = getattr(ve_cfg, "image_size", 256)

        dummy = torch.zeros((1, 3, image_size, image_size), device=self.ng.device, dtype=self.ng.dtype)
        out = ve(dummy)
        lhs = out["last_hidden_state"] if isinstance(out, dict) else getattr(out, "last_hidden_state", None)
        if lhs is None: return 256
        return int(lhs.shape[1])

    def _cap_num_frames(self, T: int) -> int:
        extra = 1 if self._uses_game_token() else 0
        max_frames = (self.max_seq_len - extra) // self.tokens_per_image
        if max_frames <= 0: max_frames = 1
        return min(T, max_frames)

    def _uses_game_token(self) -> bool:
        return getattr(self.ng, "game_mapping", None) is not None and self.default_game_id is not None

    def _build_vl_tokens(self, B: int, num_frames: int, device: torch.device) -> torch.Tensor:
        extra = 1 if self._uses_game_token() else 0
        vl_len = num_frames * self.tokens_per_image + extra
        vl = torch.full((B, vl_len), _IMG_TOKEN, dtype=torch.long, device=device)
        if extra == 1:
            vl[:, 0] = _GAME_ID_TOKEN
        return vl

    def _build_sa_tokens(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.full((B, self.action_horizon), _ACT_TOKEN, dtype=torch.long, device=device)

    def _build_masks(self, B: int, vl_len: int, num_frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        dropped_images = torch.zeros((B, num_frames), dtype=torch.long, device=device)
        vl_attn_mask = torch.ones((B, vl_len), dtype=torch.long, device=device)
        return dropped_images, vl_attn_mask

    def _vision_image_size(self) -> int:
        ve = getattr(self.ng, "vision_encoder", None)
        cfg = getattr(ve, "config", None) if ve is not None else None
        image_size = getattr(cfg, "image_size", None)
        return int(image_size) if image_size is not None else 256

    def _preprocess_images(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        target = self._vision_image_size()
        if H == target and W == target: return frames
        x = frames.reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        return x.reshape(B, T, C, target, target)

    @torch.inference_mode()
    def forward(
        self,
        frames: torch.Tensor,
        *,
        seed: Optional[int] = None,
        game_id: Optional[int] = None,
        take_step: int = 0,
        return_continuous: bool = False,
        return_raw: bool = False,
        raw_max_items: int = 16,
    ) -> Any:
        
        B, T = int(frames.shape[0]), int(frames.shape[1])
        device = frames.device

        num_frames = self._cap_num_frames(T)
        images = self._preprocess_images(frames[:, -num_frames:].contiguous())

        vl_token_ids = self._build_vl_tokens(B, num_frames, device=device)
        sa_token_ids = self._build_sa_tokens(B, device=device)
        dropped_images, vl_attn_mask = self._build_masks(
            B, vl_len=vl_token_ids.shape[1], num_frames=num_frames, device=device
        )

        data: dict[str, Any] = {
            "embodiment_id": torch.zeros((B,), dtype=torch.long, device=device),
            "images": images,
            "vl_token_ids": vl_token_ids,
            "sa_token_ids": sa_token_ids,
            "dropped_images": dropped_images,
            "vl_attn_mask": vl_attn_mask,
            "game_ids": torch.full((B,), 0, dtype=torch.long, device=device),
        }

        gid = game_id if game_id is not None else self.default_game_id
        if gid is not None:
            data["game_ids"] = torch.full((B,), self.default_game_id, dtype=torch.long, device=device)



        if seed is not None:
            with torch.random.fork_rng(devices=[device]):
                torch.manual_seed(int(seed))
                out = self.ng.get_action(data)
        else:
            out = self.ng.get_action(data)

        actions = out["action_tensor"]
        action_vec = actions[:, take_step, :]

        primary: Any
        if return_continuous or self.discrete is None:
            primary = action_vec
        else:
            primary = self.discrete(action_vec)

        if not return_raw:
            return primary

        raw_summary: dict[str, Any] = {
            "take_step": int(take_step),
            "data": {
                "vl_token_ids": _summarize_tensor(vl_token_ids, max_items=raw_max_items),
                "images": _summarize_tensor(images, max_items=raw_max_items),
            },
            "primary": _summarize_tensor(primary, max_items=raw_max_items),
        }
        return primary, raw_summary