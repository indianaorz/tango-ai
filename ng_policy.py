from __future__ import annotations
import importlib
import warnings
import traceback
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
except Exception:
    NitrogenTokenizer = None
    NitrogenTokenizerConfig = None

# -----------------------------------------------------------------------------
# JSON Summary Helper
# -----------------------------------------------------------------------------
def _summarize_tensor(x: Any, *, max_items: int = 16) -> Any:
    if isinstance(x, torch.Tensor):
        t = x.detach()
        flat = t.reshape(-1)
        n = int(flat.numel())
        k = min(max_items, n)
        sample = []
        if k > 0:
            sample = flat[:k].to("cpu", non_blocking=True).float().tolist()
        return {
            "type": "tensor",
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "device": str(t.device),
            "numel": n,
            "sample": sample,
        }
    if isinstance(x, (list, tuple)):
        return {"type": "list", "len": len(x), "items": [_summarize_tensor(it, max_items=max_items) for it in x[:max_items]]}
    if isinstance(x, dict):
        keys = list(x.keys())
        keys_sorted = sorted(keys)[:64]
        return {str(k): _summarize_tensor(x[k], max_items=max_items) for k in keys_sorted}
    return {"type": "repr", "value": repr(x)[:200]}

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
# Checkpoint Loading
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class NgLoaded:
    model: nn.Module
    ckpt_config: Any
    tokenizer_cfg: Any
    device: torch.device

class ObjConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ObjConfig(v))
            else:
                setattr(self, k, v)
    def get(self, key, default=None):
        return getattr(self, key, default)

def load_ng_checkpoint(ckpt_path: str, device: torch.device) -> NgLoaded:
    if not ckpt_path: raise NgPolicyError("ckpt_path is empty.")
    if device is None: raise NgPolicyError("device is None.")

    warnings.filterwarnings("ignore", message=r'.*protected namespace "model_".*', category=UserWarning)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if not isinstance(ckpt, dict):
        raise NgPolicyError(f"Checkpoint is not a dict. Got: {type(ckpt)}")
    
    raw_config = ckpt.get("ckpt_config", {})
    raw_tokenizer_cfg = None
    
    if isinstance(raw_config, dict):
        raw_tokenizer_cfg = raw_config.get("tokenizer_cfg", None)
        if "model_cfg" in raw_config:
            mcfg = raw_config["model_cfg"]
            if isinstance(mcfg, dict):
                if mcfg.get("num_inference_timesteps") is None:
                    mcfg["num_inference_timesteps"] = 10 

    cfg_mod = _first_non_none(
        _try_import("nitrogen.cfg"),
        _try_import("nitrogen.config"),
        _try_import("config"),
    )
    
    CkptConfig = _get_attr(cfg_mod, "CkptConfig")
    NitroGen_Config = _get_attr(cfg_mod, "NitroGen_Config")
    
    ckpt_config = None
    model_cfg = None

    if CkptConfig is not None and isinstance(raw_config, dict):
        try:
            if "experiment_name" not in raw_config: raw_config["experiment_name"] = "inf"
            if "modality_cfg" not in raw_config: raw_config["modality_cfg"] = {"modality_type": "image"}
            ckpt_config = CkptConfig.model_validate(raw_config)
            model_cfg = getattr(ckpt_config, "model_cfg", None)
        except Exception: pass

    if model_cfg is None and "model_cfg" in raw_config:
        raw_model_cfg = raw_config["model_cfg"]
        print("⚠️ Using raw dict for model config (Converted to Object).")
        model_cfg = ObjConfig(raw_model_cfg)
        ckpt_config = ObjConfig({"model_cfg": raw_model_cfg})

    if model_cfg is None:
         raise NgPolicyError("Failed to resolve model_cfg from checkpoint.")

    tokenizer_cfg = raw_tokenizer_cfg
    tok_mod = _try_import("nitrogen.mm_tokenizers")
    NitrogenTokenizerConfig = _get_attr(tok_mod, "NitrogenTokenizerConfig")
    
    if tokenizer_cfg is not None and NitrogenTokenizerConfig is not None:
        try:
            tokenizer_cfg = NitrogenTokenizerConfig.model_validate(tokenizer_cfg)
        except Exception: pass

    mod = _try_import("nitrogen.flow_matching_transformer.nitrogen")
    NitroGen = _get_attr(mod, "NitroGen")
    
    try:
        model = NitroGen(config=model_cfg)
    except TypeError:
        model = NitroGen(config=model_cfg, game_mapping=None)

    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"Check point loaded with {len(missing)} missing keys.")

    model.to(device).eval()
    return NgLoaded(model=model, ckpt_config=ckpt_config, tokenizer_cfg=tokenizer_cfg, device=device)

# -----------------------------------------------------------------------------
# Policy Classes
# -----------------------------------------------------------------------------
class NgDiscreteActionScorer(nn.Module):
    def __init__(self, action_prototypes: torch.Tensor, temperature: float = 1.0):
        super().__init__()
        self.register_buffer("protos", action_prototypes.float(), persistent=False)
        self.temperature = float(temperature)

    def forward(self, action_vec: torch.Tensor) -> torch.Tensor:
        diff = action_vec.unsqueeze(1) - self.protos.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)
        return -dist2 / self.temperature

def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.number))

def _tensorize_numeric(v):
    if torch.is_tensor(v): return v
    if isinstance(v, np.ndarray): 
        if v.dtype == object: return v 
        return torch.from_numpy(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 0: return torch.tensor([])
        if all(_is_num(x) for x in v):
            return torch.tensor(v)
        try:
            if all(isinstance(x, (list, tuple, np.ndarray)) for x in v):
                return torch.tensor(v)
        except Exception:
            return v
    return v

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
        self.cfg = getattr(ng.ckpt_config, "model_cfg", None)
        if self.cfg is None: self.cfg = ng.ckpt_config
        self.default_game_id = default_game_id
        
        self._use_tokenizer = os.getenv("NG_USE_TOKENIZER", "1").strip() not in ("0", "false", "False")
        self._tokenize_on_cpu = os.getenv("NG_TOKENIZE_ON_CPU", "1").strip() not in ("0", "false", "False")
        self.tokenizer = None
        self.tokenizer_cfg = getattr(ng, "tokenizer_cfg", None)

        if self._use_tokenizer:
            if NitrogenTokenizer is None or self.tokenizer_cfg is None:
                print("⚠️ Tokenizer missing. Falling back to manual.")
                self._use_tokenizer = False
            else:
                try:
                    cfg_obj = self.tokenizer_cfg
                    if isinstance(cfg_obj, dict) and NitrogenTokenizerConfig is not None:
                        cfg_obj = NitrogenTokenizerConfig.model_validate(cfg_obj)
                    self.tokenizer = NitrogenTokenizer(cfg_obj)
                    if hasattr(self.tokenizer, "vision_encoder"):
                        self.tokenizer.vision_encoder.eval()
                except Exception as e:
                    print(f"⚠️ Tokenizer init failed: {e}")
                    self._use_tokenizer = False

        self.discrete = None
        if discrete_action_prototypes is not None:
            self.discrete = NgDiscreteActionScorer(discrete_action_prototypes, temperature=discrete_temperature)

    def _collate_encoded(self, encoded_list: list[dict[str, Any]], *, device: torch.device) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            keys = encoded_list[0].keys()
            for k in keys:
                vals = [e[k] for e in encoded_list]
                
                # Try to convert to tensor
                first = _tensorize_numeric(vals[0])
                
                if torch.is_tensor(first):
                    t_vals = [_tensorize_numeric(v) for v in vals]
                    if all(torch.is_tensor(tv) for tv in t_vals):
                        try:
                            # --- CRITICAL FIX ---
                            # If items are 0-dim (scalar) -> Stack -> [B]
                            # If items are 1-dim (vector) -> Stack -> [B, Dim]
                            # If items are >1-dim (e.g. [1, T, ...]) -> Cat -> [B, T, ...]
                            
                            ndim = t_vals[0].ndim
                            if ndim <= 1:
                                batched = torch.stack(t_vals, dim=0)
                            else:
                                try:
                                    batched = torch.cat(t_vals, dim=0)
                                except Exception:
                                    # Fallback for ragged or mismatched
                                    batched = torch.stack(t_vals, dim=0)
                        except Exception:
                            out[k] = vals
                            continue
                        
                        batched = batched.to(device=device, non_blocking=True)
                        curr_bs = batched.shape[0]

                        # --- FIX DIMENSIONS FOR MODEL ---
                        # Vision: [B, T, C, H, W]
                        if k in ("pixel_values", "frames", "images"):
                            if batched.ndim == 4:
                                batched = batched.unsqueeze(1)
                        
                        # Tokens: [B, Seq]
                        # Catch ANY key ending in _ids, _mask, or known special keys
                        is_token_key = k in ("sa_token_ids", "sa_mask", "input_ids", "attention_mask", "vl_token_ids")
                        if is_token_key or k.endswith("_ids") or k.endswith("_mask"):
                            # If we have [B*Seq] (1D) -> unsqueeze -> [B*Seq, 1] 
                            # (But ideally we avoided this via stack above)
                            if batched.ndim == 1: 
                                batched = batched.unsqueeze(1)
                            elif batched.ndim > 2:
                                batched = batched.view(curr_bs, -1)
                        
                        out[k] = batched
                        continue
                
                out[k] = vals
            return out
        except Exception as e:
            print(f"[DEBUG] Collation CRASH: {e}")
            traceback.print_exc()
            raise

    def _encode_with_tokenizer(self, frames: torch.Tensor) -> dict[str, Any]:
        if self.tokenizer is None:
            raise NgPolicyError("Tokenizer is not initialized.")
        
        B, T, C, H, W = frames.shape
        encoded_list: list[dict[str, Any]] = []
        
        frames_src = frames
        if self._tokenize_on_cpu and frames_src.is_cuda:
            frames_src = frames_src.detach().cpu()
        frames_src = frames_src.to(dtype=torch.float32, copy=False)

        for i in range(B):
            j_left  = torch.zeros((1, 1, 2), dtype=torch.float32)
            j_right = torch.zeros((1, 1, 2), dtype=torch.float32)
            buttons = torch.zeros((1, 1, 21), dtype=torch.float32)
            dropped = torch.zeros((1, 1), dtype=torch.bool)
            
            sample = {
                "frames": frames_src[i], 
                "j_left": j_left,
                "j_right": j_right,
                "buttons": buttons,
                "dropped_frames": dropped,
                "game": "bn6",
            }
            enc = self.tokenizer.encode(sample)
            encoded_list.append(enc)
            
        return self._collate_encoded(encoded_list, device=self.ng.device)

    @torch.inference_mode()
    def forward(
        self,
        frames: torch.Tensor,
        *,
        seed: Optional[int] = None,
        take_step: int = 0,
        return_continuous: bool = False,
        return_raw: bool = False,
        raw_max_items: int = 16,
    ) -> Any:
        
        try:
            x = frames
            if x.dtype != torch.float32: x = x.float()

            if self._use_tokenizer and self.tokenizer is not None:
                data = self._encode_with_tokenizer(x)
            else:
                raise NgPolicyError("Manual token construction not supported.")

            model_dev = self.ng.device
            if seed is not None:
                device_type = "cuda" if model_dev.type == "cuda" else "cpu"
                if device_type == "cuda":
                    with torch.random.fork_rng(devices=[model_dev]):
                        torch.manual_seed(int(seed))
                        out = self.ng.get_action(data)
                else:
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
                "data_keys": sorted(list(data.keys()))[:64],
                "data": {
                    "pixel_values": _summarize_tensor(data.get("pixel_values"), max_items=raw_max_items),
                    "sa_token_ids": _summarize_tensor(data.get("sa_token_ids"), max_items=raw_max_items),
                },
                "primary": _summarize_tensor(primary, max_items=raw_max_items),
            }
            return primary, raw_summary
        
        except Exception as e:
            print(f"[DEBUG] Forward loop CRASHED: {e}")
            traceback.print_exc()
            raise