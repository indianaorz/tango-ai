# ng_policy.py
from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F




def _summarize_tensor(x: Any, *, max_items: int = 16) -> Any:
    """
    JSON-safe summary for debugging.
    - If x is a tensor: returns dict with shape/dtype + a small sample + basic stats
    - If x is list/tuple: summarize first few items
    - Else: return str(x) fallback
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
        # Move a tiny slice to CPU for stable JSON
        flat = t.reshape(-1)
        n = int(flat.numel())
        k = min(max_items, n)

        sample = []
        if k > 0:
            sample = flat[:k].to("cpu", non_blocking=True).float().tolist()

        # Stats: only if numeric and non-empty
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
            # bool -> int for readability
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
        # summarize keys, but don't explode: keep stable order and cap
        keys = list(x.keys())
        keys_sorted = sorted(keys)[:64]
        return {str(k): _summarize_tensor(x[k], max_items=max_items) for k in keys_sorted}

    # primitive
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return {"type": "repr", "value": repr(x)[:400]}




class NgPolicyError(RuntimeError):
    pass


class NgPolicyWrapper(nn.Module):
    """
    Wrap a NitroGen policy (or compatible backbone) and add a small discrete-action head.

    Expects the backbone to return a continuous vector of shape [B, D] when called.
    We project that to logits [B, num_actions].

    This uses LazyLinear so D can be unknown at construction time.
    """

    def __init__(self, backbone: nn.Module, num_actions: int):
        super().__init__()
        if num_actions <= 0:
            raise ValueError(f"num_actions must be > 0, got {num_actions}")
        self.backbone = backbone
        self.head = nn.LazyLinear(num_actions)

    def forward(
        self,
        frames: torch.Tensor,
        _game_features: Optional[Any] = None,
        *,
        seed: Optional[int] = None,
        return_continuous: bool = True,
    ) -> torch.Tensor:
        """
        frames: [B,T,C,H,W]
        returns: logits [B, num_actions]
        """
        if not isinstance(frames, torch.Tensor):
            raise NgPolicyError(f"frames must be a torch.Tensor, got {type(frames)}")
        if frames.dim() != 5:
            raise NgPolicyError(f"frames must be [B,T,C,H,W], got shape={tuple(frames.shape)}")

        # Call backbone in the most compatible way possible.
        try:
            cont = self.backbone(frames, seed=seed, return_continuous=return_continuous)
        except TypeError:
            # Fallback if backbone doesn't accept seed/return_continuous kwargs
            cont = self.backbone(frames)

        if not isinstance(cont, torch.Tensor):
            raise NgPolicyError(f"backbone output must be torch.Tensor, got {type(cont)}")
        if cont.dim() != 2:
            raise NgPolicyError(f"backbone output must be [B,D], got shape={tuple(cont.shape)}")

        logits = self.head(cont)
        return logits
    
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


@dataclass(frozen=True)
class NgLoaded:
    model: nn.Module
    ckpt_config: Any
    device: torch.device


def load_ng_checkpoint(ckpt_path: str, device: torch.device) -> NgLoaded:
    if not ckpt_path:
        raise NgPolicyError("ckpt_path is empty.")
    if device is None:
        raise NgPolicyError("device is None.")

    # Try to silence pydantic protected namespace warnings (best-effort).
    warnings.filterwarnings(
        "ignore",
        message=r'.*protected namespace "model_".*',
        category=UserWarning,
        module=r"pydantic\..*",
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise NgPolicyError(f"Checkpoint is not a dict. Got: {type(ckpt)}")
    if "model" not in ckpt or "ckpt_config" not in ckpt:
        raise NgPolicyError(f"Checkpoint missing required keys. Found: {list(ckpt.keys())}")

    # ---- Import config + validate ckpt_config ----
    cfg_mod = _first_non_none(
        _try_import("nitrogen.cfg"),
        _try_import("nitrogen.config"),
        _try_import("nitrogen.configs"),
        _try_import("cfg"),
        _try_import("config"),
    )
    if cfg_mod is None:
        raise NgPolicyError(
            "Could not import Nitrogen config module.\n"
            "Tried: nitrogen.cfg / nitrogen.config / nitrogen.configs / cfg / config\n"
            "Fix: ensure NitroGen repo/package is installed or on PYTHONPATH."
        )

    CkptConfig = _get_attr(cfg_mod, "CkptConfig")
    if CkptConfig is None:
        raise NgPolicyError("Config module does not expose CkptConfig.")

    try:
        ckpt_config = CkptConfig.model_validate(ckpt["ckpt_config"])
    except Exception as e:
        raise NgPolicyError(f"Failed to validate ckpt_config via CkptConfig.model_validate: {e}") from e

    model_cfg = getattr(ckpt_config, "model_cfg", None)
    if model_cfg is None:
        raise NgPolicyError("ckpt_config has no model_cfg attribute.")

    # ---- Import NitroGen class ----
    mod = _try_import("nitrogen.flow_matching_transformer.nitrogen")
    NitroGen = _get_attr(mod, "NitroGen")
    if NitroGen is None:
        raise NgPolicyError("Could not import NitroGen from nitrogen.flow_matching_transformer.nitrogen")

    # ---- Optional game_mapping support ----
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

    # ---- Instantiate + load ----
    try:
        model = NitroGen(config=model_cfg, game_mapping=game_mapping)
    except TypeError:
        model = NitroGen(config=model_cfg)

    state = ckpt["model"]
    if not isinstance(state, dict):
        raise NgPolicyError(f'Checkpoint["model"] is not a state_dict dict. Got: {type(state)}')

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 50:
        raise NgPolicyError(f"Too many missing keys when loading {ckpt_path}: {len(missing)}. Example: {missing[:10]}")
    if len(unexpected) > 50:
        raise NgPolicyError(
            f"Too many unexpected keys when loading {ckpt_path}: {len(unexpected)}. Example: {unexpected[:10]}"
        )

    model.to(device).eval()
    return NgLoaded(model=model, ckpt_config=ckpt_config, device=device)


# These token IDs come directly from the NitroGen file you pasted.
_PAD_TOKEN = 0
_IMG_TOKEN = 1
_ACT_TOKEN = 4
_GAME_ID_TOKEN = 6


class NgDiscreteActionScorer(nn.Module):
    """
    Deterministic "logits over a discrete action set" by scoring a continuous action
    against prototype vectors.

    logits[a] = -||action - proto[a]||^2 / temperature
    """

    def __init__(self, action_prototypes: torch.Tensor, temperature: float = 1.0):
        super().__init__()
        if not isinstance(action_prototypes, torch.Tensor):
            raise ValueError("action_prototypes must be a torch.Tensor")
        if action_prototypes.dim() != 2:
            raise ValueError(f"action_prototypes must be [A, D]. Got {tuple(action_prototypes.shape)}")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.register_buffer("protos", action_prototypes.float(), persistent=False)
        self.temperature = float(temperature)

    def forward(self, action_vec: torch.Tensor) -> torch.Tensor:
        """
        action_vec: [B, D]
        returns logits: [B, A]
        """
        if action_vec.dim() != 2:
            raise NgPolicyError(f"action_vec must be [B,D]. Got {tuple(action_vec.shape)}")
        if action_vec.shape[1] != self.protos.shape[1]:
            raise NgPolicyError(
                f"action_dim mismatch: action_vec has D={action_vec.shape[1]} but protos have D={self.protos.shape[1]}"
            )

        # [B,1,D] - [1,A,D] -> [B,A,D]
        diff = action_vec.unsqueeze(1) - self.protos.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)  # [B,A]
        return -dist2 / self.temperature


class NgNitroGenPolicy(nn.Module):
    """
    Correct inference wrapper around NitroGen.

    - Calls NitroGen.get_action(data) (NOT forward()).
    - Builds required token tensors in the shape expected by prepare_input_embs().
    - Optionally returns discrete logits by scoring the sampled continuous action
      against a prototype set.

    Assumptions (explicit):
      - frames are already resized to the vision encoder input size (often 256x256).
      - frames are already normalized the same way they were during training.
        (If not, we should add explicit preprocessing here.)
    """

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
        self.default_game_id = default_game_id

        # Cache these from config
        self.action_dim = int(self.cfg.action_dim)
        self.action_horizon = int(self.cfg.action_horizon)
        self.max_seq_len = int(getattr(self.cfg, "max_seq_len", 1024))

        # Infer tokens_per_image by running vision encoder once on a dummy image
        self.tokens_per_image = self._infer_tokens_per_image()

        self.discrete = None
        if discrete_action_prototypes is not None:
            if discrete_action_prototypes.shape[1] != self.action_dim:
                raise NgPolicyError(
                    f"prototype action_dim mismatch: protos D={discrete_action_prototypes.shape[1]} "
                    f"but model action_dim={self.action_dim}"
                )
            self.discrete = NgDiscreteActionScorer(discrete_action_prototypes, temperature=discrete_temperature)

    @torch.no_grad()
    def _infer_tokens_per_image(self) -> int:
        # We infer token count from vision encoder output length.
        # This must match what encode_images() produces: last_hidden_state length.
        ve = getattr(self.ng, "vision_encoder", None)
        if ve is None:
            raise NgPolicyError("NitroGen model has no vision_encoder attribute; cannot infer tokens_per_image.")

        # Use config image size if present; otherwise assume already correct and use 256.
        image_size = None
        ve_cfg = getattr(ve, "config", None)
        if ve_cfg is not None:
            image_size = getattr(ve_cfg, "image_size", None)
        if image_size is None:
            image_size = 256

        dummy = torch.zeros((1, 3, image_size, image_size), device=self.ng.device, dtype=self.ng.dtype)
        out = ve(dummy)
        lhs = out["last_hidden_state"] if isinstance(out, dict) else getattr(out, "last_hidden_state", None)
        if lhs is None:
            raise NgPolicyError("vision_encoder output missing last_hidden_state; cannot infer tokens_per_image.")
        return int(lhs.shape[1])

    def _cap_num_frames(self, T: int) -> int:
        # vl_token_ids length will be:
        #   num_frames * tokens_per_image (+1 if we include a game token)
        extra = 1 if self._uses_game_token() else 0
        max_frames = (self.max_seq_len - extra) // self.tokens_per_image
        if max_frames <= 0:
            raise NgPolicyError(
                f"max_seq_len={self.max_seq_len} too small for tokens_per_image={self.tokens_per_image}."
            )
        return min(T, max_frames)

    def _uses_game_token(self) -> bool:
        # NitroGen only uses game token if it was constructed with a non-None mapping AND you pass game_ids.
        return getattr(self.ng, "game_mapping", None) is not None and self.default_game_id is not None

    def _build_vl_tokens(self, B: int, num_frames: int, device: torch.device) -> torch.Tensor:
        extra = 1 if self._uses_game_token() else 0
        vl_len = num_frames * self.tokens_per_image + extra
        if vl_len > self.max_seq_len:
            raise NgPolicyError(f"vl_len={vl_len} exceeds max_seq_len={self.max_seq_len} after capping.")

        vl = torch.full((B, vl_len), _IMG_TOKEN, dtype=torch.long, device=device)
        if extra == 1:
            # put the game token at position 0
            vl[:, 0] = _GAME_ID_TOKEN
        return vl

    def _build_sa_tokens(self, B: int, device: torch.device) -> torch.Tensor:
        # Minimal SA stream: exactly action_horizon ACT tokens.
        return torch.full((B, self.action_horizon), _ACT_TOKEN, dtype=torch.long, device=device)

    def _build_masks(self, B: int, vl_len: int, num_frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        dropped_images = torch.zeros((B, num_frames), dtype=torch.long, device=device)  # 0 = keep
        vl_attn_mask = torch.ones((B, vl_len), dtype=torch.long, device=device)        # 1 = attend
        return dropped_images, vl_attn_mask


    def _vision_image_size(self) -> int:
        ve = getattr(self.ng, "vision_encoder", None)
        cfg = getattr(ve, "config", None) if ve is not None else None
        image_size = getattr(cfg, "image_size", None)
        return int(image_size) if image_size is not None else 256

    def _preprocess_images(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Ensures images match vision encoder expected spatial size.
        NOTE: This does NOT apply dataset-specific normalization.
            If NitroGen training normalized in a particular way, mirror it here.
        """
        if frames.dim() != 5:
            raise NgPolicyError(f"frames must be [B,T,C,H,W], got {tuple(frames.shape)}")

        B, T, C, H, W = frames.shape
        target = self._vision_image_size()

        if H == target and W == target:
            return frames

        # Resize per-frame: (B*T,C,H,W) -> (B,T,C,target,target)
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
        """
        frames: [B,T,C,H,W]

        Returns:
          - default (return_raw=False):
              action_vec [B,D] OR discrete logits [B,A]
          - debug (return_raw=True):
              (action_vec/logits, raw_summary_dict)

        raw_summary_dict is JSON-safe and intentionally small.
        """
        if not isinstance(frames, torch.Tensor) or frames.dim() != 5:
            raise NgPolicyError(
                f"frames must be rank-5 torch.Tensor [B,T,C,H,W]. Got {type(frames)} {getattr(frames,'shape',None)}"
            )

        B, T = int(frames.shape[0]), int(frames.shape[1])
        if T <= 0:
            raise NgPolicyError("frames T must be > 0.")
        if not (0 <= take_step < self.action_horizon):
            raise NgPolicyError(f"take_step must be in [0, {self.action_horizon-1}] but got {take_step}")

        device = frames.device

        # Cap frames to fit max_seq_len (keep the most recent frames)
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
            data["game_ids"] = torch.full((B,), int(gid), dtype=torch.long, device=device)

        # Deterministic sampling (NitroGen.get_action uses torch.randn internally)
        if seed is not None:
            with torch.random.fork_rng(devices=[device]):
                torch.manual_seed(int(seed))
                out = self.ng.get_action(data)
        else:
            out = self.ng.get_action(data)

        if not isinstance(out, dict) or "action_tensor" not in out:
            raise NgPolicyError(
                f"Unexpected NitroGen.get_action output: keys={list(out.keys()) if isinstance(out, dict) else type(out)}"
            )

        actions = out["action_tensor"]  # [B, H, D]
        if actions.dim() != 3 or actions.shape[2] != self.action_dim:
            raise NgPolicyError(
                f"Unexpected action_tensor shape: {tuple(actions.shape)}; expected [B, action_horizon, action_dim]"
            )

        action_vec = actions[:, take_step, :]  # [B, D]

        # Compute the “primary” output this wrapper returns
        primary: Any
        if return_continuous or self.discrete is None:
            primary = action_vec
        else:
            primary = self.discrete(action_vec)

        if not return_raw:
            return primary

        # JSON-safe raw summary for debugging/parsing
        raw_summary: dict[str, Any] = {
            "take_step": int(take_step),
            "num_frames_used": int(num_frames),
            "tokens_per_image": int(self.tokens_per_image),
            "max_seq_len": int(self.max_seq_len),
            "action_horizon": int(self.action_horizon),
            "action_dim": int(self.action_dim),
            "data": {
                # include the exact token/mask tensors you’re building
                "vl_token_ids": _summarize_tensor(vl_token_ids, max_items=raw_max_items),
                "sa_token_ids": _summarize_tensor(sa_token_ids, max_items=raw_max_items),
                "dropped_images": _summarize_tensor(dropped_images, max_items=raw_max_items),
                "vl_attn_mask": _summarize_tensor(vl_attn_mask, max_items=raw_max_items),
                "game_ids": _summarize_tensor(data.get("game_ids"), max_items=raw_max_items),
                "images": _summarize_tensor(images, max_items=raw_max_items),
            },
            "out": _summarize_tensor(out, max_items=raw_max_items),
            "action_vec": _summarize_tensor(action_vec, max_items=raw_max_items),
            "primary": _summarize_tensor(primary, max_items=raw_max_items),
        }

        return primary, raw_summary