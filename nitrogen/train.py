# nitrogen/train.py
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nitrogen.mm_tokenizers import NitrogenTokenizer

# Shared utils from your repo (must already exist, as in your current training script)
import train_utils as U

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
BASE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 96,
    "lr": 1e-4,
    "epochs": 1000,
    "save_every": 1000,
    "num_workers": 8,
    "max_keep_ckpts": 3,
    "base_ckpt": "weights/ng.pt",
    "prefetch_factor": 8,
}

DEFAULT_CACHE_DIR = "data/nitrogen_battle_cache_bellman"
DEFAULT_CKPT_DIR = "checkpoints/nitrogen_battle_cache_bellman"
DEFAULT_LOG_DIR = "logs/nitrogen_battle_cache_bellman"
MANIFEST_NAME = "manifest.json"


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _load_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    mp = cache_dir / MANIFEST_NAME
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _estimate_total_batches(cache_dir: Path, batch_size: int) -> Optional[int]:
    m = _load_manifest(cache_dir)
    if not m:
        return None
    total_samples = int(m.get("total_samples", 0) or 0)
    if total_samples <= 0:
        return None
    return int(math.ceil(total_samples / max(1, int(batch_size))))


def _suggest_norm_factor_from_manifest(manifest: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Reads precache-written manifest['values_summary'] and returns a good default norm_factor.
    We prefer abs_p90 median (more stable) and fall back to abs_p95 median.
    """
    if not manifest:
        return None
    vs = manifest.get("values_summary") or {}
    v = vs.get("abs_p90_median_across_files", None)
    if v is None:
        v = vs.get("abs_p95_median_across_files", None)
    try:
        if v is None:
            return None
        fv = float(v)
        if not math.isfinite(fv) or fv <= 0.0:
            return None
        return fv
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Weighting (Bellman values -> sample weights)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueWeightCfg:
    """
    Turns per-sample value (Bellman return) into a positive weight.

        w = 1 + scale * tanh( clamp(value / norm_factor, [-tanh_clip, tanh_clip]) )

    Interpretation:
      - big positive value => weight up
      - big negative value => weight down
      - near 0 => ~1
    """
    norm_factor: float = 100.0
    tanh_clip: float = 5.0
    scale: float = 2.0
    min_weight: float = 0.1
    max_weight: float = 10.0
    positive_only_weights: bool = False  # if True: negative values do not reduce weights


def compute_value_weights(values: torch.Tensor, cfg: ValueWeightCfg) -> torch.Tensor:
    v = values.to(torch.float32)
    if cfg.positive_only_weights:
        v = torch.clamp(v, min=0.0)

    x = v / float(cfg.norm_factor)
    x = torch.clamp(x, -float(cfg.tanh_clip), float(cfg.tanh_clip))
    a = torch.tanh(x)
    w = 1.0 + float(cfg.scale) * a
    return torch.clamp(w, float(cfg.min_weight), float(cfg.max_weight))


def weighted_mean(loss_per_sample: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(loss_per_sample.dtype).clamp_min(1e-8)
    return (w * loss_per_sample).sum() / w.sum()


# -----------------------------------------------------------------------------
# Dataset: streams cached battle shards (Bellman cache)
# -----------------------------------------------------------------------------
class BattleBellmanCacheDataset(IterableDataset):
    """
    Reads cache files from nitrogen/precache.py (battle-only + bellman values).
    Expects:
      - frames:  [N, 3, H, W] uint8
      - actions: [N, AH, ACTION_DIM] float32
      - values:  [N] float32   (Bellman HP-return for that battle frame)
    """

    def __init__(
        self,
        *,
        cache_dir: str,
        vision_horizon: int,
        action_horizon: int,
        seed: int = 42,
        shuffle_files: bool = True,
        shuffle_within_file: bool = True,
        game: str = "bn6",
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.vision_horizon = int(max(1, vision_horizon))
        self.action_horizon = int(max(1, action_horizon))
        self.seed = int(seed)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_within_file = bool(shuffle_within_file)
        self.game = str(game)

        self._files = sorted(self.cache_dir.glob("*.pt"))
        if not self._files:
            raise RuntimeError(f"No .pt cache files found in: {self.cache_dir}")

        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _iter_files_for_worker(self) -> List[Path]:
        files = list(self._files)
        wi = get_worker_info()
        if wi is not None:
            files = files[wi.id :: wi.num_workers]

        if self.shuffle_files:
            rng = random.Random((self.seed * 1000003) ^ (self._epoch * 9176) ^ 0xA5A5)
            rng.shuffle(files)

        return files

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        files = self._iter_files_for_worker()

        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        rng = random.Random((self.seed * 1000003) ^ (self._epoch * 9176) ^ (wid * 1315423911))

        for fp in files:
            payload = torch.load(fp, map_location="cpu")

            frames: torch.Tensor = payload["frames"]  # [N, 3, H, W]
            actions: torch.Tensor = payload["actions"]  # [N, AH, ACTION_DIM]
            values: torch.Tensor = payload.get("values", None)

            if values is None:
                raise RuntimeError(f"Cache file missing 'values': {fp}")

            n = int(frames.shape[0])
            if n == 0:
                continue

            cache_ah = int(actions.shape[1])
            if cache_ah != self.action_horizon:
                raise RuntimeError(
                    f"Action horizon mismatch in cache file {fp.name}: cache has {cache_ah}, "
                    f"tokenizer expects {self.action_horizon}. "
                    f"Fix: recache with --action_horizon={self.action_horizon}."
                )

            order = list(range(n))
            if self.shuffle_within_file:
                rng.shuffle(order)

            for i in order:
                f = frames[i]  # [3, H, W]
                a = actions[i]  # [AH, ACTION_DIM]
                v = values[i]  # scalar

                j_left = a[:, 0:2].to(torch.float32)
                j_right = a[:, 2:4].to(torch.float32)
                buttons = a[:, 4:].to(torch.float32)

                frames_seq = f.unsqueeze(0).repeat(self.vision_horizon, 1, 1, 1)  # [V, 3, H, W]
                dropped = torch.zeros((self.vision_horizon,), dtype=torch.float32)

                yield {
                    "frames": frames_seq,
                    "j_left": j_left,
                    "j_right": j_right,
                    "buttons": buttons,
                    "dropped_frames": dropped,
                    "game": self.game,
                    "value": v.to(torch.float32),
                }


# -----------------------------------------------------------------------------
# Model output helpers
# -----------------------------------------------------------------------------
def _get_loss_per_sample(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, dict) and "loss_per_sample" in output:
        return output["loss_per_sample"]
    return None


def _get_scalar_loss(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, dict) and "loss" in output:
        return output["loss"]
    return None


def _save_ckpt(
    *,
    save_path: Path,
    model: torch.nn.Module,
    loaded: Any,
    cache_dir: Path,
    action_horizon: int,
    vision_horizon: int,
    weight_cfg: ValueWeightCfg,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    torch.save(
        {
            "model": model.state_dict(),
            "ckpt_config": U.to_dict(loaded.ckpt_config),
            "tokenizer_cfg": U.to_dict(loaded.tokenizer_cfg),
            "train_meta": {
                "cache_dir": str(cache_dir),
                "action_horizon": int(action_horizon),
                "vision_horizon": int(vision_horizon),
                "value_weight_cfg": {
                    "norm_factor": weight_cfg.norm_factor,
                    "tanh_clip": weight_cfg.tanh_clip,
                    "scale": weight_cfg.scale,
                    "min_weight": weight_cfg.min_weight,
                    "max_weight": weight_cfg.max_weight,
                    "positive_only_weights": bool(weight_cfg.positive_only_weights),
                },
            },
        },
        tmp,
    )
    tmp.replace(save_path)


# -----------------------------------------------------------------------------
# Collate/tokenize in workers (Windows-safe: top-level picklable class)
# -----------------------------------------------------------------------------
def _stack_if_possible(xs: List[Any]) -> Any:
    if not xs:
        return xs
    x0 = xs[0]
    if torch.is_tensor(x0):
        try:
            return torch.stack(xs, dim=0)
        except Exception:
            return xs
    if isinstance(x0, (int, float, bool)):
        return torch.tensor(xs)
    return xs


class CollateAndEncode:
    """
    Picklable collate_fn for Windows spawn.

    We pass tokenizer_cfg (usually a pydantic/dataclass config) which is picklable.
    Each worker process gets its own instance (and we lazily create the tokenizer).
    """

    def __init__(self, tokenizer_cfg: Any) -> None:
        self._tokenizer_cfg = tokenizer_cfg
        self._tok: Optional[NitrogenTokenizer] = None

    def _get_tok(self) -> NitrogenTokenizer:
        if self._tok is None:
            tok = NitrogenTokenizer(self._tokenizer_cfg)
            tok.train()
            self._tok = tok
        return self._tok

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tok = self._get_tok()

        encoded: List[Dict[str, Any]] = []
        values: List[torch.Tensor] = []
        games: List[str] = []

        for s in samples:
            # tokenizer.encode mutates input, so build a fresh dict
            d = {
                "frames": s["frames"].unsqueeze(0) if s["frames"].ndim == 4 else s["frames"],
                "j_left": s["j_left"].unsqueeze(0) if s["j_left"].ndim == 2 else s["j_left"],
                "j_right": s["j_right"].unsqueeze(0) if s["j_right"].ndim == 2 else s["j_right"],
                "buttons": s["buttons"].unsqueeze(0) if s["buttons"].ndim == 2 else s["buttons"],
                "dropped_frames": s["dropped_frames"].unsqueeze(0) if s["dropped_frames"].ndim == 1 else s["dropped_frames"],
                "game": s["game"],
            }
            enc = tok.encode(d)
            if enc:
                encoded.append(enc)
                values.append(s["value"] if torch.is_tensor(s["value"]) else torch.as_tensor(s["value"], dtype=torch.float32))
                games.append(s["game"])

        if not encoded:
            return {"encoded": [], "value": torch.empty((0,), dtype=torch.float32), "game": []}

        v = _stack_if_possible(values).to(torch.float32)
        return {"encoded": encoded, "value": v, "game": games}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nitrogen battle-only training with Bellman-cache value-weighted loss."
    )
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--resume", type=str, default="", help="Path to resume checkpoint (optional).")

    # Weighting knobs (Bellman cache)
    parser.add_argument("--norm_factor", type=float, default=0.0, help="0 => auto from manifest (abs_p90 median) if available.")
    parser.add_argument("--tanh_clip", type=float, default=5.0)
    parser.add_argument("--weight_scale", type=float, default=2.0)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--max_weight", type=float, default=10.0)
    parser.add_argument("--positive_only_weights", action="store_true", help="Do not down-weight negative values (clamp value>=0).")

    # Train knobs
    parser.add_argument("--batch_size", type=int, default=BASE_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=BASE_CONFIG["lr"])
    parser.add_argument("--epochs", type=int, default=BASE_CONFIG["epochs"])
    parser.add_argument("--save_every", type=int, default=BASE_CONFIG["save_every"])
    parser.add_argument("--num_workers", type=int, default=BASE_CONFIG["num_workers"])
    parser.add_argument("--max_keep_ckpts", type=int, default=BASE_CONFIG["max_keep_ckpts"])
    parser.add_argument("--base_ckpt", type=str, default=BASE_CONFIG["base_ckpt"])
    parser.add_argument("--device", type=str, default=BASE_CONFIG["device"])

    # Perf knobs
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul (Ampere+), usually faster.")
    parser.add_argument("--compile", action="store_true", help="torch.compile(model) (PyTorch 2.x). Helps if GPU-bound.")

    # Shuffle controls
    parser.add_argument("--no_shuffle_files", action="store_true")
    parser.add_argument("--no_shuffle_within_file", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("🚀 Nitrogen battle training (bellman-cache value-weighted)")
    print(f"   cache_dir: {cache_dir}")
    print(f"   ckpt_dir:  {ckpt_dir}")
    print(f"   log_dir:   {log_dir}")
    print(f"   device:    {device}")
    if args.tf32:
        print("   tf32:      enabled")
    if args.compile:
        print("   compile:   enabled")

    manifest = _load_manifest(cache_dir)
    total_batches = _estimate_total_batches(cache_dir, int(args.batch_size))

    # --- Determine norm_factor (auto if 0) ---
    if float(args.norm_factor) > 0.0:
        norm_factor = float(args.norm_factor)
        norm_src = "cli"
    else:
        auto_nf = _suggest_norm_factor_from_manifest(manifest)
        if auto_nf is not None:
            norm_factor = float(auto_nf)
            norm_src = "manifest(abs_p90 median)"
        else:
            norm_factor = 100.0
            norm_src = "fallback(100.0)"

    weight_cfg = ValueWeightCfg(
        norm_factor=norm_factor,
        tanh_clip=float(args.tanh_clip),
        scale=float(args.weight_scale),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
        positive_only_weights=bool(args.positive_only_weights),
    )

    # --- Load model (resume/auto/base) ---
    if args.resume:
        load_path = args.resume
        print(f"🔄 Resuming from explicit: {load_path}")
    else:
        existing = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else -1,
        )
        if existing:
            load_path = str(existing[-1])
            print(f"🔄 Auto-resuming from latest: {load_path}")
        else:
            load_path = args.base_ckpt
            print(f"🌱 Starting from base: {load_path}")

    loaded = U.load_ng_checkpoint_faithful(load_path, device)

    tokenizer = NitrogenTokenizer(loaded.tokenizer_cfg)
    tokenizer.train()

    action_horizon = int(getattr(tokenizer, "action_horizon", 0) or 0)
    if action_horizon <= 0:
        raise RuntimeError("Tokenizer missing a valid action_horizon; cannot train safely.")

    vision_horizon = int(getattr(tokenizer, "vision_horizon", 1) or 1)
    if vision_horizon <= 0:
        vision_horizon = 1

    print(f"   tokenizer horizons: action_horizon={action_horizon} vision_horizon={vision_horizon}")

    # --- Dataset / loader ---
    dataset = BattleBellmanCacheDataset(
        cache_dir=str(cache_dir),
        vision_horizon=vision_horizon,
        action_horizon=action_horizon,
        seed=int(args.seed),
        shuffle_files=not args.no_shuffle_files,
        shuffle_within_file=not args.no_shuffle_within_file,
        game="bn6",
    )

    # IMPORTANT: pass config, not a closure
    collate_fn = CollateAndEncode(loaded.tokenizer_cfg)

    dl_kwargs: Dict[str, Any] = dict(
        batch_size=int(args.batch_size),
        shuffle=False,  # IterableDataset should not use DataLoader shuffle
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=collate_fn,
    )
    if int(args.num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(BASE_CONFIG["prefetch_factor"])

    loader = DataLoader(dataset, **dl_kwargs)

    # --- Model / optim ---
    model = loaded.model
    model.train()

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ torch.compile failed, continuing without compile: {e}")

    # Optional: freeze vision tower
    for name, p in model.named_parameters():
        if "vision" in name or "siglip" in name:
            p.requires_grad = False

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    writer = SummaryWriter(str(log_dir))

    # step init
    step = 0
    try:
        p = Path(load_path)
        if "step" in p.name and "_" in p.stem:
            step = int(p.stem.split("_")[1])
    except Exception:
        step = 0

    if total_batches is not None:
        print(f"   epoch batches (from manifest): {total_batches:,}")
    else:
        print("   epoch batches: (unknown; manifest missing or empty)")

    print(
        f"   weights: norm_factor={weight_cfg.norm_factor} ({norm_src}) tanh_clip={weight_cfg.tanh_clip} "
        f"scale={weight_cfg.scale} clamp=[{weight_cfg.min_weight},{weight_cfg.max_weight}] "
        f"positive_only_weights={weight_cfg.positive_only_weights}"
    )
    print(f"🔥 Training Loop Start (continuing from step={step})")

    interrupted = False

    try:
        for epoch in range(int(args.epochs)):
            dataset.set_epoch(epoch)

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{int(args.epochs)}",
                total=total_batches,  # may be None; tqdm handles it
                dynamic_ncols=True,
            )

            for batch in pbar:
                step += 1

                encoded_list: List[Dict[str, Any]] = batch["encoded"]
                if not encoded_list:
                    continue

                values = batch["value"].to(torch.float32)  # [B']
                w = compute_value_weights(values, weight_cfg).to(device=device, dtype=torch.float32)  # [B']

                model_input = U.collate_encoded(encoded_list, device=device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    output = model(model_input)

                loss_ps = _get_loss_per_sample(output)
                if loss_ps is not None:
                    loss_ps = loss_ps.to(torch.float32)
                    weighted_loss = weighted_mean(loss_ps, w)
                    unweighted_loss = loss_ps.mean()
                else:
                    scalar = _get_scalar_loss(output)
                    if scalar is None:
                        raise RuntimeError(
                            "Model output has neither 'loss_per_sample' nor 'loss'. "
                            "Update the model to return loss_per_sample for weighted training."
                        )
                    weighted_loss = scalar
                    unweighted_loss = scalar

                weighted_loss.backward()
                optimizer.step()

                wl = float(weighted_loss.item())
                ul = float(unweighted_loss.item())
                w_mean = float(w.mean().item())
                w_min = float(w.min().item())
                w_max = float(w.max().item())

                writer.add_scalar("Train/LossWeighted", wl, step)
                writer.add_scalar("Train/LossUnweighted", ul, step)
                writer.add_scalar("Train/WeightMean", w_mean, step)
                writer.add_scalar("Train/WeightMin", w_min, step)
                writer.add_scalar("Train/WeightMax", w_max, step)

                pbar.set_postfix(
                    {
                        "wloss": f"{wl:.4f}",
                        "loss": f"{ul:.4f}",
                        "wμ": f"{w_mean:.2f}",
                        "wmin": f"{w_min:.2f}",
                        "wmax": f"{w_max:.2f}",
                    }
                )

                if step % int(args.save_every) == 0:
                    save_path = ckpt_dir / f"step_{step}.pt"
                    _save_ckpt(
                        save_path=save_path,
                        model=model,
                        loaded=loaded,
                        cache_dir=cache_dir,
                        action_horizon=action_horizon,
                        vision_horizon=vision_horizon,
                        weight_cfg=weight_cfg,
                    )
                    U.cleanup_old_checkpoints(ckpt_dir, int(args.max_keep_ckpts))

    except KeyboardInterrupt:
        interrupted = True
        print("\n🛑 Interrupted (Ctrl+C). Saving an interrupt checkpoint...")

    finally:
        if interrupted:
            try:
                save_path = ckpt_dir / f"interrupt_step_{step}_{_now_tag()}.pt"
                _save_ckpt(
                    save_path=save_path,
                    model=model,
                    loaded=loaded,
                    cache_dir=cache_dir,
                    action_horizon=action_horizon,
                    vision_horizon=vision_horizon,
                    weight_cfg=weight_cfg,
                )
                U.cleanup_old_checkpoints(ckpt_dir, int(args.max_keep_ckpts))
                print(f"💾 Saved: {save_path}")
            except Exception as e:
                print(f"⚠️ Failed to save interrupt checkpoint: {e}")

        try:
            writer.flush()
            writer.close()
        except Exception:
            pass

        if interrupted:
            raise SystemExit(1)

    print("✅ Done.")


if __name__ == "__main__":
    main()
