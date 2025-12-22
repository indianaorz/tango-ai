from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import torch

from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig


def _load_ckpt(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # support weights-only file
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt
    return {"model": ckpt}


def _pick_random_frame(pt_path: Path) -> Dict[str, Any]:
    data = torch.load(pt_path, map_location="cpu")
    T = int(data["frames"].shape[0])
    i = random.randrange(T)

    frame_uint8 = data["frames"][i]          # [3,H,W] uint8
    action = data["actions"][i].float()      # [25] float

    sample = {
        "frames": frame_uint8.float().div(255.0).unsqueeze(0),  # [1,3,H,W] float
        "j_left":  action[0:2].unsqueeze(0).unsqueeze(0),       # [1,1,2]
        "j_right": action[2:4].unsqueeze(0).unsqueeze(0),       # [1,1,2]
        "buttons": action[4:].unsqueeze(0).unsqueeze(0),        # [1,1,21]
        "dropped_frames": torch.zeros(1, 1, dtype=torch.bool),  # [1,1]
        "game": "bn6",
    }
    return sample


def main() -> None:
    torch.set_grad_enabled(False)

    ckpt_path = os.environ.get("CKPT", "checkpoints/step_5000.pt")
    dataset_dir = Path(os.environ.get("DATASET_DIR", "data/dataset_cached"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _load_ckpt(ckpt_path)

    # IMPORTANT: build model/tokenizer from checkpoint configs if present
    if "ckpt_config" in ckpt:
        model_cfg = NitroGen_Config(**ckpt["ckpt_config"]["model_cfg"])
        tok_cfg = NitrogenTokenizerConfig(**ckpt["ckpt_config"]["tokenizer_cfg"])
    else:
        raise RuntimeError("Checkpoint missing ckpt_config; cannot guarantee parity.")

    tokenizer = NitrogenTokenizer(tok_cfg)
    model = NitroGen(config=model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    pt_files = sorted(dataset_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {dataset_dir}")

    pt_path = random.choice(pt_files)
    sample = _pick_random_frame(pt_path)

    encoded = tokenizer.encode(sample)

    # batchify exactly like training: tensors to device, add time dim for frames if needed
    model_input: Dict[str, Any] = {}
    for k, v in encoded.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if k in ("pixel_values", "frames", "images") and v.ndim == 4:
                v = v.unsqueeze(1)  # [B,1,C,H,W]
        model_input[k] = v

    out = model(model_input)
    loss = out["loss"]
    if torch.is_tensor(loss) and loss.ndim != 0:
        loss = loss.mean()

    print(f"ckpt={ckpt_path}")
    print(f"sample_pt={pt_path.name}")
    print(f"loss={float(loss.item()):.6f}")
    print("out keys:", list(out.keys()))

    # If NitroGen returns predicted actions/logits, print their stats too
    for name in ("actions", "action", "pred_actions", "action_logits", "buttons_logits"):
        if name in out and torch.is_tensor(out[name]):
            t = out[name].detach().float().cpu()
            print(f"{name}: shape={tuple(t.shape)} min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}")


if __name__ == "__main__":
    main()
