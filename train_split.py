import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nitrogen.mm_tokenizers import NitrogenTokenizer

# Import shared utils
import train_utils as U

# --- DEFAULTS ---
BASE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 96,
    "lr": 1e-4,
    "epochs": 50,
    "save_every": 5000,
    "num_workers": 16,
    "max_keep_ckpts": 3,
    "base_ckpt": "checkpoints_old/step_150000.pt", 
}

MASK_FILENAME = "chip_window_mask.png"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["battle", "plan"], help="Training mode")
    parser.add_argument("--resume", type=str, default="", help="Path to resume checkpoint")
    args = parser.parse_args()

    # --- SETUP PATHS ---
    mode = args.mode
    root_dir = Path("data")
    
    if mode == "battle":
        dataset_dir = root_dir / "battle_cache"
        ckpt_dir = Path("checkpoints/battle")
        log_dir = Path("logs/battle")
    else:
        dataset_dir = root_dir / "planning_cache"
        ckpt_dir = Path("checkpoints/planning")
        log_dir = Path("logs/planning")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting {mode.upper()} Training")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Checkpoints: {ckpt_dir}")

    # --- LOAD MODEL & RESUME LOGIC ---
    device = torch.device(BASE_CONFIG["device"])
    
    # 1. Determine Load Path
    if args.resume:
        load_path = args.resume
        print(f"🔄 Resuming from specific: {load_path}")
    else:
        # Auto-resume: Check for highest step in target directory
        existing = sorted(
            ckpt_dir.glob("step_*.pt"), 
            key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else -1
        )
        
        if existing:
            load_path = str(existing[-1])
            print(f"🔄 Auto-resuming from latest: {load_path}")
        else:
            load_path = BASE_CONFIG["base_ckpt"]
            print(f"🌱 Starting fresh from base: {load_path}")

    loaded = U.load_ng_checkpoint_faithful(load_path, device)
    
    tokenizer = NitrogenTokenizer(loaded.tokenizer_cfg)
    tokenizer.train()
    
    model = loaded.model
    model.train()

    # Freeze Vision Tower
    for name, p in model.named_parameters():
        if "vision" in name or "siglip" in name:
            p.requires_grad = False

    # --- DATASET & MASKING ---
    mask_path = None
    if mode == "plan":
        # Check current dir first, then data/assets
        if os.path.exists(MASK_FILENAME):
            mask_path = MASK_FILENAME
        elif os.path.exists(os.path.join("data", "assets", MASK_FILENAME)):
            mask_path = os.path.join("data", "assets", MASK_FILENAME)
        
        if mask_path:
            print(f"🎭 Masking enabled for Plan mode: {mask_path}")

    dataset = U.CachedSplitDataset(
        root_dir=str(dataset_dir),
        vision_horizon=int(getattr(tokenizer, "vision_horizon", 1)),
        balance_sampling=False, 
        active_ratio=0.7,
        press_threshold=0.5,
        base_seed=42,
        mask_path=mask_path # Only passes mask if found + plan mode
    )

    loader = DataLoader(
        dataset, 
        batch_size=BASE_CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=BASE_CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=BASE_CONFIG["lr"])
    writer = SummaryWriter(str(log_dir))

    # --- STEP INITIALIZATION ---
    step = 0
    try:
        # Extract step number from filename "step_150000.pt" -> 150000
        # If filename is "ng.pt", stays 0.
        if "step" in Path(load_path).name and "_" in Path(load_path).stem:
            step = int(Path(load_path).stem.split("_")[1])
    except: 
        pass

    print(f"🔥 Training Loop Start (Continuing from Step {step})")
    
    for epoch in range(BASE_CONFIG["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            step += 1
            
            # Prepare Batch
            bs = batch["frames"].shape[0]
            encoded_list = []
            
            for i in range(bs):
                sample = {
                    "frames": batch["frames"][i].unsqueeze(0),
                    "j_left": batch["j_left"][i].unsqueeze(0),
                    "j_right": batch["j_right"][i].unsqueeze(0),
                    "buttons": batch["buttons"][i].unsqueeze(0),
                    "dropped_frames": batch["dropped_frames"][i].unsqueeze(0),
                    "game": "bn6"
                }
                enc = tokenizer.encode(sample)
                if enc: encoded_list.append(enc)
            
            if not encoded_list: continue

            # Collate (Using robust version from utils)
            model_input = U.collate_encoded(encoded_list, device=device)

            # Step
            optimizer.zero_grad(set_to_none=True)
            # Use new PyTorch AMP syntax
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(model_input)
                loss = output["loss"] if isinstance(output, dict) else output[0]
            
            loss.backward()
            optimizer.step()

            # Log
            loss_val = loss.item()
            writer.add_scalar("Train/Loss", loss_val, step)
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            # Save
            if step % BASE_CONFIG["save_every"] == 0:
                save_path = ckpt_dir / f"step_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "ckpt_config": U.to_dict(loaded.ckpt_config),
                    "tokenizer_cfg": U.to_dict(loaded.tokenizer_cfg)
                }, save_path)
                U.cleanup_old_checkpoints(ckpt_dir, BASE_CONFIG["max_keep_ckpts"])

    print("✅ Done.")

if __name__ == "__main__":
    main()