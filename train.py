import os
import bisect
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- Import Nitrogen ---
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config, DiTConfig, SelfAttentionTransformerConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,       
    "lr": 1e-4,             
    "epochs": 10,
    "save_every": 5000,
    "dataset_dir": "data/dataset_cached", # Points to the pre-cached tensors
    "resolution": (256, 256), 
    "log_dir": "logs/tango_cached"
}

# -----------------------------------------------------------------------------
# Cached Dataset (Loads pre-processed .pt files)
# -----------------------------------------------------------------------------

class CachedTangoDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.files = sorted(list(self.root.glob("*.pt")))
        
        if not self.files:
            print(f"❌ No .pt files found in {self.root}. Did you run precache_dataset.py?")
            exit(1)

        self.cumulative_sizes = []
        self.file_lengths = []
        
        total_frames = 0
        print(f"🔍 Scanning cached dataset at {self.root}...")
        
        # Quick scan to build the index
        for pt_file in tqdm(self.files, desc="Indexing"):
            try:
                # We load just to check the size. 
                # This is fast because we aren't decoding pixels, just reading tensor metadata.
                data = torch.load(pt_file, weights_only=True)
                n = data["frames"].shape[0]
                
                total_frames += n
                self.cumulative_sizes.append(total_frames)
                self.file_lengths.append(n)
            except Exception as e:
                print(f"⚠️ Error reading {pt_file}: {e}")

        print(f"✅ Indexed {len(self.files)} files, {total_frames:,} total frames.")

        # Per-worker cache to avoid re-loading the same file repeatedly
        self.cache_idx = -1
        self.cache_data = None

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, global_idx):
        # 1. Binary search to find which file this frame lives in
        file_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        
        if file_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_sizes[file_idx - 1]

        # 2. Load file if not currently in memory
        if self.cache_idx != file_idx:
            # Add mmap=True (Available in PyTorch 2.1+)
            # This keeps the file on disk and only loads the specific frames you need
            try:
                self.cache_data = torch.load(self.files[file_idx], weights_only=True, mmap=True)
            except TypeError:
                # Fallback for older PyTorch
                self.cache_data = torch.load(self.files[file_idx], weights_only=True)
            
            self.cache_idx = file_idx
        
        # 3. Retrieve Data
        # Frame: uint8 [0-255] -> float [0.0-1.0]
        # Data is stored as [T, 3, H, W] in the .pt file
        frame_uint8 = self.cache_data["frames"][local_idx] 
        frames_tensor = frame_uint8.float().div_(255.0)

        # Action: float32 [25]
        action_vec = self.cache_data["actions"][local_idx]
        
        # Split action vector
        j_left = action_vec[0:2]   # [2]
        j_right = action_vec[2:4]  # [2]
        buttons = action_vec[4:]   # [21]

        return {
            "frames": frames_tensor,          # [3, H, W]
            "j_left": j_left.unsqueeze(0),    # [1, 2]
            "j_right": j_right.unsqueeze(0),  # [1, 2]
            "buttons": buttons.unsqueeze(0),  # [1, 21]
            "dropped_frames": torch.zeros(1, dtype=torch.bool),
            "game": "bn6"
        }

# -----------------------------------------------------------------------------
# Collation Utilities
# -----------------------------------------------------------------------------

def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.number))

def _tensorize_numeric(v):
    if torch.is_tensor(v): return v
    if isinstance(v, np.ndarray): return torch.from_numpy(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 0: return torch.tensor([], dtype=torch.float32)
        if all(_is_num(x) for x in v):
            return torch.tensor(v, dtype=torch.float32)
        if all(isinstance(x, (list, tuple, np.ndarray)) for x in v):
            if all(all(_is_num(y) for y in x) for x in v):
                return torch.tensor(v, dtype=torch.float32)
    return v

def _collate_values(key: str, vals: list):
    first = vals[0]
    
    # Handle Tensors
    if torch.is_tensor(first):
        if first.ndim == 0: return torch.stack(vals, dim=0)
        else: return torch.cat(vals, dim=0)
    
    # Handle Numeric Lists -> Tensors
    t_first = _tensorize_numeric(first)
    if torch.is_tensor(t_first):
        t_vals = [_tensorize_numeric(v) for v in vals]
        if not all(torch.is_tensor(tv) for tv in t_vals): return vals
        
        # Special handling for actions to ensure [B, H, A] shape
        if key in ("actions", "action", "action_values"):
            fixed = []
            for tv in t_vals:
                tv = tv.to(dtype=torch.float32)
                if tv.ndim == 1: tv = tv.unsqueeze(0)
                fixed.append(tv)
            return torch.stack(fixed, dim=0)

        # Standard stacking
        return torch.stack([tv if tv.ndim > 0 else tv.unsqueeze(0) for tv in t_vals], dim=0)
    
    if isinstance(first, (int, float, np.number)):
        return torch.tensor(vals, dtype=torch.float32)
    
    return vals

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    print("🚀 Initializing Optimized Training (Cached + Mixed Precision)...")
    
    writer = SummaryWriter(CONFIG["log_dir"])
    
    # --- Model Config (Fixed Dimensions) ---
    MAX_SEQ_LEN = 1024
    ACTION_HORIZON = 1

    HIDDEN = 1024
    HEAD_DIM = 128
    NUM_HEADS = HIDDEN // HEAD_DIM  # 12

    model_cfg = NitroGen_Config(
        # 1. Force top-level config to 768
        hidden_size=HIDDEN,
        
        # 2. Configure DiT with CORRECT output dimension
        diffusion_model_cfg=DiTConfig(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            num_layers=12,
            output_dim=HIDDEN,  # <--- CRITICAL FIX
            action_dim=25,
        ),
        
        # 3. Configure Attention
        vl_self_attention_cfg=SelfAttentionTransformerConfig(
            hidden_size=HIDDEN,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            num_layers=4,
        ),
        
        action_dim=25,
        action_horizon=ACTION_HORIZON,
        max_seq_len=MAX_SEQ_LEN,
        vision_encoder_name="google/siglip-large-patch16-256",  # <--- CHANGE THIS!
        vision_hidden_size=HIDDEN, 
        tune_vision_tower=True, # Fine-tune vision? (Optional, set False for speed)
    )
    
    tokenizer_cfg = NitrogenTokenizerConfig(
        training=True,
        action_horizon=ACTION_HORIZON,
        max_action_dim=25,
        max_seq_len=MAX_SEQ_LEN,
        old_layout=False 
    )
    
    tokenizer = NitrogenTokenizer(tokenizer_cfg)
    model = NitroGen(config=model_cfg).to(CONFIG["device"])

    # Load your local weights
    ckpt_path = "weights/ng.pt"  # Or "checkpoints/ng.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading local pretrained weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # If it's a full checkpoint with "model" key, use state_dict["model"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)  # strict=False ignores mismatches
    else:
        print("No pretrained weights found - starting from scratch.")
    
    # Optional: Compile model for speed (Linux/Ampere+ GPUs)
    # model = torch.compile(model) 
    
    print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- Dataset & Loader ---
    dataset = CachedTangoDataset(CONFIG["dataset_dir"])
    
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True,
        num_workers=0,      # High workers is safe now (lightweight dataset)
        pin_memory=True,
        persistent_workers=False 
    )

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    os.makedirs("checkpoints", exist_ok=True)
    
    scaler = torch.cuda.amp.GradScaler() # For mixed precision
    
    
    # --- RESUME LOGIC ---
    resume_path = "checkpoints/step_10000.pt"  # <--- POINT TO YOUR LAST SUCCESSFUL SAVE
    global_step = 0

    if os.path.exists(resume_path):
        print(f"🔄 Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location="cpu") # Load to CPU first to save GPU RAM
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
        print(f"✅ Resumed at step {global_step}")
    else:
        print("🆕 Starting fresh training")

    model.train()
    
    print(f"🔥 Starting Training on {len(dataset)} frames...")
    
    for epoch in range(CONFIG["epochs"]):
        print(f"--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        
        pbar = tqdm(loader)
        for batch in pbar:
            tokenizer.train() 
            
            # --- 1. Tokenize Batch ---
            encoded_samples = []
            curr_bs = batch["frames"].shape[0]

            try:
                for i in range(curr_bs):
                    single_sample = {
                        "frames": batch["frames"][i].unsqueeze(0), 
                        "j_left": batch["j_left"][i].unsqueeze(0),
                        "j_right": batch["j_right"][i].unsqueeze(0),
                        "buttons": batch["buttons"][i].unsqueeze(0),
                        "dropped_frames": batch["dropped_frames"][i].unsqueeze(0),
                        "game": batch["game"][i]
                    }
                    encoded = tokenizer.encode(single_sample)
                    encoded_samples.append(encoded)
            except ValueError as e:
                print(f"⚠️ Tokenizer Error: {e}")
                continue

            if not encoded_samples: continue

           # --- 2. Collate ---
            model_input = {}
            keys = encoded_samples[0].keys()

            for key in keys:
                raw_values = [sample[key] for sample in encoded_samples]
                batched = _collate_values(key, raw_values)

                if torch.is_tensor(batched):
                    batched = batched.to(CONFIG["device"], non_blocking=True)
                    
                    # Fix shapes
                    if key in ["sa_token_ids", "sa_mask", "input_ids", "attention_mask"]:
                        # Ensure [B, Seq]
                        if batched.ndim == 1: batched = batched.unsqueeze(1)
                        elif batched.ndim > 2: batched = batched.view(curr_bs, -1)
                        
                    elif key in ["pixel_values", "frames", "images"]: # Added "images" just in case
                        # Ensure [B, T, C, H, W]
                        # If we have [B, C, H, W], unsqueeze dim 1 to make it [B, 1, C, H, W]
                        if batched.ndim == 4: 
                            batched = batched.unsqueeze(1)

                model_input[key] = batched

            # --- 3. Forward/Backward (Mixed Precision) ---
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(model_input)
                loss = outputs["loss"]
            
            # Scale loss for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # --- 4. Logging ---
            global_step += 1
            loss_val = loss.item()
            pbar.set_description(f"Loss: {loss_val:.4f}")
            writer.add_scalar("Training/Loss", loss_val, global_step)
            
            if global_step % CONFIG["save_every"] == 0:
                save_path = f"checkpoints/step_{global_step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "ckpt_config": {
                        "model_cfg": model_cfg.model_dump(),
                        "tokenizer_cfg": tokenizer_cfg.model_dump()
                    }
                }, f"checkpoints/step_{global_step}.pt")
                
                # --- AUTO-CLEANUP: Keep only last 3 checkpoints ---
                # Find all step_*.pt files
                all_ckpts = sorted(Path("checkpoints").glob("step_*.pt"), 
                                   key=lambda f: int(f.stem.split('_')[1]))
                
                # If we have more than 3, delete the oldest ones
                if len(all_ckpts) > 3:
                    for old_ckpt in all_ckpts[:-3]:
                        print(f"🗑️ Deleting old checkpoint: {old_ckpt}")
                        old_ckpt.unlink() # Delete file

    print("✅ Training Complete.")
    writer.close()

if __name__ == "__main__":
    main()