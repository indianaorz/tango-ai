import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config, DiTConfig, SelfAttentionTransformerConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 1024
SOUTH_IDX = 14
DATASET_DIR = "data/dataset_cached"

# -----------------------------------------------------------------------------
# 1. Dataset: Loads ONE real file, but overwrites labels to SOUTH
# -----------------------------------------------------------------------------
class SingleRealFrameDataset(Dataset):
    def __init__(self):
        # Find a real file
        files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.pt")))
        if not files:
            raise FileNotFoundError(f"No .pt files in {DATASET_DIR}")
        
        self.filepath = files[0]
        print(f"📄 Training on real file: {os.path.basename(self.filepath)}")
        
        # Load it
        data = torch.load(self.filepath, map_location="cpu", weights_only=True)
        
        # Take just the first frame
        # Data is [0-255]. We need to normalize it EXACTLY like the viewer/train script
        raw_frame = data["frames"][100].float() / 255.0 # [0, 1]
        self.frame_norm = (raw_frame - 0.5) / 0.5     # [-1, 1]
        
        # Create FAKE label (Always South)
        self.action = torch.zeros(25)
        self.action[4 + SOUTH_IDX] = 1.0 

    def __len__(self): return 100 # Repeat it 100 times

    def __getitem__(self, idx):
        return {
            "frames": self.frame_norm, # [3, H, W]
            "j_left": self.action[0:2].unsqueeze(0),
            "j_right": self.action[2:4].unsqueeze(0),
            "buttons": self.action[4:].unsqueeze(0),
            "dropped_frames": torch.zeros(1, dtype=torch.bool),
            "game": "bn6"
        }

# -----------------------------------------------------------------------------
# 2. Main Loop
# -----------------------------------------------------------------------------
def main():
    print("🧪 Starting REAL DATA OVERFIT")
    
    # 1. Setup Model
    model_cfg = NitroGen_Config(
        hidden_size=HIDDEN,
        diffusion_model_cfg=DiTConfig(
            hidden_size=HIDDEN, num_attention_heads=16, attention_head_dim=64, num_layers=12,
            output_dim=HIDDEN, action_dim=25,
        ),
        vl_self_attention_cfg=SelfAttentionTransformerConfig(
            hidden_size=HIDDEN, num_attention_heads=16, attention_head_dim=64, num_layers=4,
        ),
        action_dim=25, action_horizon=1, max_seq_len=1024,
        vision_encoder_name="google/siglip-large-patch16-256",
        vision_hidden_size=HIDDEN, tune_vision_tower=True,
    )
    tokenizer_cfg = NitrogenTokenizerConfig(training=True, action_horizon=1, max_action_dim=25, max_seq_len=1024)
    tokenizer = NitrogenTokenizer(tokenizer_cfg)
    
    model = NitroGen(config=model_cfg).to(DEVICE)
    
    # Load Weights
    if os.path.exists("weights/ng.pt"):
        print("   Loading weights/ng.pt...")
        state_dict = torch.load("weights/ng.pt", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataset = SingleRealFrameDataset()
    loader = DataLoader(dataset, batch_size=4)
    
    model.train()
    
    # 2. Train Loop
    print("\n👇 Training...")
    for step in range(100):
        batch = next(iter(loader))
        
        # Cheat Code: Freeze Noise so we can reach 0 loss
        torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42)

        # Tokenize
        tokenizer.train()
        samples = []
        for i in range(batch["frames"].shape[0]):
            sample_input = {k: v[i].unsqueeze(0) for k,v in batch.items() if k!="game"}
            samples.append(tokenizer.encode(sample_input))
        
        # Collate
        model_input = {}
        for k in samples[0].keys():
            vals = [s[k] for s in samples]
            if len(vals) > 0 and not isinstance(vals[0], torch.Tensor):
                vals = [torch.from_numpy(v) for v in vals]
            t = torch.stack(vals).to(DEVICE)
            
            if k in ["sa_token_ids", "input_ids"]: t = t.view(t.shape[0], -1)
            elif k in ["dropped_frames", "dropped_images"]: 
                if t.ndim == 3: t = t.squeeze(-1)
            model_input[k] = t

        # Forward
        outputs = model(model_input)
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    print(f"Final Loss: {loss.item():.6f}")
    
    # 3. Save
    if loss.item() < 0.1:
        print("✅ SUCCESS! Saving 'overfit.pt'...")
        torch.save({
            "model": model.state_dict(),
            "step": 100,
            "ckpt_config": {
                "model_cfg": model_cfg.model_dump(),
                "tokenizer_cfg": tokenizer_cfg.model_dump()
            }
        }, "checkpoints/overfit.pt")
        
        print(f"\n📢 TEST INSTRUCTION:")
        print(f"1. Open Viewer.")
        print(f"2. Go to file: {os.path.basename(dataset.filepath)}")
        print(f"3. Go to Frame 0.")
        print(f"4. Click Inference. It MUST predict SOUTH.")

if __name__ == "__main__":
    main()