import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .dataset import PlanningDataset, collate_fn
from .model import PlanningCritic, PlanningConfig

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to strategy_v2.jsonl")
    ap.add_argument("--save", default="checkpoints/planning_critic.pt")
    ap.add_argument("--tb_dir", default="runs/planning_critic", help="TensorBoard log directory")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    # Load Data
    ds = PlanningDataset(args.data, shuffle_hand=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # Init Model
    model = PlanningCritic(PlanningConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.HuberLoss(delta=1.0)
    
    # Init TensorBoard
    writer = SummaryWriter(log_dir=args.tb_dir)
    
    print(f"Training on {len(ds)} samples. Logging to: {args.tb_dir}")
    global_step = 0
    
    for ep in range(args.epochs):
        t0 = time.time()
        total_loss = 0
        n = 0
        
        for batch in loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            opt.zero_grad()
            
            pred = model(batch)
            loss = loss_fn(pred, batch['target'])
            
            loss.backward()
            opt.step()
            
            # --- Logging ---
            loss_val = loss.item()
            total_loss += loss_val * len(pred)
            n += len(pred)
            global_step += 1
            
            writer.add_scalar("Train/HuberLoss", loss_val, global_step)
            writer.add_scalar("Train/LR", opt.param_groups[0]['lr'], global_step)
            
        avg_loss = total_loss / max(1, n)
        dt = time.time() - t0
        rate = n / dt if dt > 0 else 0
        
        writer.add_scalar("Epoch/AvgLoss", avg_loss, ep + 1)
        print(f"Epoch {ep+1}/{args.epochs}: Loss={avg_loss:.4f} ({rate:.0f} samp/s)")
        
        # Save every epoch (overwriting last)
        torch.save(model.state_dict(), args.save)

    writer.close()
    print(f"Done. Saved to {args.save}")

if __name__ == "__main__":
    train()