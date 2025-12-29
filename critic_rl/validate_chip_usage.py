import argparse
import torch
import sys
from critic_rl.dataset import InMemoryCriticRLTDDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--cache_dir", type=str, required=True)
    # Default params matching your training
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--folder_len", type=int, default=30)
    args = ap.parse_args()

    print(f"--- Loading Cache: {args.cache_dir} ---")
    
    # Load Validation set
    try:
        ds = InMemoryCriticRLTDDataset(
            args.dataset_dir, 
            cache_dir=args.cache_dir, 
            stride=args.stride, 
            folder_len=args.folder_len, 
            seq_len=args.seq_len, 
            require_cust_gt0=True, 
            split="val",
            val_ratio=0.1
        )
    except Exception as e:
        print(f"FATAL: {e}")
        return

    # Extract relevant tensors
    # Shapes: [N, T]
    last_used = ds.x['last_used_id_p']
    on_deck = ds.x['current_chip_p']
    valid = ds.valid
    
    # We need to look for events where last_used > 0
    # Since dataset is sliced into overlapping sequences, we can just iterate sequences
    # or flatten. Flattening is easier for counting, but we need T-1 context.
    
    print("\n--- Scanning for Chip Usage Events ---")
    
    found_events = 0
    matches = 0
    mismatches = 0
    
    # Check random samples of sequences
    num_seqs = last_used.shape[0]
    seq_len = last_used.shape[1]
    
    # Let's look at the first 500 sequences (or all if fewer)
    indices = range(min(num_seqs, 5000))
    
    print(f"Scanning {len(indices)} sequences for usage events...")
    print(f"{'Seq':<5} {'Frame':<5} | {'T-1 (On Deck)':<20} -> {'T (Just Used)':<20} | {'Status'}")
    print("-" * 80)

    for i in indices:
        # Get sequence [T]
        seq_used = last_used[i]
        seq_deck = on_deck[i]
        seq_valid = valid[i]
        
        # Look for usage (skip frame 0 because we need T-1)
        for t in range(1, seq_len):
            if not seq_valid[t]: continue
            
            used_id = seq_used[t].item()
            
            if used_id > 0: # A chip was used this frame
                found_events += 1
                
                # Look at previous frame's "On Deck"
                prev_deck_id = seq_deck[t-1].item()
                curr_deck_id = seq_deck[t].item() # usually changes to 0 or next chip
                
                # Ideally: used_id == prev_deck_id
                status = "OK"
                if used_id == prev_deck_id:
                    matches += 1
                else:
                    status = "MISMATCH (!)"
                    mismatches += 1
                
                # Print first 20 events
                if found_events <= 20:
                    print(f"{i:<5} {t:<5} | ID {prev_deck_id:<17} -> ID {used_id:<17} | {status}")

    print("-" * 80)
    print(f"Total Events Found: {found_events}")
    print(f"Perfect Matches:    {matches} ({matches/max(1,found_events)*100:.1f}%)")
    print(f"Mismatches:         {mismatches}")
    
    if mismatches > 0:
        print("\nNOTE: Mismatches can happen if:")
        print("1. The 'stride' skipped the exact frame where the transition happened.")
        print("2. The game logic updated 'player_chip' slightly faster/slower than the usage flag.")
        print("3. Dataset stride is > 1 (You are using stride=8).")
        
    print("\nSince you use stride=8, we expect mostly MISMATCHES in this view because")
    print("Frame T and Frame T-1 in the dataset are actually 8 real frames apart.")
    print("The model still learns causality if 'On Deck' is consistent for the 8 frames prior.")

if __name__ == "__main__":
    main()