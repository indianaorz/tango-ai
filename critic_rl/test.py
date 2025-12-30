import torch
from pathlib import Path

# Pick the same combat file
file_path = list(Path("data/cache_rl/s2_s256_2").glob("*.pt"))[0]
data = torch.load(file_path, map_location="cpu")

# Scan sequences for a hit
found = False
for s_idx in range(data['r'].shape[0]):
    for t_idx in range(data['r'].shape[1]):
        reward = data['r'][s_idx, t_idx]
        if abs(reward) > 0.1:
            # Found a hit! Print the transition
            hp_p = data['x']['scalars'][s_idx, t_idx, 0] * 2500
            hp_e = data['x']['scalars'][s_idx, t_idx, 1] * 2500
            
            # Since stride is 2, the reward corresponds to the state 2 frames later
            # But let's just look at the sign for now
            print(f"HIT FOUND at Seq {s_idx}, Frame {t_idx}")
            print(f"Current State: P_HP={hp_p:.0f} E_HP={hp_e:.0f}")
            print(f"Target Reward for this state: {reward:.2f}")
            
            if reward > 0:
                print(">>> DATASET SAYS: Positive Reward (Good)")
            else:
                print(">>> DATASET SAYS: Negative Reward (Bad)")
            
            found = True
            break
    if found: break

if not found:
    print("No hits found in this file. Try another file or check if rewards are all zeros.")