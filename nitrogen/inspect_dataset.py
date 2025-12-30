import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Visualize Bellman signal health from metadata.")
    parser.add_argument("--cache_dir", type=str, default="data/nitrogen_battle_cache_bellman")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"❌ No manifest found at {manifest_path}. Did you run precache?")
        return

    print(f"📂 Loading manifest from {cache_dir}...")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Collect stats from the manifest's file list
    # (The manifest usually contains the full file entries with 'values_stats' 
    # if you ran the latest version of precache.py)
    
    files = manifest.get("files", [])
    if not files:
        print("⚠️ Manifest is empty.")
        return

    print(f"📊 Analyzing {len(files)} replays...")

    # Data containers
    medians = []
    p90s = []
    p99s = []
    mins = []
    maxs = []
    
    # Check if stats are in manifest or if we need to read individual .meta.json files
    # (Your code writes full stats into the manifest['files'] list, so this is fast)
    missing_stats = 0
    
    for entry in files:
        stats = entry.get("values_stats")
        if not stats:
            missing_stats += 1
            continue
            
        medians.append(stats.get("p50", 0))
        p90s.append(stats.get("abs_p90", 0)) # Using abs_p90 is best for "magnitude"
        p99s.append(stats.get("abs_p99", 0))
        mins.append(stats.get("min", 0))
        maxs.append(stats.get("max", 0))

    if missing_stats == len(files):
        print("❌ No stats found in manifest. You might need to re-run precache with --rebuild_manifest")
        return

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Signal Magnitude (The "clipping" check)
    # We want to see if values are generally within [-5, 5] or huge like [-1000, 1000]
    data = [p90s, p99s]
    ax1.boxplot(data, labels=["Abs P90 (Typical Max)", "Abs P99 (Extreme Max)"])
    ax1.set_title("Signal Magnitude (Absolute Return)")
    ax1.set_ylabel("Bellman Return Value")
    ax1.grid(True, alpha=0.3)
    
    # Add a reference line for your default tanh clip (usually 5.0)
    ax1.axhline(y=5.0, color='r', linestyle='--', label="Typical Tanh Clip (5.0)")
    ax1.legend()

    # 2. Distribution Center (The "Bias" check)
    # We want to know if the median is 0.0 (good) or drifting positive/negative
    ax2.hist(medians, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title("Distribution of Medians (Bias Check)")
    ax2.set_xlabel("Median Value per Replay")
    ax2.set_ylabel("Count of Replays")
    ax2.axvline(x=0.0, color='k', linestyle='--', label="Zero")
    ax2.legend()

    # Summary
    avg_p90 = np.mean(p90s)
    print("\n🧐 Diagnosis:")
    print(f"   Replays with stats: {len(p90s)}")
    print(f"   Avg Abs P90:        {avg_p90:.4f}")
    print(f"   Avg Abs P99:        {np.mean(p99s):.4f}")
    print(f"   Median Centering:   {np.mean(medians):.4f} (should be close to 0)")
    
    print("-" * 40)
    if avg_p90 < 0.1:
        print("⚠️  SIGNAL TOO WEAK: Your rewards are tiny. Increase --reward_scale.")
    elif avg_p90 > 20.0:
        print("⚠️  SIGNAL TOO STRONG: Your rewards are huge. Decrease --reward_scale or increase --norm_factor.")
    else:
        print(f"✅ SIGNAL HEALTHY: Values are in a learnable range.")
        print(f"   Recommended training --norm_factor: {avg_p90:.1f}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()