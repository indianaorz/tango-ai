import json
import collections
import statistics
from pathlib import Path

DATA_PATH = "data/chipwindows_v2/strategy_v2.jsonl"

def analyze():
    if not Path(DATA_PATH).exists():
        print(f"File not found: {DATA_PATH}")
        return

    x_coords = []
    y_coords = []
    points = []

    print(f"Scanning {DATA_PATH}...")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                row = json.loads(line)
                # Check both player and enemy to get full board coverage
                p_pos = row.get("player_pos_open")
                e_pos = row.get("enemy_pos_open")
                
                if p_pos and len(p_pos) == 2:
                    x, y = int(p_pos[0]), int(p_pos[1])
                    x_coords.append(x)
                    y_coords.append(y)
                    points.append((x, y))
                    
                if e_pos and len(e_pos) == 2:
                    x, y = int(e_pos[0]), int(e_pos[1])
                    x_coords.append(x)
                    y_coords.append(y)
                    points.append((x, y))
                    
            except: continue

    if not x_coords:
        print("No coordinate data found.")
        return

    # --- ANALYZE X (Columns) ---
    print("\n" + "="*40)
    print(" X COORDINATE CLUSTERS (COLUMNS)")
    print("="*40)
    
    # Simple clustering: Round to nearest 10 to group noisy coords
    x_clusters = collections.defaultdict(list)
    for x in x_coords:
        bucket = round(x / 10) * 10
        x_clusters[bucket].append(x)
        
    sorted_buckets = sorted(x_clusters.keys())
    
    print(f"{'Bucket':<10} | {'Count':<8} | {'Min':<6} | {'Max':<6} | {'Avg':<6}")
    print("-" * 50)
    
    valid_cols = []
    
    for b in sorted_buckets:
        vals = x_clusters[b]
        if len(vals) < 50: continue # Skip noise
        
        vmin = min(vals)
        vmax = max(vals)
        vavg = statistics.mean(vals)
        valid_cols.append(vavg)
        
        print(f"{b:<10} | {len(vals):<8} | {vmin:<6} | {vmax:<6} | {vavg:<6.1f}")

    # --- ANALYZE Y (Rows) ---
    print("\n" + "="*40)
    print(" Y COORDINATE CLUSTERS (ROWS)")
    print("="*40)
    
    y_clusters = collections.defaultdict(list)
    for y in y_coords:
        bucket = round(y / 5) * 5 # Tighter grouping for Y
        y_clusters[bucket].append(y)
        
    sorted_y = sorted(y_clusters.keys())
    
    print(f"{'Bucket':<10} | {'Count':<8} | {'Min':<6} | {'Max':<6} | {'Avg':<6}")
    print("-" * 50)
    
    valid_rows = []
    
    for b in sorted_y:
        vals = y_clusters[b]
        if len(vals) < 50: continue
        
        vmin = min(vals)
        vmax = max(vals)
        vavg = statistics.mean(vals)
        valid_rows.append(vavg)
        
        print(f"{b:<10} | {len(vals):<8} | {vmin:<6} | {vmax:<6} | {vavg:<6.1f}")

    # --- SUGGESTED ALGORITHM ---
    print("\n" + "="*40)
    print(" SUGGESTED MAPPING LOGIC")
    print("="*40)
    
    if len(valid_cols) >= 6:
        # Assuming sorted buckets correspond to cols 0-5
        # Calc midpoints
        print("def pos_to_grid_idx(x, y):")
        print("    # X Thresholds based on clusters:")
        print(f"    # Cols appear centered at: {[int(c) for c in valid_cols[:6]]}")
        
        # Calculate boundaries between clusters
        boundaries = []
        for i in range(len(valid_cols)-1):
            mid = (valid_cols[i] + valid_cols[i+1]) / 2
            boundaries.append(int(mid))
            
        print(f"    # Suggested X Boundaries: {boundaries}")
        print("    if x < 40: col = 0  # (Standard BN6 width is 40)")
        print("    else: col = int(x // 40)")
        print("")
        print("    # Y Thresholds based on clusters:")
        print(f"    # Rows appear centered at: {[int(r) for r in valid_rows]}")
        
    else:
        print("Not enough column data to generate logic automatically.")

if __name__ == "__main__":
    analyze()