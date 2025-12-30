# critic_rl/update_manifest.py
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    #default data/cache_rl/s2_s256_2
    parser.add_argument("cache_dir", type=str, help="Path to the cache directory containing .pt files.")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"Error: {cache_path} does not exist.")
        return

    files = sorted([p for p in cache_path.glob("*.pt") if p.name != "_manifest.pt"])
    print(f"Scanning {len(files)} files in {cache_path}...")

    total_samples = 0
    valid_files = 0

    # Iterate and sum 'r' (rewards) dimension 0 which is the sample count
    for p in tqdm(files, unit="file"):
        try:
            # map_location='cpu' prevents filling VRAM
            data = torch.load(p, map_location="cpu")
            if "r" in data:
                n = int(data["r"].shape[0])
                total_samples += n
                valid_files += 1
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    print(f"\nTotal Samples Found: {total_samples}")

    # Update or Create Manifest
    manifest_path = cache_path / "_manifest.pt"
    if manifest_path.exists():
        manifest = torch.load(manifest_path)
    else:
        manifest = {}

    manifest["total_samples"] = total_samples
    manifest["counted_files"] = valid_files
    
    torch.save(manifest, manifest_path)
    print(f"Manifest updated at: {manifest_path}")

if __name__ == "__main__":
    main()