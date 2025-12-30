import argparse
import torch
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Check cache files for correct Action Horizon.")
    parser.add_argument("--cache_dir", type=str, default="data/nitrogen_battle_cache_bellman")
    parser.add_argument("--expected", type=int, default=18, help="The horizon we expect (e.g. 18)")
    parser.add_argument("--delete_bad", action="store_true", help="Automatically delete mismatching files.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    expected_h = args.expected
    
    files = sorted(list(cache_dir.glob("*.pt")))
    if not files:
        print(f"⚠️ No .pt files found in {cache_dir}")
        return

    print(f"🔍 Scanning {len(files)} files for Action Horizon = {expected_h}...")

    bad_files = []
    good_count = 0

    # We use a progress bar, but print errors immediately above it
    pbar = tqdm(files, unit="file")
    
    for pt_file in pbar:
        try:
            # Map to CPU for speed check
            data = torch.load(pt_file, map_location="cpu")
            
            if "actions" not in data:
                msg = f"❌ BAD: {pt_file.name} (Missing 'actions')"
                tqdm.write(msg)  # Print immediately
                bad_files.append(pt_file)
                continue

            actual_h = data["actions"].shape[1]
            
            if actual_h != expected_h:
                msg = f"❌ BAD: {pt_file.name} (Horizon {actual_h} != {expected_h})"
                tqdm.write(msg)  # Print immediately
                bad_files.append(pt_file)
                
                # If delete flag is on, delete immediately
                if args.delete_bad:
                    try:
                        pt_file.unlink()
                        meta = pt_file.with_suffix(".meta.json")
                        if meta.exists():
                            meta.unlink()
                        tqdm.write(f"   🗑️ Deleted {pt_file.name}")
                    except Exception as e:
                        tqdm.write(f"   ⚠️ Failed to delete: {e}")
            else:
                good_count += 1
                
        except Exception as e:
            tqdm.write(f"❌ ERROR: {pt_file.name} ({e})")
            bad_files.append(pt_file)

    print("\n" + "="*40)
    print(f"✅ Correct Files: {good_count}")
    print(f"❌ Bad Files:     {len(bad_files)}")
    print("="*40)

    if bad_files and not args.delete_bad:
        print("\n💡 Run again with --delete_bad to remove these automatically.")

if __name__ == "__main__":
    main()