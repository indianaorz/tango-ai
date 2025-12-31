# critic_minimal/dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch


def _load_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    mpath = cache_dir / "manifest.json"
    if not mpath.exists():
        return None
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else None
    except Exception:
        return None


@dataclass(frozen=True)
class CacheSchema:
    seq_len: int
    action_dim: int
    scalar_dim: int


def _infer_schema_from_one_file(pt_path: Path) -> CacheSchema:
    ck = torch.load(pt_path, map_location="cpu")
    if not isinstance(ck, dict):
        raise RuntimeError(f"Bad cache file (not dict): {pt_path}")

    x = ck.get("x", None)
    y = ck.get("y", None)
    valid = ck.get("valid", None)

    if not isinstance(x, dict) or not torch.is_tensor(y) or not torch.is_tensor(valid):
        raise RuntimeError(f"Bad cache file (missing x/y/valid): {pt_path}")

    # y: [N,T]
    if y.ndim != 2:
        raise RuntimeError(f"Bad y shape in {pt_path}: {tuple(y.shape)}")
    T = int(y.shape[1])

    # action: [N,T,A]
    action = x.get("action", None)
    scalars = x.get("scalars", None)
    if not torch.is_tensor(action) or action.ndim != 3:
        raise RuntimeError(f"Bad x[action] in {pt_path}: {None if action is None else tuple(action.shape)}")
    if not torch.is_tensor(scalars) or scalars.ndim != 3:
        raise RuntimeError(f"Bad x[scalars] in {pt_path}: {None if scalars is None else tuple(scalars.shape)}")

    A = int(action.shape[2])
    S = int(scalars.shape[2])

    if int(action.shape[1]) != T or int(scalars.shape[1]) != T:
        raise RuntimeError(f"Time dim mismatch in {pt_path}: yT={T} actionT={action.shape[1]} scalarsT={scalars.shape[1]}")

    return CacheSchema(seq_len=T, action_dim=A, scalar_dim=S)


class StreamingMinimalQDataset(torch.utils.data.IterableDataset):
    """
    Stream-load cache files one by one (deterministic order/shuffle).

    Yields:
      (x: dict [T,...], y_raw: [T], valid:[T])
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        max_sequences: Optional[int] = None,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.seed = int(seed)
        self.max_sequences = max_sequences

        all_files = sorted([p for p in self.cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
        if not all_files:
            raise RuntimeError(f"No cache files in {cache_dir}")

        # deterministic split by filename stem
        rng = __import__("random").Random(self.seed)
        names = [p.stem for p in all_files]
        rng.shuffle(names)

        if float(val_ratio) <= 0:
            n_val = 0
        else:
            n_val = max(1, int(len(names) * float(val_ratio)))

        val_set = set(names[:n_val])

        if self.split == "val":
            self.files = [p for p in all_files if p.stem in val_set]
        else:
            self.files = [p for p in all_files if p.stem not in val_set]

        if not self.files:
            raise RuntimeError(f"{split} split has 0 files (cache_dir={cache_dir})")

        # length estimate from manifest (optional)
        self._approx_len = None
        m = _load_manifest(self.cache_dir)
        if m is not None:
            try:
                total = int(m.get("total_sequences", 0) or 0)
                num_files = int(m.get("num_files", 0) or len(all_files))
                if total > 0 and num_files > 0:
                    avg = total / float(num_files)
                    self._approx_len = int(len(self.files) * avg)
            except Exception:
                self._approx_len = None

        # schema inference (fast, 1 file)
        self.schema = _infer_schema_from_one_file(self.files[0])

    def __len__(self) -> int:
        if self._approx_len is None:
            return int(len(self.files) * 512)
        return int(self._approx_len)

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()

        # shard files across workers
        if worker_info is None:
            my_files = list(self.files)
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)
            per_worker = int(__import__("math").ceil(len(self.files) / float(num_workers)))
            a = worker_id * per_worker
            b = min(a + per_worker, len(self.files))
            my_files = self.files[a:b]

        # deterministic per-worker shuffle (no time dependence)
        rng = __import__("random").Random(self.seed + 1009 * worker_id + 9176 * num_workers)
        rng.shuffle(my_files)

        # deterministic per-worker row permutation generator
        g = torch.Generator()
        g.manual_seed(self.seed + 1000003 * worker_id)

        yielded = 0

        for p in my_files:
            ck = None
            try:
                ck = torch.load(p, map_location="cpu")
                if not isinstance(ck, dict):
                    continue
                x_dict = ck.get("x", None)
                y = ck.get("y", None)
                valid = ck.get("valid", None)
                if not isinstance(x_dict, dict) or not torch.is_tensor(y) or not torch.is_tensor(valid):
                    continue

                # basic shape checks
                if y.ndim != 2 or valid.ndim != 2:
                    continue
                if int(y.shape[1]) != self.schema.seq_len:
                    continue

                n_rows = int(y.shape[0])
                if n_rows == 0:
                    continue

                idxs = torch.randperm(n_rows, generator=g)

                for i in idxs.tolist():
                    x_out = {k: v[i] for k, v in x_dict.items() if torch.is_tensor(v)}
                    yield x_out, y[i], valid[i]
                    yielded += 1
                    if self.max_sequences is not None and yielded >= int(self.max_sequences):
                        return
            except Exception:
                continue
            finally:
                if ck is not None:
                    del ck


def collate_minimal_q(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    xs, ys, vs = zip(*batch)
    out: Dict[str, torch.Tensor] = {}
    for k in xs[0].keys():
        out[k] = torch.stack([x[k] for x in xs], dim=0)  # [B,T,...]
    y = torch.stack(list(ys), dim=0)  # [B,T] RAW
    valid = torch.stack(list(vs), dim=0)  # [B,T]
    return out, y, valid


__all__ = ["CacheSchema", "StreamingMinimalQDataset", "collate_minimal_q"]
