# critic_minimal/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FlowDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        context_len: int = 256,
        pred_len: int = 18,
        *,
        ret_scale: float = 50.0,  # tune this; 30-100 are typical
        ret_clip: float = 200.0,  # safety clamp before tanh
    ):
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.ret_scale = float(ret_scale)
        self.ret_clip = float(ret_clip)

        if self.context_len <= 0:
            raise ValueError("context_len must be > 0")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be > 0")

        self.data_store: List[Dict[str, torch.Tensor]] = []
        self.indices: List[Tuple[int, int]] = []
        self.weights: List[float] = []

        files = sorted(list(Path(cache_dir).glob("*.pt")))
        if not files:
            raise RuntimeError(f"No files in {cache_dir}")

        print(f"Loading {len(files)} replays into RAM...")

        for f_path in tqdm(files):
            try:
                d = torch.load(f_path, map_location="cpu")
                if "states" not in d:
                    continue

                states = d["states"].float()    # [N, feat]
                actions = d["actions"].float()  # [N, act]
                returns = d["returns"].float()  # [N]
                weights = d["weights"].float()  # [N]

                N = int(weights.shape[0])
                if N <= 0:
                    continue
                if actions.shape[0] != N or states.shape[0] != N or returns.shape[0] != N:
                    # corrupted / mismatched cache
                    continue

                # Make sure we NEVER sample indices that don't have full future action slice.
                # Valid t must satisfy: t + pred_len <= N  => t <= N - pred_len
                # We enforce this in two ways:
                #   (1) zero out tail weights
                #   (2) explicitly check t + pred_len <= N when collecting indices
                if N > self.pred_len:
                    weights[(N - self.pred_len + 1) :].fill_(0.0)  # conservative tail kill
                else:
                    weights.fill_(0.0)

                valid_t = torch.nonzero(weights > 0).flatten()

                store_idx = len(self.data_store)
                self.data_store.append({"s": states, "a": actions, "r": returns})

                for t in valid_t:
                    ti = int(t.item())
                    if ti < 0:
                        continue
                    if ti + self.pred_len > N:
                        continue  # hard safety
                    self.indices.append((store_idx, ti))
                    self.weights.append(float(weights[ti].item()))

            except Exception as e:
                print(f"Skipping {f_path.name}: {e}")

        if not self.indices:
            raise RuntimeError("No valid samples found (all weights zero?)")

        # sampler likes double
        self.weights_tensor = torch.tensor(self.weights, dtype=torch.double)
        print(f"Dataset Ready. Samples: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def _encode_return(self, raw_val: torch.Tensor) -> torch.Tensor:
        x = raw_val.clamp(min=-self.ret_clip, max=self.ret_clip)
        return torch.tanh(x / self.ret_scale)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, t = self.indices[idx]
        data = self.data_store[f_idx]
        states = data["s"]  # [N, feat]
        actions = data["a"] # [N, act]
        returns = data["r"] # [N]

        # History slice [t-context_len : t], left-pad with zeros if needed
        start_hist = t - self.context_len
        if start_hist < 0:
            pad = torch.zeros((abs(start_hist), states.shape[1]), dtype=states.dtype)
            hist_slice = states[0:t]
            history = torch.cat([pad, hist_slice], dim=0)
        else:
            history = states[start_hist:t]

        # Future actions [t : t+pred_len] (guaranteed full by construction)
        future_actions = actions[t : t + self.pred_len]
        if future_actions.shape[0] != self.pred_len:
            raise IndexError("future_actions shorter than pred_len; check cache weights / index filtering")

        raw_ret = returns[t]
        ret_val = self._encode_return(raw_ret).unsqueeze(0)

        return {
            "history": history,         # [context_len, feat]
            "action": future_actions,   # [pred_len, act]
            "return": ret_val,          # [1]
        }
