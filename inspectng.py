import torch

ckpt = torch.load("weights/ng.pt", map_location="cpu", weights_only=False)

print("Top keys:", ckpt.keys())
print("model type:", type(ckpt["model"]))
print("ckpt_config type:", type(ckpt["ckpt_config"]))
print("step/epoch:", ckpt.get("step"), ckpt.get("epoch"))

m = ckpt["model"]
if isinstance(m, dict):
    keys = list(m.keys())
    print("state_dict entries:", len(keys))
    print("first 20 keys:")
    for k in keys[:20]:
        v = m[k]
        print(" ", k, "->", type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
else:
    print("WARNING: ckpt['model'] is not a dict; value:", repr(m)[:300])
