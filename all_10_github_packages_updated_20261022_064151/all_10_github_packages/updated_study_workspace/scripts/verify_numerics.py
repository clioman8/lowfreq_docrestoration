"""Measure implementation errors without drawing restoration conclusions."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment"))
from docrestore.operators import forward_numpy, forward_torch
from docrestore.losses import build_fourier_basis, full_consistency_loss, subspace_consistency_loss
from docrestore.train import seed_everything

seed_everything(813)
torch.set_num_threads(4)
rng = np.random.default_rng(813)
x = rng.uniform(0, 1, (7, 10))
eta = dict(kappa=.8, alpha=.15, T=1.1, n_steps=5, h=.7)
def graph(n):
    out = np.zeros((n, n))
    for i in range(n-1):
        out[i, i] += 1
        out[i+1, i+1] += 1
        out[i, i+1] -= 1
        out[i+1, i] -= 1
    return out
L = (np.kron(graph(7), np.eye(10)) + np.kron(np.eye(7), graph(10))) / eta["h"]**2
M = np.eye(70) + eta["T"]/eta["n_steps"] * (eta["alpha"] * np.eye(70) + eta["kappa"] * L)
ink = (1-x).ravel()
for _ in range(eta["n_steps"]):
    ink = np.linalg.solve(M, ink)
dense = 1-ink.reshape(x.shape)
cpu = forward_numpy(x, **eta)
Q = build_fourier_basis(256, 256, 32)
r = rng.normal(size=(256, 256))
flat = r.ravel()
projected = Q @ (Q.T @ flat)
gain = (1 + eta["T"] * eta["alpha"]/eta["n_steps"])**(-eta["n_steps"])
result = {
    "kind": "implementation_verification_not_restoration_results",
    "seed": 813,
    "dense_implicit_euler_max_abs_error": float(np.max(np.abs(dense-cpu))),
    "identity_max_abs_error": float(np.max(np.abs(forward_numpy(x, 0, 0, 3)-x))),
    "reaction_only_max_abs_error": float(np.max(np.abs(forward_numpy(x, 0, eta["alpha"], eta["T"], eta["n_steps"])-(1-(1-x)*gain)))),
    "heat_mean_error": float(abs(forward_numpy(x, 2, 0, 3).mean()-x.mean())),
    "main_basis_orthogonality_max_abs_error": float(np.max(np.abs(Q.T @ Q-np.eye(32)))),
    "main_projector_idempotence_relative_error": float(np.linalg.norm(Q @ (Q.T @ projected)-projected)/np.linalg.norm(projected)),
    "random_projected_energy_fraction": subspace_consistency_loss(r, Q)/full_consistency_loss(r),
    "main_basis_shape": list(Q.shape),
    "torch_version": str(torch.__version__),
    "cuda_available": torch.cuda.is_available(),
}
for device in ("cpu", "cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        continue
    for dtype, label in ((torch.float64, "float64"), (torch.float32, "float32")):
        t = torch.tensor(x, device=device, dtype=dtype, requires_grad=True)
        y = forward_torch(t, **eta)
        (y.square().mean()).backward()
        result[f"{device}_{label}_numpy_max_abs_error"] = float(np.max(np.abs(y.detach().cpu().numpy()-cpu)))
        result[f"{device}_{label}_finite_gradient"] = bool(torch.isfinite(t.grad).all())
        result[f"{device}_{label}_nonzero_gradient"] = bool(t.grad.abs().max() > 0)
if torch.cuda.is_available():
    result["gpu"] = torch.cuda.get_device_name()
result["logical_processors"] = os.cpu_count()
out = ROOT / "experiment/provenance/numerical_verification.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
