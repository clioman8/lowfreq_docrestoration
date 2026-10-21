"""Matched, differentiable diffusion--reaction degradation.

The image convention is black ink on a white background, u in [0, 1].
Ink density v = 1 - u obeys v_t = kappa * Laplacian(v) - alpha * v.
No clipping, noise, or quantization is performed inside the operator.

The spatial operator is the symmetric nearest-neighbour graph Laplacian:
edge pixels have one neighbour per spatial axis.  This is the zero-flux
finite-volume / reflecting-endpoint discretization, diagonalized by an
orthonormal DCT-II.  Pixel spacing h defaults to 1 and must be recorded.
Each backward-Euler step has spectral multiplier
1 / (1 + dt * (alpha + kappa * mu)), where
mu[p,q] = 4/h**2 * (sin(pi*p/(2H))**2 + sin(pi*q/(2W))**2).

Thus D_eta(u) = A_eta u + (1 - A_eta 1), and for matched pairs
D_eta(pred) - D_eta(target) = A_eta (pred - target).  Full consistency
already weights high-frequency reconstruction errors less strongly.

NumPy uses scipy.fft; PyTorch implements the same DCT with FFTs, retaining
gradients with respect to both images and supplied tensor parameters.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from scipy.fft import dctn, idctn

try:
    import torch
    from torch import nn
except ImportError:  # NumPy generation and its checks do not require torch.
    torch = None
    nn = None


def _check_grid(n_steps: int, h: float) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not math.isfinite(float(h)) or float(h) <= 0:
        raise ValueError("h must be finite and positive")


def _numpy_images(u):
    x = np.asarray(u, dtype=np.float64)
    original_ndim = x.ndim
    if original_ndim == 2:
        x = x[None, None, :, :]
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape) < 1:
        raise ValueError("u must have shape (H,W) or (B,1,H,W), with nonempty axes")
    if not np.isfinite(x).all():
        raise ValueError("u must contain finite values")
    return x, original_ndim


def _numpy_parameter(value, batch: int, name: str):
    x = np.asarray(value, dtype=np.float64)
    if x.size == 1:
        x = np.full(batch, x.item(), dtype=np.float64)
    elif x.size == batch:
        x = x.reshape(batch)
    else:
        raise ValueError(f"{name} must be scalar or contain one value per image")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return x.reshape(batch, 1, 1, 1)


def forward_numpy(u, kappa, alpha, T, n_steps: int = 8, h: float = 1.0):
    """Degrade an image/batch, returning float64 with the input image shape.

    Parameters may be scalars or arrays with B values.  The n_steps value is
    fixed for the batch; each image may have its own T and hence dt=T/n_steps.
    Input values are not range-clipped, so the affine map remains exact.
    """
    _check_grid(n_steps, h)
    x, original_ndim = _numpy_images(u)
    batch, _, height, width = x.shape
    kappa = _numpy_parameter(kappa, batch, "kappa")
    alpha = _numpy_parameter(alpha, batch, "alpha")
    duration = _numpy_parameter(T, batch, "T")
    row = 4.0 * np.sin(np.pi * np.arange(height) / (2.0 * height)) ** 2
    col = 4.0 * np.sin(np.pi * np.arange(width) / (2.0 * width)) ** 2
    eigenvalues = (row[:, None] + col[None, :]) / float(h) ** 2
    gain = (1.0 + (duration / n_steps) *
            (alpha + kappa * eigenvalues[None, None, :, :])) ** (-n_steps)
    ink_spectrum = dctn(1.0 - x, type=2, axes=(-2, -1), norm="ortho")
    result = 1.0 - idctn(ink_spectrum * gain, type=2,
                        axes=(-2, -1), norm="ortho")
    return result[0, 0] if original_ndim == 2 else result


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for differentiable degradation")


def _dct_ortho(x, dim: int = -1):
    """Orthonormal DCT-II via one complex FFT; works for odd and even N."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    reordered = torch.cat((x[..., ::2], x[..., 1::2].flip(-1)), dim=-1)
    spectrum = torch.fft.fft(reordered, dim=-1)
    theta = -torch.arange(n, dtype=x.dtype, device=x.device) * (math.pi / (2 * n))
    values = spectrum.real * theta.cos() - spectrum.imag * theta.sin()
    scale = torch.full((n,), math.sqrt(2.0 / n), dtype=x.dtype, device=x.device)
    scale[0] = math.sqrt(1.0 / n)
    return (values * scale).movedim(-1, dim)


def _idct_ortho(x, dim: int = -1):
    """Inverse of _dct_ortho, without detaching the autograd graph."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    scale = torch.full((n,), math.sqrt(2.0 / n), dtype=x.dtype, device=x.device)
    scale[0] = math.sqrt(1.0 / n)
    real = x / scale
    imag = torch.cat((torch.zeros_like(real[..., :1]),
                      -real.flip(-1)[..., :-1]), dim=-1)
    theta = torch.arange(n, dtype=x.dtype, device=x.device) * (math.pi / (2 * n))
    spectrum = torch.complex(real * theta.cos() - imag * theta.sin(),
                             real * theta.sin() + imag * theta.cos())
    reordered = torch.fft.ifft(spectrum, dim=-1).real
    index = torch.arange(n, device=x.device)
    undo = torch.where(index.remainder(2) == 0,
                       index.div(2, rounding_mode="floor"),
                       n - 1 - index.div(2, rounding_mode="floor"))
    return reordered.index_select(-1, undo).movedim(-1, dim)


def _torch_parameter(value, image, batch: int, name: str):
    x = torch.as_tensor(value, device=image.device, dtype=image.dtype)
    if x.numel() == 1:
        x = x.reshape(1).expand(batch)
    elif x.numel() == batch:
        x = x.reshape(batch)
    else:
        raise ValueError(f"{name} must be scalar or contain one value per image")
    if not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return x.reshape(batch, 1, 1, 1)


def forward_torch(u, kappa, alpha, T, n_steps: int = 8, h: float = 1.0):
    """Torch counterpart of forward_numpy, differentiable in u and eta.

    Float32/64 retain their dtype. Half/bfloat16 inputs compute in float32
    and return to the input dtype. Input must be a floating-point tensor.
    """
    _require_torch()
    _check_grid(n_steps, h)
    if not isinstance(u, torch.Tensor) or not torch.is_floating_point(u):
        raise TypeError("u must be a floating-point torch.Tensor")
    original_ndim, original_dtype = u.ndim, u.dtype
    x = u[None, None, :, :] if original_ndim == 2 else u
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape) < 1:
        raise ValueError("u must have shape (H,W) or (B,1,H,W), with nonempty axes")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("u must contain finite values")
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    batch, _, height, width = x.shape
    kappa = _torch_parameter(kappa, x, batch, "kappa")
    alpha = _torch_parameter(alpha, x, batch, "alpha")
    duration = _torch_parameter(T, x, batch, "T")
    row = 4 * torch.sin(math.pi * torch.arange(height, dtype=x.dtype,
                                              device=x.device) / (2 * height)) ** 2
    col = 4 * torch.sin(math.pi * torch.arange(width, dtype=x.dtype,
                                              device=x.device) / (2 * width)) ** 2
    eigenvalues = (row[:, None] + col[None, :]) / float(h) ** 2
    gain = (1 + (duration / n_steps) *
            (alpha + kappa * eigenvalues[None, None, :, :])).pow(-n_steps)
    ink_spectrum = _dct_ortho(_dct_ortho(1 - x, -1), -2)
    ink = _idct_ortho(_idct_ortho(ink_spectrum * gain, -1), -2)
    result = (1 - ink).to(original_dtype)
    return result[0, 0] if original_ndim == 2 else result


_ModuleBase = nn.Module if nn is not None else object


class DiffusionReaction(_ModuleBase):
    """Fixed-discretization wrapper usable as an nn.Module or NumPy generator."""

    def __init__(self, n_steps: int = 8, h: float = 1.0):
        super().__init__()
        _check_grid(n_steps, h)
        self.n_steps = int(n_steps)
        self.h = float(h)

    def forward(self, u, kappa, alpha, T):
        return forward_torch(u, kappa, alpha, T, self.n_steps, self.h)

    def numpy(self, u, kappa, alpha, T):
        return forward_numpy(u, kappa, alpha, T, self.n_steps, self.h)

    if nn is None:
        __call__ = forward
