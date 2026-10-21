"""Shared data loss and correctly normalized full/projected residual losses.

The Fourier vectors analyze image-domain residuals. They do not change the
Neumann boundary conditions of the degradation solver, and are not assumed
to diagonalize that solver. Subspace loss divides energy by pixel count N,
not by the number m of retained real directions.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from scipy.ndimage import correlate

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def build_fourier_basis(H: int, W: int, m: int):
    """Return a real orthonormal matrix Q of shape (H*W, m), float64.

    Specification, to be recorded with each experiment:
    * constant mode is included first;
    * conjugate frequency classes are ordered by squared cycles-per-pixel
      radius, then signed row frequency, signed column frequency;
    * the lexicographically smaller raw (row,column) index represents a
      conjugate pair; cosine precedes sine;
    * self-conjugate DC/Nyquist classes contribute only one cosine;
    * truncate this real-vector list at exactly m vectors. Consequently an
      even m such as 32 can end with an unpaired cosine. This intentional,
      deterministic convention is not rotationally invariant.

    Work/storage is O(H*W*m), apart from sorting O(H*W) frequency indices.
    A complete basis is only intended for small numerical verification.
    """
    for name, value in (("H", H), ("W", W), ("m", m)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    N = H * W
    if m > N:
        raise ValueError("m cannot exceed H*W")
    classes = []
    for ky in range(H):
        for kx in range(W):
            other = ((-ky) % H, (-kx) % W)
            if (ky, kx) > other:
                continue
            fy = ky if ky <= H // 2 else ky - H
            fx = kx if kx <= W // 2 else kx - W
            # Integer key avoids floating-point tie changes for equal radii.
            radius = fy * fy * W * W + fx * fx * H * H
            classes.append((radius, fy, fx, ky, kx, other == (ky, kx)))
    classes.sort()
    yy = np.arange(H, dtype=np.float64)[:, None] / H
    xx = np.arange(W, dtype=np.float64)[None, :] / W
    Q = np.empty((N, m), dtype=np.float64)
    column = 0
    for _, fy, fx, _, _, self_conjugate in classes:
        phase = 2 * np.pi * (fy * yy + fx * xx)
        scale = 1 / math.sqrt(N) if self_conjugate else math.sqrt(2 / N)
        Q[:, column] = (scale * np.cos(phase)).reshape(-1)
        column += 1
        if column == m:
            break
        if not self_conjugate:
            Q[:, column] = (scale * np.sin(phase)).reshape(-1)
            column += 1
            if column == m:
                break
    if column != m:
        raise RuntimeError("internal Fourier basis dimension error")
    return Q


def _is_torch(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _flat_images(value):
    if value.ndim == 2:
        return value.reshape(1, -1)
    if value.ndim == 4 and value.shape[1] == 1:
        return value.reshape(value.shape[0], -1)
    raise ValueError("residual must have shape (H,W) or (B,1,H,W)")


def full_consistency_loss(residual):
    """Mean squared forward residual, averaged over batch and all pixels."""
    if _is_torch(residual):
        _flat_images(residual)
        return residual.square().mean()
    residual = np.asarray(residual)
    _flat_images(residual)
    return float(np.mean(np.square(residual, dtype=np.float64)))


def subspace_consistency_loss(residual, Q):
    """Mean projected energy / N; Q must contain orthonormal columns.

    For B grayscale images flattened as rows R, this is
    sum((R @ Q)**2)/(B*N), exactly the mean of ||QQ^T r||^2/N.
    Q is a fixed test space; no dense N-by-N projector is constructed.
    """
    if _is_torch(residual):
        flat = _flat_images(residual)
        basis = torch.as_tensor(Q, dtype=flat.dtype, device=flat.device)
        if basis.ndim != 2 or basis.shape[0] != flat.shape[1] or basis.shape[1] < 1:
            raise ValueError("Q must have shape (H*W,m), m>=1")
        coefficients = flat @ basis
        return coefficients.square().sum() / flat.numel()
    flat = _flat_images(np.asarray(residual, dtype=np.float64))
    basis = np.asarray(Q, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != flat.shape[1] or basis.shape[1] < 1:
        raise ValueError("Q must have shape (H*W,m), m>=1")
    return float(np.square(flat @ basis).sum() / flat.size)


def _window_size(height, width, requested):
    if isinstance(requested, bool) or not isinstance(requested, Integral) or requested < 1:
        raise ValueError("window_size must be a positive integer")
    size = min(height, width, requested)
    return size if size % 2 else size - 1


def _numpy_image_pair(pred, target):
    x, y = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("pred and target must have identical shapes")
    if x.ndim == 2:
        x, y = x[None, None], y[None, None]
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape) < 1:
        raise ValueError("images must have shape (H,W) or (B,1,H,W)")
    return x, y


def ssim_numpy(pred, target, window_size: int = 11, sigma: float = 1.5,
               data_range: float = 1.0):
    """Mean VALID local Gaussian SSIM, matching ssim_torch.

    Uses population moments, C1=(.01*range)^2, C2=(.03*range)^2.
    The largest odd window fitting the image is used for tiny QA images.
    Only locations with a complete window enter the mean. Boundary values
    used by scipy.correlate are discarded, so no padding convention affects
    the reported value. At the default window this removes a five-pixel
    border, agreeing with the evaluation metric's valid-interior definition.
    """
    x, y = _numpy_image_pair(pred, target)
    if sigma <= 0 or data_range <= 0 or not math.isfinite(sigma + data_range):
        raise ValueError("sigma and data_range must be finite and positive")
    size = _window_size(x.shape[-2], x.shape[-1], window_size)
    axis = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    smooth = lambda z: correlate(z, kernel, mode="mirror")
    mx, my = smooth(x), smooth(y)
    vx = np.maximum(smooth(x * x) - mx * mx, 0)
    vy = np.maximum(smooth(y * y) - my * my, 0)
    covariance = smooth(x * y) - mx * my
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    score = ((2 * mx * my + c1) * (2 * covariance + c2) /
             ((mx * mx + my * my + c1) * (vx + vy + c2)))
    border = size // 2
    if border:
        score = score[..., border:-border, border:-border]
    return float(np.clip(score, -1, 1).mean())


def ssim_torch(pred, target, window_size: int = 11, sigma: float = 1.5,
               data_range: float = 1.0):
    """Differentiable VALID SSIM for grayscale images; no reflect padding.

    Avoiding reflection_pad2d_backward also permits strict deterministic CUDA
    training, subject to the rest of the model and the installed backend.
    """
    if torch is None:
        raise ImportError("PyTorch is required for differentiable SSIM")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    x, y = (pred[None, None], target[None, None]) if pred.ndim == 2 else (pred, target)
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape) < 1:
        raise ValueError("images must have shape (H,W) or (B,1,H,W)")
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise TypeError("images must be floating point")
    if sigma <= 0 or data_range <= 0 or not math.isfinite(sigma + data_range):
        raise ValueError("sigma and data_range must be finite and positive")
    # Avoid loss of local moment accuracy in half precision.
    if x.dtype not in (torch.float32, torch.float64):
        x, y = x.float(), y.float()
    size = _window_size(x.shape[-2], x.shape[-1], window_size)
    axis = torch.arange(size, device=x.device, dtype=x.dtype) - size // 2
    g = torch.exp(-axis.square() / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    def smooth(z):
        return F.conv2d(z, kernel)

    mx, my = smooth(x), smooth(y)
    vx = (smooth(x.square()) - mx.square()).clamp_min(0)
    vy = (smooth(y.square()) - my.square()).clamp_min(0)
    covariance = smooth(x * y) - mx * my
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    score = ((2 * mx * my + c1) * (2 * covariance + c2) /
             ((mx.square() + my.square() + c1) * (vx + vy + c2)))
    return score.clamp(-1, 1).mean()


def data_loss(pred, target, beta: float = 0.1, window_size: int = 11,
              sigma: float = 1.5, data_range: float = 1.0):
    """Mean L1 + beta*(1-SSIM), with identical NumPy/Torch conventions."""
    if beta < 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    if _is_torch(pred):
        if not _is_torch(target):
            raise TypeError("pred and target must both be torch tensors")
        similarity = ssim_torch(pred, target, window_size, sigma, data_range)
        return (pred - target).abs().mean() + beta * (1 - similarity)
    x, y = _numpy_image_pair(pred, target)
    similarity = ssim_numpy(x, y, window_size, sigma, data_range)
    return float(np.abs(x - y).mean() + beta * (1 - similarity))
