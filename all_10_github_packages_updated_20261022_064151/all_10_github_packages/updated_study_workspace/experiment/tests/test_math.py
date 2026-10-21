"""Numerical definition checks; these do not assert restoration performance."""

import unittest

import numpy as np

from docrestore.operators import DiffusionReaction, forward_numpy, forward_torch
from docrestore.losses import (build_fourier_basis, data_loss,
                               full_consistency_loss, subspace_consistency_loss,
                               ssim_numpy, ssim_torch)

try:
    import torch
except ImportError:
    torch = None


def _graph_laplacian(size):
    """Positive Neumann graph Laplacian, including degenerate size=1."""
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size - 1):
        matrix[i, i] += 1
        matrix[i + 1, i + 1] += 1
        matrix[i, i + 1] -= 1
        matrix[i + 1, i] -= 1
    return matrix


class NumpyOperatorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(813)
        self.image = self.rng.uniform(0, 1, (7, 10))

    def test_identity_and_zero_duration(self):
        np.testing.assert_allclose(forward_numpy(self.image, 0, 0, 3), self.image,
                                   atol=2e-14)
        np.testing.assert_allclose(forward_numpy(self.image, 3, 2, 0), self.image,
                                   atol=2e-14)

    def test_reaction_fades_towards_white(self):
        alpha, duration, steps = .4, 1.7, 9
        result = forward_numpy(self.image, 0, alpha, duration, steps)
        expected = 1 - (1 - self.image) / (1 + duration * alpha / steps) ** steps
        np.testing.assert_allclose(result, expected, atol=2e-14)
        self.assertTrue(np.all(result >= self.image - 1e-14))

    def test_heat_preserves_constant_and_mean_ink(self):
        constant = np.full((7, 10), .35)
        np.testing.assert_allclose(forward_numpy(constant, 2, 0, 3), constant,
                                   atol=2e-14)
        result = forward_numpy(self.image, 2, 0, 3)
        self.assertAlmostEqual(result.mean(), self.image.mean(), places=13)

    def test_dct_solver_matches_repeated_dense_implicit_steps(self):
        height, width = self.image.shape
        kappa, alpha, duration, steps, h = .8, .15, 1.1, 5, .7
        positive_laplacian = (np.kron(_graph_laplacian(height), np.eye(width)) +
                              np.kron(np.eye(height), _graph_laplacian(width))) / h ** 2
        matrix = np.eye(height * width) + duration / steps * (
            alpha * np.eye(height * width) + kappa * positive_laplacian)
        ink = (1 - self.image).reshape(-1)
        for _ in range(steps):
            ink = np.linalg.solve(matrix, ink)
        expected = 1 - ink.reshape(height, width)
        result = forward_numpy(self.image, kappa, alpha, duration, steps, h)
        np.testing.assert_allclose(result, expected, atol=2e-13, rtol=2e-13)

    def test_batch_parameters_and_affine_residual(self):
        batch = self.rng.uniform(0, 1, (3, 1, 7, 10))
        kappa, alpha, duration = [.1, .3, .9], [0, .2, .5], [.5, 1, 2]
        result = forward_numpy(batch, kappa, alpha, duration)
        for i in range(3):
            expected = forward_numpy(batch[i, 0], kappa[i], alpha[i], duration[i])
            np.testing.assert_allclose(result[i, 0], expected, atol=2e-14)
        other = self.rng.uniform(0, 1, batch.shape)
        # D(x)-D(y) equals A(x-y); D(e)-D(0) isolates A e.
        left = result - forward_numpy(other, kappa, alpha, duration)
        right = (forward_numpy(batch - other, kappa, alpha, duration) -
                 forward_numpy(np.zeros_like(batch), kappa, alpha, duration))
        np.testing.assert_allclose(left, right, atol=3e-14)

    def test_one_pixel_and_wrapper(self):
        image = np.array([[.2]])
        result = DiffusionReaction(n_steps=4).numpy(image, 99, .4, 2)
        expected = 1 - .8 / (1 + .4 * 2 / 4) ** 4
        self.assertAlmostEqual(result.item(), expected, places=14)

    def test_invalid_parameters(self):
        for kwargs in ({"kappa": -1}, {"alpha": np.nan}, {"T": -1},
                       {"n_steps": 0}, {"h": 0}, {"n_steps": 1.5}):
            values = dict(kappa=.1, alpha=.2, T=1)
            values.update(kwargs)
            with self.assertRaises(ValueError):
                forward_numpy(self.image, **values)


class ProjectionAndLossTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(22)

    def test_orthogonality_all_modes_odd_even_and_32(self):
        for height, width, m in [(5, 7, 35), (4, 6, 24), (8, 10, 32), (1, 1, 1)]:
            Q = build_fourier_basis(height, width, m)
            self.assertEqual(Q.shape, (height * width, m))
            np.testing.assert_allclose(Q.T @ Q, np.eye(m), atol=3e-14)
            np.testing.assert_allclose(Q[:, 0], 1 / np.sqrt(height * width),
                                       atol=2e-15)

    def test_projector_energy_and_complete_basis(self):
        residual = self.rng.normal(size=(3, 1, 5, 7))
        Q = build_fourier_basis(5, 7, 8)
        full = full_consistency_loss(residual)
        sub = subspace_consistency_loss(residual, Q)
        self.assertLessEqual(sub, full + 1e-14)
        projector = Q @ Q.T
        np.testing.assert_allclose(projector @ projector, projector, atol=2e-14)
        expected = np.mean((residual.reshape(3, -1) @ projector) ** 2)
        self.assertAlmostEqual(sub, expected, places=14)
        self.assertAlmostEqual(subspace_consistency_loss(
            residual, build_fourier_basis(5, 7, 35)), full, places=13)

    def test_orthogonal_residual_zero_subspace_loss(self):
        Q = build_fourier_basis(6, 8, 9)
        x = self.rng.normal(size=48)
        orthogonal = (x - Q @ (Q.T @ x)).reshape(6, 8)
        self.assertLess(subspace_consistency_loss(orthogonal, Q), 1e-28)

    def test_pixel_normalization_not_mode_normalization(self):
        residual = np.ones((4, 6))
        # DC is retained, so constant energy is one regardless of m.
        for m in (1, 4, 24):
            self.assertAlmostEqual(subspace_consistency_loss(
                residual, build_fourier_basis(4, 6, m)), 1, places=13)

    def test_data_loss_identity_and_tiny_windows(self):
        for shape in [(12, 15), (2, 1, 7, 8), (1, 1)]:
            image = self.rng.uniform(0, 1, shape)
            self.assertAlmostEqual(ssim_numpy(image, image), 1, places=12)
            self.assertAlmostEqual(data_loss(image, image), 0, places=12)

    def test_invalid_basis(self):
        for arguments in [(0, 5, 1), (3, 4, 13), (3, 4, 0), (3, 4, 1.5)]:
            with self.assertRaises(ValueError):
                build_fourier_basis(*arguments)


@unittest.skipIf(torch is None, "PyTorch not installed; NumPy checks still run")
class TorchMathTests(unittest.TestCase):
    def test_numpy_torch_batch_agreement_and_gradients(self):
        rng = np.random.default_rng(9)
        for shape in [(2, 1, 7, 10), (2, 1, 8, 9), (2, 1, 1, 1)]:
            source = rng.uniform(0, 1, shape)
            image = torch.tensor(source, dtype=torch.float64, requires_grad=True)
            kappa = torch.tensor([.3, .7], dtype=torch.float64, requires_grad=True)
            alpha = torch.tensor([.1, .4], dtype=torch.float64, requires_grad=True)
            duration = torch.tensor([.8, 1.3], dtype=torch.float64, requires_grad=True)
            result = forward_torch(image, kappa, alpha, duration, n_steps=5, h=.8)
            expected = forward_numpy(source, [.3, .7], [.1, .4], [.8, 1.3], 5, .8)
            np.testing.assert_allclose(result.detach().numpy(), expected,
                                       rtol=2e-12, atol=2e-12)
            result.square().mean().backward()
            for value in (image, kappa, alpha, duration):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_torch_projection_and_data_loss_agree_with_numpy(self):
        rng = np.random.default_rng(91)
        for shape in [(2, 1, 13, 15), (1, 1, 4, 6), (1, 1, 1, 1)]:
            source, target = rng.uniform(0, 1, (2,) + shape)
            pred = torch.tensor(source, dtype=torch.float64, requires_grad=True)
            truth = torch.tensor(target, dtype=torch.float64)
            Q = build_fourier_basis(shape[-2], shape[-1], min(7, shape[-2] * shape[-1]))
            residual = pred - truth
            self.assertAlmostEqual(full_consistency_loss(residual).item(),
                                   full_consistency_loss(source - target), places=12)
            sub = subspace_consistency_loss(residual, Q)
            self.assertAlmostEqual(sub.item(), subspace_consistency_loss(
                source - target, Q), places=12)
            ssim = ssim_torch(pred, truth)
            self.assertAlmostEqual(ssim.item(), ssim_numpy(source, target), places=10)
            loss = data_loss(pred, truth) + sub
            self.assertAlmostEqual(data_loss(pred, truth).item(),
                                   data_loss(source, target), places=10)
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_torch_identity_and_module(self):
        image = torch.rand((2, 1, 7, 8), dtype=torch.float64)
        output = DiffusionReaction(n_steps=3)(image, 0, 0, 1)
        torch.testing.assert_close(output, image, atol=2e-14, rtol=2e-14)


if __name__ == "__main__":
    unittest.main()
