import unittest

import numpy as np
from scipy.spatial import ConvexHull

from src.parser import args_parser
from src.utils import (
    get_brillouin_zone_3d,
    get_conventional_reciprocal_cell,
    sample_extended_grid,
)


class ConventionalBZTests(unittest.TestCase):
    def test_flag_is_opt_in(self):
        parser = args_parser()
        self.assertFalse(parser.parse_args(["-n", "example"]).conventional_bz)
        self.assertTrue(
            parser.parse_args(["-n", "example", "--conventional-bz"]).conventional_bz
        )

    def test_centred_orthorhombic_reciprocal_lattice(self):
        cell = np.array([[0, 2, 3], [1, 0, 3], [1, 2, 0]], dtype=float)
        conventional = get_conventional_reciprocal_cell(cell)
        vertices, _, _ = get_brillouin_zone_3d(conventional)
        self.assertEqual(len(vertices), 8)
        np.testing.assert_allclose(vertices.min(axis=0), [-1, -2, -3])
        np.testing.assert_allclose(vertices.max(axis=0), [1, 2, 3])
        self.assertAlmostEqual(ConvexHull(vertices).volume / abs(np.linalg.det(cell)), 4)

    def test_hexagonal_prism_preserves_orientation(self):
        cell = np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 2.7]])
        angle = 0.37
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        cell = cell @ rotation
        expected, _, _ = get_brillouin_zone_3d(cell)
        conventional = get_conventional_reciprocal_cell(cell)
        vertices, _, facets = get_brillouin_zone_3d(conventional)
        self.assertEqual(len(vertices), 12)
        self.assertEqual(sorted(map(len, facets)), [4] * 6 + [6] * 2)
        for vertex in expected:
            self.assertLess(np.linalg.norm(vertices - vertex, axis=1).min(), 1e-10)

    def test_tetragonal_box(self):
        conventional = get_conventional_reciprocal_cell(np.diag([1., 1., 2.7]))
        vertices, _, _ = get_brillouin_zone_3d(conventional)
        self.assertEqual(len(vertices), 8)
        np.testing.assert_allclose(vertices.max(axis=0), [0.5, 0.5, 1.35])

    def test_periodic_sampling_fills_extended_zone(self):
        cell = np.array([[0, 2, 3], [1, 0, 3], [1, 2, 0]], dtype=float)
        conventional = get_conventional_reciprocal_cell(cell)
        axes = [np.linspace(0, 1, n) for n in [13, 15, 17]]
        fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        values = np.sum(np.cos(2 * np.pi * fractional), axis=-1)
        grid, _, _ = sample_extended_grid(values, cell, conventional, scale=1, order=3)
        expected = np.sum(np.cos(2 * np.pi * (grid.points @ np.linalg.inv(cell))), axis=1)
        np.testing.assert_allclose(grid["values"], expected, atol=5e-4)
        vertices, _, _ = get_brillouin_zone_3d(conventional)
        self.assertTrue(np.all(grid.points.min(axis=0) < vertices.min(axis=0)))
        self.assertTrue(np.all(grid.points.max(axis=0) > vertices.max(axis=0)))


if __name__ == "__main__":
    unittest.main()
