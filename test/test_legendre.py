#!/usr/bin/env python
"""
test_legendre.py (11/2021)
"""

import numpy as np
import spatial_interpolators as spi


# PURPOSE: test unnormalized Legendre polynomials
def test_unnormalized(l=3, x=[-1.0, -0.9, -0.8]):
    obs = spi.legendre(l, x)
    expected = np.array(
        [
            [-1.00000, -0.47250, -0.08000],
            [0.00000, -1.99420, -1.98000],
            [0.00000, -2.56500, -4.32000],
            [0.00000, -1.24229, -3.24000],
        ]
    )
    assert np.isclose(obs, expected, atol=1e-05).all()


# PURPOSE: test fully-normalized Legendre polynomials
def test_normalized(l=3, x=[-1.0, -0.9, -0.8]):
    obs = spi.legendre(l, x, normalize=True)
    expected = np.array(
        [
            [-2.64575, -1.25012, -0.21166],
            [-0.00000, 2.15398, 2.13864],
            [0.00000, -0.87611, -1.47556],
            [-0.00000, 0.17323, 0.45180],
        ]
    )
    assert np.isclose(obs, expected, atol=1e-05).all()
