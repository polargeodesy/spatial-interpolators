#!/usr/bin/env python
"""
inpaint.py
Written by Tyler Sutterley (05/2022)
Inpaint over missing data in a two-dimensional array using a
    penalized least square method based on discrete cosine
    transforms

INPUTS:
    xs: input x-coordinates
    ys: input y-coordinates
    zs: input data

OPTIONS:
    n: Number of iterations
        Use 0 for nearest neighbors interpolation
    s0: Smoothing
    z0: Initial guess for input data
    power: power for lambda function
    epsilon: relaxation factor

REFERENCES:
    D. Garcia, Robust smoothing of gridded data in one and higher
        dimensions with missing values. Computational Statistics &
        Data Analysis, 54(4), 1167--1178 (2010)
        https://doi.org/10.1016/j.csda.2009.09.020

    G. Wang, D. Garcia, Y. Liu, R. de Jeu, and A. J. Dolman,
        A three-dimensional gap filling method for large geophysical
        datasets: Application to global satellite soil moisture
        observations, Environmental Modelling & Software,
        30, 139--142 (2012)
        https://doi.org/10.1016/j.envsoft.2011.10.015

UPDATE HISTORY:
    Written 06/2022
"""

import numpy as np
import scipy.fftpack
import scipy.spatial


def inpaint(
    xs,
    ys,
    zs,
    n=100,
    s0=3,
    z0=None,
    power=2,
    epsilon=2,
):
    """
    Inpaint over missing data in a two-dimensional array using a
    penalized least square method based on discrete cosine transforms
    :cite:p:`Garcia:2010hn,Wang:2012ei`

    Parameters
    ----------
    xs: float
        input x-coordinates
    ys: float
        input y-coordinates
    zs: float
        input data
    n: int, default 100
        Number of iterations
        Use 0 for nearest neighbors interpolation
    s0: int, default 3
        Smoothing
    z0: float or NoneType, default None
        Initial guess for input data
    power: int, default 2
        power for lambda function
    epsilon: int, default 2
        relaxation factor
    """

    # find masked values
    if isinstance(zs, np.ma.MaskedArray):
        W = np.logical_not(zs.mask)
    else:
        W = np.isfinite(zs)
    # no valid values can be found
    if not np.any(W):
        raise ValueError("No valid values found")

    # dimensions of input grid
    ny, nx = np.shape(zs)
    # calculate lambda function
    L = np.zeros((ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(ny) / ny)[:, None], (ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(nx) / nx)[None, :], (ny, nx))
    LAMBDA = np.power(2.0 * (2.0 - L), power)

    # calculate initial values using nearest neighbors
    if z0 is None:
        z0 = nearest_neighbors(xs, ys, zs, W)

    # copy data to new array with 0 values for mask
    ZI = np.zeros((ny, nx), dtype=zs.dtype)
    ZI[W] = np.copy(z0[W])

    # smoothness parameters
    s = np.logspace(s0, -6, n)
    for i in range(n):
        # calculate discrete cosine transform
        GAMMA = 1.0 / (1.0 + s[i] * LAMBDA)
        discos = GAMMA * scipy.fftpack.dctn(W * (ZI - z0) + z0, norm="ortho")
        # update interpolated grid
        z0 = (
            epsilon * scipy.fftpack.idctn(discos, norm="ortho")
            + (1.0 - epsilon) * z0
        )

    # reset original values
    z0[W] = np.copy(zs[W])
    # return the inpainted grid
    return z0


# PURPOSE: use nearest neighbors to form an initial guess
def nearest_neighbors(xs, ys, zs, W):
    """
    Calculate nearest neighbors to form an initial for
    missing values

    Parameters
    ----------
    xs: float
        input x-coordinates
    ys: float
        input y-coordinates
    zs: float
        input data
    W: bool
        mask with valid points
    """
    # computation of distance Matrix
    # use scipy spatial KDTree routines
    xgrid, ygrid = np.meshgrid(xs, ys)
    tree = scipy.spatial.KDTree(np.c_[xgrid[W], ygrid[W]])
    # find nearest neighbors
    masked = np.logical_not(W)
    _, ii = tree.query(np.c_[xgrid[masked], ygrid[masked]], k=1)
    # dimensions of input grid
    ny, nx = np.shape(zs)
    # copy valid original values
    z0 = np.zeros((ny, nx), dtype=zs.dtype)
    z0[W] = np.copy(zs[W])
    # copy nearest neighbors
    z0[masked] = zs[W][ii]
    # return initial guess
    return z0
