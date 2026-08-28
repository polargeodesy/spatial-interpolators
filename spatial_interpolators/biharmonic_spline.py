#!/usr/bin/env python
"""
biharmonic_spline.py
Written by Tyler Sutterley (05/2022)

Interpolates data using 2D biharmonic splines (Sandwell, 1987)
With or without tension parameters (Wessel and Bercovici, 1998)
or using the regularized function of Mitasova and Mitas 1993

CALLING SEQUENCE:
    ZI = biharmonic_spline(xs, ys, zs, XI, YI)

INPUTS:
    xs: input X data
    ys: input Y data
    zs: input data
    XI: grid X for output ZI
    YI: grid Y for output ZI

OUTPUTS:
    ZI: interpolated grid

OPTIONS:
    metric: distance metric to use (default euclidean)
    tension: tension to use in interpolation (between 0 and 1)
    regular: use regularized function of Mitasova and Mitas
    eps: minimum distance value for valid points (default 1e-7)
    scale: scale factor for normalized lengths (default 2e-2)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

REFERENCES:
    Sandwell, Biharmonic spline interpolation of GEOS-3 and SEASAT
        altimeter data, Geophysical Research Letters, 14(2), (1987)
    Wessel and Bercovici, Interpolation with Splines in Tension: A
        Green's Function Approach, Mathematical Geology, 30(1), (1998)
    Mitasova and Mitas, Mathematical Geology, 25(6), (1993)

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
        update regularized spline function to use arrays
    Updated 07/2021: using scipy spatial distance routines
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Updated 08/2016: detrend input data and retrend output data. calculate c
        added regularized function of Mitasova and Mitas
    Updated 06/2016: added TENSION parameter (Wessel and Bercovici, 1998)
    Written 06/2016
"""

import numpy as np
import scipy.spatial


def biharmonic_spline(
    xs,
    ys,
    zs,
    XI,
    YI,
    metric="euclidean",
    tension=0,
    regular=False,
    eps=1e-7,
    scale=0.02,
):
    r"""
    Interpolates a sparse grid using 2D biharmonic splines with or
    without tension parameters or regularized functions
    :cite:p:`Sandwell:1987is,Wessel:1998ex`
    :cite:p:`Mitasova:1993gz`

    Parameters
    ----------
    xs: float
        input x-coordinates
    ys: float
        input y-coordinates
    zs: float
        input data
    XI: float
        output x-coordinates for data grid
    YI: float
        output y-coordinates for data grid
    metric: str, default 'euclidean'
        distance metric to use
    tension: float, default 0
        tension to use in interpolation
        value must be between 0 and 1
    regular: bool, default False
        Use regularized function of Mitasova and Mitas
    eps: float, default 1e-7
        minimum distance value for valid points
    scale: float, default 2e-2
        scale factor for normalized lengths

    Returns
    -------
    ZI: float
        interpolated data grid
    """
    # remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    # size of new matrix
    if np.ndim(XI) == 1:
        nx = len(XI)
    else:
        nx, ny = np.shape(XI)

    # Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception("Length of input arrays must be equal")
    if np.shape(XI) != np.shape(YI):
        raise Exception("Size of output arrays must be equal")
    if (tension < 0) or (tension >= 1):
        raise ValueError("tension must be greater than 0 and less than 1")

    # Compute GG matrix for GG*m = d inversion problem
    npts = len(zs)
    GG = np.zeros((npts, npts))
    # Computation of distance Matrix (data to data)
    if metric == "brute":
        # use linear algebra to compute euclidean distances
        Rd = distance_matrix(np.array([xs, ys]), np.array([xs, ys]))
    else:
        # use scipy spatial distance routines
        Rd = scipy.spatial.distance.cdist(
            np.array([xs, ys]).T, np.array([xs, ys]).T, metric=metric
        )
    # Calculate length scale for regularized case (Mitasova and Mitas)
    length_scale = np.sqrt(
        (XI.max() - XI.min()) ** 2 + (YI.max() - YI.min()) ** 2
    )
    # calculate Green's function for valid points (with or without tension)
    ii, jj = np.nonzero(Rd >= eps)
    if tension == 0:
        GG[ii, jj] = (Rd[ii, jj] ** 2) * (np.log(Rd[ii, jj]) - 1.0)
    elif regular:
        GG[ii, jj] = regular_spline2D(Rd[ii, jj], tension, scale * length_scale)
    else:
        GG[ii, jj] = green_spline2D(Rd[ii, jj], tension)
    # detrend dataset
    z0, r0, p = detrend2D(xs, ys, zs)
    # Compute model m for detrended data
    m = np.linalg.lstsq(GG, z0, rcond=-1)[0]

    # Computation of distance Matrix (data to mesh points)
    if metric == "brute":
        # use linear algebra to compute euclidean distances
        Re = distance_matrix(
            np.array([XI.flatten(), YI.flatten()]), np.array([xs, ys])
        )
    else:
        # use scipy spatial distance routines
        Re = scipy.spatial.distance.cdist(
            np.array([XI.flatten(), YI.flatten()]).T,
            np.array([xs, ys]).T,
            metric=metric,
        )
    gg = np.zeros_like(Re)
    # calculate Green's function for valid points (with or without tension)
    ii, jj = np.nonzero(Re >= eps)
    if tension == 0:
        gg[ii, jj] = (Re[ii, jj] ** 2) * (np.log(Re[ii, jj]) - 1.0)
    elif regular:
        gg[ii, jj] = regular_spline2D(Re[ii, jj], tension, scale * length_scale)
    else:
        gg[ii, jj] = green_spline2D(Re[ii, jj], tension)

    # Find 2D interpolated surface through irregular/regular X, Y grid points
    if np.ndim(XI) == 1:
        ZI = np.squeeze(np.dot(gg, m))
    else:
        ZI = np.zeros((nx, ny))
        ZI[:, :] = np.dot(gg, m).reshape(nx, ny)
    # return output matrix after retrending
    return (ZI + r0[2]) + (XI - r0[0]) * p[0] + (YI - r0[1]) * p[1]


# Removing mean and slope in 2-D dataset
# http://www.soest.hawaii.edu/wessel/tspline/
def detrend2D(xi, yi, zi):
    # Find mean values
    r0 = np.zeros((3))
    r0[0] = xi.mean()
    r0[1] = yi.mean()
    r0[2] = zi.mean()
    # Extract mean values from X, Y and Z
    x0 = xi - r0[0]
    y0 = yi - r0[1]
    z0 = zi - r0[2]
    # Find slope parameters
    p = np.linalg.lstsq(np.transpose([x0, y0]), z0, rcond=-1)[0]
    # Extract slope from data
    z0 = z0 - x0 * p[0] - y0 * p[1]
    # return the detrended value, the mean values, and the slope parameters
    return (z0, r0, p)


# calculate Euclidean distances between points as matrices
def distance_matrix(x, cntrs):
    s, M = np.shape(x)
    s, N = np.shape(cntrs)
    D = np.zeros((M, N))
    for d in range(s):
        (ii,) = np.dot(d, np.ones((1, N))).astype(np.int64)
        (jj,) = np.dot(d, np.ones((1, M))).astype(np.int64)
        dx = x[ii, :].transpose() - cntrs[jj, :]
        D += dx**2
    D = np.sqrt(D)
    return D


# Green function for 2-D spline in tension (Wessel et al, 1998)
# http://www.soest.hawaii.edu/wessel/tspline/
def green_spline2D(x, t):
    # in tension: G(u) = G(u) - log(u)
    # where u = c * x and c = sqrt (t/(1-t))
    c = np.sqrt(t / (1.0 - t))
    # allocate for output Green's function
    G = np.zeros_like(x)
    # inverse of tension parameter
    inv_c = 1.0 / c
    # log(2) - 0.5772156
    g0 = np.log(2) - np.euler_gamma
    # find points below (or equal to) 2 times inverse tension parameter
    (ii,) = np.nonzero(x <= (2.0 * inv_c))
    u = c * x[ii]
    y = (0.5 * u) ** 2
    z = (u / 3.75) ** 2
    # Green's function for points ii (less than or equal to 2.0*c)
    # from modified Bessel function of order zero
    G[ii] = (
        -np.log(0.5 * u)
        * (
            z
            * (
                3.5156229
                + z
                * (
                    3.0899424
                    + z
                    * (
                        1.2067492
                        + z * (0.2659732 + z * (0.360768e-1 + z * 0.45813e-2))
                    )
                )
            )
        )
    ) + (
        y
        * (
            0.42278420
            + y
            * (
                0.23069756
                + y
                * (
                    0.3488590e-1
                    + y * (0.262698e-2 + y * (0.10750e-3 + y * 0.74e-5))
                )
            )
        )
    )
    # find points above 2 times inverse tension parameter
    (ii,) = np.nonzero(x > 2.0 * inv_c)
    y = 2.0 * inv_c / x[ii]
    u = c * x[ii]
    # Green's function for points ii (greater than 2.0*c)
    G[ii] = (
        (np.exp(-u) / np.sqrt(u))
        * (
            1.25331414
            + y
            * (
                -0.7832358e-1
                + y
                * (
                    0.2189568e-1
                    + y
                    * (
                        -0.1062446e-1
                        + y
                        * (0.587872e-2 + y * (-0.251540e-2 + y * 0.53208e-3))
                    )
                )
            )
        )
        + np.log(u)
        - g0
    )
    return G


# Regularized spline in tension (Mitasova and Mitas, 1993)
def regular_spline2D(r, t, l):
    # calculate tension parameter
    p = np.sqrt(t / (1.0 - t)) / l
    z = (0.5 * p * r) ** 2
    # allocate for output Green's function
    G = np.zeros_like(r)
    # Green's function for points A (less than or equal to 1)
    (A,) = np.nonzero(z <= 1.0)
    Pa = [0.0, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857]
    G[A] = polynomial_sum(Pa, z[A])
    # Green's function for points B (greater than 1)
    (B,) = np.nonzero(z > 1.0)
    Pn = [0.2677737343, 8.6347608925, 18.0590169730, 8.5733287401, 1]
    Pd = [3.9584869228, 21.0996530827, 25.6329561486, 9.5733223454, 1]
    En = polynomial_sum(Pn, z[B])
    Ed = polynomial_sum(Pd, z[B])
    G[B] = np.log(z[B]) + np.euler_gamma + (En / Ed) / (z[B] * np.exp(z[B]))
    return G


# calculate the sum of a polynomial function of a variable
def polynomial_sum(x1, x2):
    """
    Calculates the sum of a polynomial function of a variable

    Arguments
    ---------
    x1: leading coefficient of polynomials of increasing order
    x2: coefficients to be raised by polynomials
    """
    # convert variable to array if importing a single value
    x2 = np.atleast_1d(x2)
    return np.sum([c * (x2**i) for i, c in enumerate(x1)], axis=0)
