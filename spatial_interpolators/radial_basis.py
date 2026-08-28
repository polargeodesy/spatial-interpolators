#!/usr/bin/env python
"""
radial_basis.py
Written by Tyler Sutterley (08/2026)

Interpolates data using radial basis functions

CALLING SEQUENCE:
    ZI = radial_basis(xs, ys, zs, XI, YI, polynomial=0,
        smooth=smooth, epsilon=epsilon, method='inverse')

INPUTS:
    xs: scaled input X data
    ys: scaled input Y data
    zs: input data
    XI: scaled grid X for output ZI
    YI: scaled grid Y for output ZI

OUTPUTS:
    ZI: interpolated data grid

OPTIONS:
    smooth: smoothing weights
    metric: distance metric to use (default euclidean)
    epsilon: adjustable constant for distance functions
        default is mean Euclidean distance
    polynomial: polynomial order if augmenting radial basis functions
        default None: no polynomials
    method: radial basis function
        multiquadric
        inverse_multiquadric or inverse (default)
        inverse_quadratic
        gaussian
        linear (first-order polyharmonic spline)
        cubic (third-order polyharmonic spline)
        quintic (fifth-order polyharmonic spline)
        thin_plate: thin-plate spline

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

REFERENCES:
    R. L. Hardy, Multiquadric equations of topography and other irregular
        surfaces, J. Geophys. Res., 76(8), 1905-1915, 1971.
    M. Buhmann, "Radial Basis Functions", Cambridge Monographs on Applied and
        Computational Mathematics, 2003.

UPDATE HISTORY:
    Updated 08/2026: fix case where floating point errors cause negative roots
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Updated 07/2021: using scipy spatial distance routines
    Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
    Updated 01/2017: epsilon in polyharmonic splines (linear, cubic, quintic)
    Updated 08/2016: using format text within ValueError, edit constant vector
        added low-order polynomial option (previously used default constant)
    Updated 01/2016: new hierarchical_radial_basis function
        that first reduces to points within distance.  added cutoff option
    Updated 10/2014: added third dimension (spherical)
    Written 08/2014
"""

from __future__ import print_function, division
import numpy as np
import scipy.spatial


def radial_basis(
    xs,
    ys,
    zs,
    XI,
    YI,
    smooth=0.0,
    metric="euclidean",
    epsilon=None,
    method="inverse",
    polynomial=None,
    **kwargs,
):
    """
    Interpolates data using radial basis functions :cite:p:`Hardy:1971em`
    :cite:p:`Buhmann:2003cc`

    Parameters
    ----------
    xs: float
        scaled input x-coordinates
    ys: float
        scaled input y-coordinates
    zs: float
        input data
    XI: float
        scaled output x-coordinates for data grid
    YI: float
        scaled output y-coordinates for data grid
    smooth: float, default 0.0
        smoothing weights
    metric: str, default 'euclidean'
        distance metric to use
    epsilon: float or NoneType, default None
        adjustable constant for distance functions
    method: str, default 'inverse'
        radial basis function

        * ``'multiquadric'``
        * ``'inverse_multiquadric'`` or ``'inverse'``
        * ``'inverse_quadratic'``
        * ``'gaussian'``
        * ``'linear'``
        * ``'cubic'``
        * ``'quintic'``
        * ``'thin_plate'``
    polynomial: int or NoneType, default None
        polynomial order if augmenting radial basis functions

    Returns
    -------
    ZI: interpolated data grid
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

    # get formula for radial basis function
    RBF, kwargs = formula(method)

    # Computation of data distance matrix (data to data)
    if metric == "brute":
        # use linear algebra to compute euclidean distances
        Rd = distance_matrix(np.array([xs, ys]), np.array([xs, ys]))
    else:
        # use scipy spatial distance routines
        Rd = scipy.spatial.distance.cdist(
            np.array([xs, ys]).T, np.array([xs, ys]).T, metric=metric
        )
    # shape of distance matrix
    N, M = np.shape(Rd)

    # if epsilon is not specified
    if epsilon is None:
        # calculate norm with mean euclidean distance
        uix, uiy = np.nonzero(np.tri(N, M=M, k=-1))
        epsilon = np.mean(Rd[uix, uiy])

    # possible augmentation of the PHI Matrix with polynomial Vectors
    if polynomial is None:
        # calculate radial basis function for data-to-data with smoothing
        PHI = RBF(epsilon, Rd, **kwargs) + np.eye(N, M=M) * smooth
        DMAT = zs.copy()
    else:
        # number of polynomial coefficients
        nt = (polynomial**2 + 3 * polynomial) // 2 + 1
        # calculate radial basis function for data-to-data with smoothing
        PHI = np.zeros((N + nt, M + nt))
        PHI[:N, :M] = RBF(epsilon, Rd, **kwargs) + np.eye(N, M=M) * smooth
        # augmentation of PHI matrix with polynomials
        POLY = polynomial_matrix(xs, ys, polynomial)
        DMAT = np.concatenate(([zs, np.zeros((nt))]), axis=0)
        # augment PHI matrix
        for t in range(nt):
            PHI[:N, M + t] = POLY[:, t]
            PHI[N + t, :M] = POLY[:, t]

    # Computation of the Weights
    w = np.linalg.lstsq(PHI, DMAT[:, np.newaxis], rcond=-1)[0]

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
    # calculate radial basis function for data-to-mesh matrix
    E = RBF(epsilon, Re, **kwargs)

    # possible augmentation of the Evaluation Matrix with polynomial vectors
    if polynomial is not None:
        P = polynomial_matrix(XI.flatten(), YI.flatten(), polynomial)
        E = np.concatenate(([E, P]), axis=1)
    # calculate output interpolated array (or matrix)
    if np.ndim(XI) == 1:
        ZI = np.squeeze(np.dot(E, w))
    else:
        ZI = np.zeros((nx, ny))
        ZI[:, :] = np.dot(E, w).reshape(nx, ny)
    # return the interpolated array (or matrix)
    return ZI


# select radial basis function
def formula(method):
    # check if formula name is listed
    RBF_formulas = [
        "multiquadric",
        "inverse_multiquadric",
        "inverse",
        "quadratic",
        "inverse_quadratic",
        "gaussian",
        "linear",
        "cubic",
        "quintic",
        "thin_plate",
    ]
    assert method in RBF_formulas, f"Method {method} not implemented"
    # create python dictionary of radial basis function formulas
    radial_basis_functions = {key: {} for key in RBF_formulas}
    radial_basis_functions["multiquadric"]["functional"] = poly
    radial_basis_functions["multiquadric"]["order"] = 0.5
    radial_basis_functions["inverse_multiquadric"]["functional"] = poly
    radial_basis_functions["inverse_multiquadric"]["order"] = -0.5
    radial_basis_functions["inverse"]["functional"] = poly
    radial_basis_functions["inverse"]["order"] = -0.5
    radial_basis_functions["quadratic"]["functional"] = poly
    radial_basis_functions["quadratic"]["order"] = 1.0
    radial_basis_functions["inverse_quadratic"]["functional"] = poly
    radial_basis_functions["inverse_quadratic"]["order"] = -1.0
    radial_basis_functions["gaussian"]["functional"] = gaussian
    radial_basis_functions["linear"]["functional"] = poly_spline
    radial_basis_functions["linear"]["order"] = 1.0
    radial_basis_functions["cubic"]["functional"] = poly_spline
    radial_basis_functions["cubic"]["order"] = 3.0
    radial_basis_functions["quintic"]["functional"] = poly_spline
    radial_basis_functions["quintic"]["order"] = 5.0
    radial_basis_functions["thin_plate"]["functional"] = thin_plate
    # radial basis function
    RBF = radial_basis_functions[method]["functional"]
    kwargs = {
        k: v
        for k, v in radial_basis_functions[method].items()
        if k != "functional"
    }
    # return functional and arguments
    return RBF, kwargs


# define radial basis function formulas
# polynomial (multiquadric, inverse multiquadric, etc)
def poly(epsilon, r, order=0.5, **kwargs):
    f = np.power((epsilon * r) ** 2 + 1.0, order)
    return f


# polyharmonic spline
def poly_spline(epsilon, r, order=1, **kwargs):
    f = np.power(epsilon * r, order)
    return f


# gaussian
def gaussian(epsilon, r, **kwargs):
    f = np.exp(-((epsilon * r) ** 2))
    return f


# thin plate spline
def thin_plate(epsilon, r, **kwargs):
    f = r**2 * np.log(r)
    # the spline is zero at zero
    f[r == 0] = 0.0
    return f


# calculate Euclidean distances between points as matrices
def distance_matrix(x, cntrs):
    s, M = np.shape(x)
    s, N = np.shape(cntrs)
    D = np.zeros((M, N))
    for d in range(s):
        (ii,) = np.dot(d, np.ones((1, N))).astype(np.int64)
        (jj,) = np.dot(d, np.ones((1, M))).astype(np.int64)
        dx = x[ii, :].T - cntrs[jj, :]
        D += dx**2
    D = np.sqrt(D)
    D[np.isnan(D)] = 0.0
    return D


# calculate polynomial matrix to augment radial basis functions
def polynomial_matrix(x, y, order):
    c = 0
    M = len(x)
    N = (order**2 + 3 * order) // 2 + 1
    P = np.zeros((M, N))
    for ii in range(order + 1):
        for jj in range(ii + 1):
            P[:, c] = (x**jj) * (y ** (ii - jj))
            c += 1
    return P
