#!/usr/bin/env python
"""
compact_radial_basis.py
Written by Tyler Sutterley (05/2022)

Interpolates data using compactly supported radial basis functions
    of minimal degree (Wendland functions) and sparse matrix algebra

    Wendland functions have the form
        p(r)    if 0 <= r <= 1
        0        if r > 1
    where p represents a univariate polynomial

CALLING SEQUENCE:
    ZI = compact_radial_basis(xs, ys, zs, XI, YI, dimension, order,
        smooth=smooth, radius=radius, method='wendland')

INPUTS:
    xs: scaled input X data
    ys: scaled input Y data
    zs: input data
    XI: scaled grid X for output ZI
    YI: scaled grid Y for output ZI
    dimension: spatial dimension of Wendland function (d)
    order: smoothness order of Wendland function (k)

OUTPUTS:
    ZI: interpolated data grid

OPTIONS:
    smooth: smoothing weights
    radius: scaling factor for the basis function (the radius of the
        support of the function)
    method: compactly supported radial basis function
        buhmann (not yet implemented)
        wendland (default)
        wu (not yet implemented)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

REFERENCES:
    Holger Wendland, "Piecewise polynomial, positive definite and compactly
        supported radial functions of minimal degree." Advances in
        Computational Mathematics, 1995.
    Holger Wendland, "Scattered Data Approximation", Cambridge Monographs on
        Applied and Computational Mathematics, 2005.
    Martin Buhmann, "Radial Basis Functions", Cambridge Monographs on
        Applied and Computational Mathematics, 2003.

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Updated 02/2019: compatibility updates for python3
    Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
    Updated 08/2016: using format text within ValueError, edit constant vector
        removed 3 dimensional option of radial basis (spherical)
        changed hierarchical_radial_basis to compact_radial_basis using
            compactly-supported radial basis functions and sparse matrices
        added low-order polynomial option (previously used default constant)
    Updated 01/2016: new hierarchical_radial_basis function
        that first reduces to points within distance.  added cutoff option
    Updated 10/2014: added third dimension (spherical)
    Written 08/2014
"""

from __future__ import print_function, division
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial


def compact_radial_basis(
    xs,
    ys,
    zs,
    XI,
    YI,
    dimension,
    order,
    smooth=0.0,
    radius=None,
    method="wendland",
):
    """
    Interpolates a sparse grid using compactly supported radial basis
    functions of minimal degree and sparse matrix algebra
    :cite:p:`Buhmann:2003cc,Wendland:1995gd,Wendland:2004hv`

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
    dimension: int
        spatial dimension of Wendland function (d)
    order: int
        smoothness order of Wendland function (k)
    smooth: float, default 0.0
        smoothing weights
    radius: float or NoneType, default None
        scaling factor for the basis function
    method: str, default `wendland`
        compactly supported radial basis function

        * ``'wendland'``

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
        raise Exception("Shape of output arrays must be equal")

    # create python dictionary of compact radial basis function formulas
    radial_basis_functions = {}
    # radial_basis_functions['buhmann'] = buhmann
    radial_basis_functions["wendland"] = wendland
    # radial_basis_functions['wu'] = wu
    # check if formula name is listed
    if method in radial_basis_functions.keys():
        cRBF = radial_basis_functions[method]
    else:
        raise ValueError(f"Method {method} not implemented")

    # construct kd-tree for Data points
    kdtree = scipy.spatial.KDTree(np.c_[xs, ys])
    if radius is None:
        # quick nearest-neighbor lookup to calculate mean radius
        ds, _ = kdtree.query(np.c_[xs, ys], k=2)
        radius = 2.0 * np.mean(ds[:, 1])

    # Creation of data-data distance sparse matrix in COOrdinate format
    Rd = kdtree.sparse_distance_matrix(kdtree, radius, output_type="coo_matrix")
    # calculate ratio between data-data distance and radius
    # replace cases where the data-data distance is greater than the radius
    r0 = np.where(Rd.data < radius, Rd.data / radius, radius / radius)
    # calculation of model PHI
    PHI = cRBF(r0, dimension, order)
    # construct sparse radial matrix
    PHI = scipy.sparse.coo_matrix((PHI, (Rd.row, Rd.col)), shape=Rd.shape)
    # Augmentation of the PHI Matrix with a smoothing factor
    if smooth != 0:
        # calculate eigenvalues of distance matrix
        eig = scipy.sparse.linalg.eigsh(
            Rd, k=1, which="LA", maxiter=1000, return_eigenvectors=False
        )[0]
        PHI += scipy.sparse.identity(len(xs), format="coo") * smooth * eig

    # Computation of the Weights
    w = scipy.sparse.linalg.spsolve(PHI, zs)

    # construct kd-tree for Mesh points
    # Data to Mesh Points
    mkdtree = scipy.spatial.KDTree(np.c_[XI.flatten(), YI.flatten()])
    # Creation of data-mesh distance sparse matrix in COOrdinate format
    Re = kdtree.sparse_distance_matrix(
        mkdtree, radius, output_type="coo_matrix"
    )
    # calculate ratio between data-mesh distance and radius
    # replace cases where the data-mesh distance is greater than the radius
    R0 = np.where(Re.data < radius, Re.data / radius, radius / radius)
    # calculation of the Evaluation Matrix
    E = cRBF(R0, dimension, order)
    # construct sparse radial matrix
    E = scipy.sparse.coo_matrix((E, (Re.row, Re.col)), shape=Re.shape)

    # calculate output interpolated array (or matrix)
    if np.ndim(XI) == 1:
        ZI = E.transpose().dot(w[:, np.newaxis])
    else:
        ZI = np.zeros((nx, ny))
        ZI[:, :] = E.transpose().dot(w[:, np.newaxis]).reshape(nx, ny)
    # return the interpolated array (or matrix)
    return ZI


# define compactly supported radial basis function formulas
def wendland(r, d, k):
    # Wendland functions of dimension d and order k
    # can replace with recursive method of Wendland for generalized case
    L = (d // 2) + k + 1
    if k == 0:
        f = (1.0 - r) ** L
    elif k == 1:
        f = (1.0 - r) ** (L + 1) * ((L + 1.0) * r + 1.0)
    elif k == 2:
        f = (1.0 - r) ** (L + 2) * (
            (L**2 + 4.0 * L + 3.0) * r**2 + (3.0 * L + 6.0) * r + 3.0
        )
    elif k == 3:
        f = (1.0 - r) ** (L + 3) * (
            (L**3 + 9.0 * L**2 + 23.0 * L + 15.0) * r**3
            + (6.0 * L**2 + 36.0 * L + 45.0) * r**2
            + (15.0 * L + 45.0) * r
            + 15.0
        )
    return f
