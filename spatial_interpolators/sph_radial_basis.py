#!/usr/bin/env python
"""
sph_radial_basis.py
Written by Tyler Sutterley (08/2026)

Interpolates data over a sphere using radial basis functions
    with QR factorization option to eliminate ill-conditioning

CALLING SEQUENCE:
    output = sph_radial_basis(lon, lat, data, longitude, latitude,
        smooth=smooth, epsilon=epsilon, method='inverse')

INPUTS:
    lon: input longitude
    lat: input latitude
    data: input data
    longitude: output longitude
    latitude: output latitude

OUTPUTS:
    output: interpolated data

OPTIONS:
    smooth: smoothing weights
    epsilon: adjustable constant for distance functions
        default is the mean Euclidean distance
    method: radial basis function (** has option for QR factorization method)
        multiquadric**
        inverse_multiquadric** or inverse** (default)
        inverse_quadratic**
        gaussian**
        linear
        cubic
        quintic
        thin_plate: thin-plate spline
    QR: use QR factorization algorithm of Fornberg (2007)
    norm: distance function for radial basis functions (if not using QR)
        euclidean: Euclidean Distance with distance_matrix (default)
        GCD: Great-Circle Distance using n-vectors with angle_matrix

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

REFERENCES:
    B Fornberg and C Piret, "A stable algorithm for flat radial basis functions
        on a sphere." SIAM J. Sci. Comput. 30(1), 60-80 (2007)
    B Fornberg, E Larsson, and N Flyer, "Stable Computations with Gaussian
        Radial Basis Functions." SIAM J. Sci. Comput. 33(2), 869-892 (2011)

UPDATE HISTORY:
    Updated 08/2026: fix case where floating point errors cause negative roots
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Updated 02/2019: compatibility updates for python3
    Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
    Updated 08/2016: finished QR factorization method, added norm option
    Forked 08/2016 from radial_basis.py for use over a sphere
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
import scipy.special as spc
from spatial_interpolators.legendre import legendre


def sph_radial_basis(
    lon,
    lat,
    data,
    longitude,
    latitude,
    smooth=0.0,
    epsilon=None,
    method="inverse",
    QR=False,
    norm="euclidean",
):
    """
    Interpolates a sparse grid over a sphere using radial basis
    functions with QR factorization option
    :cite:p:`Fornberg:2008kc,Fornberg:2011fw`

    Parameters
    ----------
    lon: float
        input longitude
    lat: float
        input latitude
    data: float
        input data
    longitude: float
        output longitude
    latitude: float
        output latitude
    smooth: float, default 0.0
        smoothing weights
    epsilon: float or NoneType, default None
        adjustable constant for distance functions
    method: str, default 'inverse'
        compactly supported radial basis function

        * ``'multiquadric'`` [#f1]_
        * ``'inverse_multiquadric'`` [#f1]_ or ``'inverse'`` [#f1]_
        * ``'inverse_quadratic'`` [#f1]_
        * ``'gaussian'`` [#f1]_
        * ``'linear'``
        * ``'cubic'``
        * ``'quintic'``
        * ``'thin_plate'``

    QR: bool, default False
        use QR factorization algorithm of :cite:p:`Fornberg:2008kc`
    norm: str, default 'euclidean'
        Distance function for radial basis functions

        * ``'euclidean'``: Euclidean Distance with distance_matrix
        * ``'GCD'``: Great-Circle Distance using n-vectors with angle_matrix

    Returns
    -------
    output: float
        interpolated data grid

    .. [#f1] has option for QR factorization method
    """

    # remove singleton dimensions
    lon = np.squeeze(lon)
    lat = np.squeeze(lat)
    data = np.squeeze(data)
    longitude = np.squeeze(longitude)
    latitude = np.squeeze(latitude)
    # size of new matrix
    if np.ndim(longitude) > 1:
        nlon, nlat = np.shape(longitude)
        sz = np.int64(nlon * nlat)
    else:
        sz = len(longitude)

    # Check to make sure sizes of input arguments are correct and consistent
    if (len(data) != len(lon)) | (len(data) != len(lat)):
        raise Exception("Length of input arrays must be equal")
    if np.shape(longitude) != np.shape(latitude):
        raise Exception("Size of output arrays must be equal")

    # create python dictionary of radial basis function formulas
    radial_basis_functions = {}
    radial_basis_functions["multiquadric"] = multiquadric
    radial_basis_functions["inverse_multiquadric"] = inverse_multiquadric
    radial_basis_functions["inverse"] = inverse_multiquadric
    radial_basis_functions["inverse_quadratic"] = inverse_quadratic
    radial_basis_functions["gaussian"] = gaussian
    radial_basis_functions["linear"] = linear
    radial_basis_functions["cubic"] = cubic
    radial_basis_functions["quintic"] = quintic
    radial_basis_functions["thin_plate"] = thin_plate
    # create python dictionary of radial basis function expansions
    radial_expansions = {}
    radial_expansions["multiquadric"] = multiquadratic_expansion
    radial_expansions["inverse_multiquadric"] = inverse_multiquadric_expansion
    radial_expansions["inverse"] = inverse_multiquadric_expansion
    radial_expansions["inverse_quadratic"] = inverse_quadratic_expansion
    radial_expansions["gaussian"] = gaussian_expansion
    # check if formula name is listed
    if method in radial_basis_functions.keys():
        RBF = radial_basis_functions[method]
    else:
        raise ValueError(f"Method {method} not implemented")
    # check if formula name is valid for QR factorization method
    if QR and (method in radial_expansions.keys()):
        expansion = radial_expansions[method]
    elif QR and (method not in radial_expansions.keys()):
        raise ValueError(f"{method} expansion not available with QR")
    # create python dictionary of distance functions (if not using QR)
    norm_functions = {}
    norm_functions["euclidean"] = distance_matrix
    norm_functions["GCD"] = angle_matrix
    if norm in norm_functions:
        norm_matrix = norm_functions[norm]
    else:
        raise ValueError(f"Distance Function {norm} not implemented")

    # convert input lat and lon into cartesian X,Y,Z over unit sphere
    phi = np.pi * lon / 180.0
    th = np.pi * (90.0 - lat) / 180.0
    xs = np.sin(th) * np.cos(phi)
    ys = np.sin(th) * np.sin(phi)
    zs = np.cos(th)
    # convert output longitude and latitude into cartesian X,Y,Z
    PHI = np.pi * longitude.flatten() / 180.0
    THETA = np.pi * (90.0 - latitude.flatten()) / 180.0
    XI = np.sin(THETA) * np.cos(PHI)
    YI = np.sin(THETA) * np.sin(PHI)
    ZI = np.cos(THETA)

    # Creation of data distance matrix (Euclidean or Great-Circle Distance)
    # Data to Data
    Rd = norm_matrix(np.array([xs, ys, zs]), np.array([xs, ys, zs]))
    N, M = np.shape(Rd)
    # if epsilon is not specified
    if epsilon is None:
        # calculate norm with mean distance
        uix, uiy = np.nonzero(np.tri(N, M=M, k=-1))
        epsilon = np.mean(Rd[uix, uiy])

    # QR factorization algorithm of Fornberg (2007)
    if QR:
        # calculate radial basis functions using spherical harmonics
        R, w = RBF_QR(th, phi, epsilon, data, expansion)
        n_harm = np.sqrt(np.shape(R)[0]).astype(np.int64)
        # counter variable for filling spherical harmonic matrix
        index = 0
        # evaluation matrix E
        E = np.zeros((sz, np.int64(n_harm**2)))
        for l in range(0, n_harm):
            # Each loop adds a block of columns of degree l to E
            E[:, index : 2 * l + index + 1] = spherical_harmonic_matrix(
                l, THETA, PHI
            )
            index += 2 * l + 1
        # calculate output interpolated array (or matrix)
        output = np.dot(E, np.dot(R, w))
    else:
        # Calculation of the PHI Matrix with smoothing
        PHI = np.zeros((N + 1, M + 1))
        PHI[:N, :M] = RBF(epsilon, Rd) + np.eye(N, M=M) * smooth
        # Augmentation of the PHI Matrix with a Constant Vector
        PHI[:N, M] = np.ones((N))
        PHI[N, :M] = np.ones((M))

        # Computation of the Weights
        DMAT = np.concatenate(([data, [0]]), axis=0)
        w = np.linalg.lstsq(PHI, DMAT[:, np.newaxis], rcond=-1)[0]

        # Computation of distance Matrix (Euclidean or Great-Circle Distance)
        # Data to Mesh Points
        Re = norm_matrix(np.array([XI, YI, ZI]), np.array([xs, ys, zs]))
        # calculate radial basis function for data-to-mesh matrix
        E = RBF(epsilon, Re)

        # Augmentation of the Evaluation Matrix with a Constant Vector
        P = np.ones((sz, 1))
        E = np.concatenate(([E, P]), axis=1)
        # calculate output interpolated array (or matrix)
        output = np.dot(E, w)

    # reshape output to original dimensions and return
    if np.ndim(longitude) == 1:
        return np.squeeze(output)
    else:
        return output.reshape(nlon, nlat)


# define radial basis function formulas
def multiquadric(epsilon, r):
    # multiquadratic
    f = np.sqrt((epsilon * r) ** 2 + 1.0)
    return f


def multiquadratic_expansion(epsilon, mu):
    c = (
        -2.0
        * np.pi
        * (
            2.0 * epsilon**2
            + 1.0
            + (mu + 1.0 / 2.0) * np.sqrt(1.0 + 4.0 * epsilon**2)
        )
        / (mu + 1.0 / 2.0)
        / (mu + 3.0 / 2.0)
        / (mu - 1.0 / 2.0)
        * (2.0 / (1.0 + np.sqrt(4.0 * epsilon**2 + 1.0))) ** (2.0 * mu + 1.0)
    )
    return c


def inverse_multiquadric(epsilon, r):
    # inverse multiquadratic
    f = 1.0 / np.sqrt((epsilon * r) ** 2 + 1.0)
    return f


def inverse_multiquadric_expansion(epsilon, mu):
    c = (
        4.0
        * np.pi
        / (mu + 1.0 / 2.0)
        * (2.0 / (1.0 + np.sqrt(4.0 * epsilon**2 + 1.0))) ** (2 * mu + 1.0)
    )
    return c


def inverse_quadratic(epsilon, r):
    # inverse quadratic
    f = 1.0 / (1.0 + (epsilon * r) ** 2)
    return f


def inverse_quadratic_expansion(epsilon, mu):
    c = (
        4.0
        * np.pi ** (3.0 / 2.0)
        * spc.factorial(mu)
        / spc.gamma(mu + 3.0 / 2.0)
        / (1.0 + 4.0 * epsilon**2) ** (mu + 1)
        * spc.hyp2f1(
            mu + 1,
            mu + 1,
            2.0 * mu + 2,
            4.0 * epsilon**2 / (1.0 + 4.0 * epsilon**2),
        )
    )
    return c


def gaussian(epsilon, r):
    # gaussian
    f = np.exp(-((epsilon * r) ** 2))
    return f


def gaussian_expansion(epsilon, mu):
    c = (
        4.0
        * np.pi ** (3.0 / 2.0)
        * np.exp(-2.0 * epsilon**2)
        * spc.iv(mu + 1.0 / 2.0, 2.0 * epsilon**2)
        / epsilon ** (2.0 * mu + 1.0)
    )
    return c


def linear(epsilon, r):
    # linear polynomial
    return r


def cubic(epsilon, r):
    # cubic polynomial
    f = r**3
    return f


def quintic(epsilon, r):
    # quintic polynomial
    f = r**5
    return f


def thin_plate(epsilon, r):
    # thin plate spline
    f = r**2 * np.log(r)
    # the spline is zero at zero
    f[r == 0] = 0.0
    return f


# calculate great-circle distance between between n-vectors
def angle_matrix(x, cntrs):
    s, M = np.shape(x)
    s, N = np.shape(cntrs)
    A = np.zeros((M, N))
    A[:, :] = np.arccos(np.dot(x.transpose(), cntrs))
    A[np.isnan(A)] = 0.0
    return A


# calculate Euclidean distances between points (default norm)
def distance_matrix(x, cntrs):
    s, M = np.shape(x)
    s, N = np.shape(cntrs)
    # decompose Euclidean distance: (x-y)^2 = x^2 - 2xy + y^2
    dx2 = np.kron(np.ones((1, N)), np.sum(x * x, axis=0)[:, np.newaxis])
    dxy = 2.0 * np.dot(x.transpose(), cntrs)
    dy2 = np.kron(np.ones((M, 1)), np.sum(cntrs * cntrs, axis=0))
    D = np.sqrt(dx2 - dxy + dy2)
    D[np.isnan(D)] = 0.0
    return D


# calculate spherical harmonics of degree l evaluated at (theta,phi)
def spherical_harmonic_matrix(l, theta, phi):
    # calculate legendre polynomials
    nth = len(theta)
    Pl = legendre(l, np.cos(theta)).transpose()
    # calculate degree dependent factors C and F
    m = np.arange(0, l + 1)  # spherical harmonic orders up to degree l
    C = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    F = np.sqrt(spc.factorial(1 + l - m - 1) / spc.factorial(1 + l + m - 1))
    F = np.kron(np.ones((nth, 1)), F[np.newaxis, :])
    # calculate Euler's of spherical harmonic order multiplied by azimuth phi
    mphi = np.exp(1j * np.dot(np.squeeze(phi)[:, np.newaxis], m[np.newaxis, :]))
    # calculate spherical harmonics
    Ylms = F * Pl[:, 0 : l + 1] * mphi
    # multiply by C and convert to reduced matrix (theta,Slm:Clm)
    SPH = C * np.concatenate((np.imag(Ylms[:, :0:-1]), np.real(Ylms)), axis=1)
    return SPH


# RBF interpolant with shape parameter epsilon through the node points
# (theta,phi) with function values f from Fornberg
# Outputs beta: the expansion coefficients of the interpolant with respect to
# the RBF_QR basis.
def RBF_QR(theta, phi, epsilon, data, RBF):
    n = len(phi)
    Y1 = np.zeros((n, n))
    B1 = np.zeros((n, n))
    # counter variable for filling spherical harmonic matrix
    index = 0
    # difference adding the next spherical harmonic degree
    d = 0.0
    # degree of the n_th spherical harmonic
    l = 0
    l_n = np.ceil(np.sqrt(n)) - 1
    # floating point machine precision
    eps = np.finfo(np.float64).eps
    while d < -np.log10(eps):
        # create new variables for Y and B which will resize if (l > (l_n -1))
        lmax = np.max([l_n, l])
        Y = np.zeros((n, int((lmax + 1) ** 2)))
        Y[:, :index] = Y1[:, :index]
        B = np.zeros((n, int((lmax + 1) ** 2)))
        B[:, :index] = B1[:, :index]
        # Each loop adds a block of columns of SPH of degree l to Y and to B.
        # Compute the spherical harmonics matrix
        Y[:, index : 2 * l + index + 1] = spherical_harmonic_matrix(
            l, theta, phi
        )
        # Compute the expansion coefficients matrix
        B[:, index : 2 * l + index + 1] = Y[:, index : 2 * l + index + 1] * RBF(
            epsilon, l
        )
        B[:, index + l] = B[:, index + l] / 2.0
        # Truncation criterion
        if l > (l_n - 1):
            dN1 = np.linalg.norm(
                B[:, int(l_n**2) : int((l_n + 1) ** 2)], ord=np.inf
            )
            dN2 = np.linalg.norm(B[:, int((l + 1) ** 2) - 1], ord=np.inf)
            d = np.log10(dN1 / dN2 * epsilon ** (2 * (l_n - l)))
        # copy B to B1 and Y to Y1
        B1 = B.copy()
        Y1 = Y.copy()
        # Calculate column index of next block
        index += 2 * l + 1
        l += 1
    # QR-factorization to find the RBF_QR basis
    Q, R = np.linalg.qr(B)
    # Introduce the powers of epsilon
    X1 = np.kron(np.ones((n, 1)), np.ceil(np.sqrt(np.arange(n, l**2))))
    X2 = np.kron(
        np.ones((1, l**2 - n)),
        (np.ceil(np.sqrt(np.arange(1, n + 1))) - 1)[:, np.newaxis],
    )
    E = epsilon ** (2.0 * (X1 - X2))
    # Solve the interpolation linear system
    R_beta = E * np.linalg.lstsq(R[:n, :n], R[:n, n:], rcond=-1)[0]
    R_new = np.concatenate((np.eye(n), R_beta.T), axis=0)
    w = np.linalg.lstsq(np.dot(Y, R_new), data, rcond=-1)[0]
    return (R_new, w)
