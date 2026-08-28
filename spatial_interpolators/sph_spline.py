#!/usr/bin/env python
"""
sph_spline.py
Written by Tyler Sutterley (05/2022)

Interpolates a sparse grid over a sphere using spherical surface splines in
    tension following Wessel and Becker (2008)
Adapted from P. Wessel, SOEST, U of Hawaii, April 2008
Uses Generalized Legendre Function algorithm from Spanier and Oldman
    "An Atlas of Functions", 1987

CALLING SEQUENCE:
    output = sph_spline(lon, lat, data, longitude, latitude, tension=0)

INPUTS:
    lon: input longitude
    lat: input latitude
    data: input data
    longitude: output longitude
    latitude: output latitude

OUTPUTS:
    output: interpolated data

OPTIONS:
    tension: tension to use in interpolation (greater than 0)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    cython: C-extensions for Python
        http://cython.org/

REFERENCES:
    Wessel, P. and J. M. Becker, 2008, Interpolation using a generalized
        Green's function for a spherical surface spline in tension,
        Geophysical Journal International, doi:10.1111/j.1365-246X.2008.03829.x

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
    Updated 08/2016: using cythonized version of generalized Legendre function
        treat case for no tension but x is equal to 1 within machine precision
    Written 08/2016
"""

import numpy as np
import scipy.special
from spatial_interpolators.PvQv_C import PvQv_C


def sph_spline(
    lon,
    lat,
    data,
    longitude,
    latitude,
    tension=0.0,
):
    """
    Interpolates a sparse grid over a sphere using spherical
    surface splines in tension :cite:p:`Wessel:2008ea`

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
    tension: float, default 0.0
        tension to use in interpolation

    Returns
    -------
    output: float
        interpolated data grid
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

    # Check to make sure sizes of input arguments are correct and consistent
    if (len(data) != len(lon)) | (len(data) != len(lat)):
        raise Exception("Length of input arrays must be equal")
    if np.shape(longitude) != np.shape(latitude):
        raise Exception("Size of output Longitude and Latitude must be equal")
    if tension < 0:
        raise ValueError("tension must be greater than 0")

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
    sz = len(longitude.flatten())

    # Find and remove mean from data
    data_mean = data.mean()
    data_range = data.max() - data.min()
    # Normalize data
    data_norm = (data - data_mean) / data_range

    # compute linear system
    N = len(data)
    GG = np.zeros((N, N))
    for i in range(N):
        Rd = np.dot(np.c_[xs, ys, zs], np.array([xs[i], ys[i], zs[i]]))
        # remove singleton dimensions and calculate spherical surface splines
        GG[i, :] = SSST(Rd, P=tension)

    # Compute model m for normalized data
    m = np.linalg.lstsq(GG, data_norm, rcond=-1)[0]

    # calculate output interpolated array (or matrix)
    output = np.zeros((sz))
    for j in range(sz):
        Re = np.dot(np.c_[xs, ys, zs], np.array([XI[j], YI[j], ZI[j]]))
        # remove singleton dimensions and calculate spherical surface splines
        gg = SSST(Re, P=tension)
        output[j] = data_mean + data_range * np.dot(gg, m)

    # reshape output to original dimensions and return
    if np.ndim(longitude) > 1:
        output = output.reshape(nlon, nlat)

    return output


# SSST: Spherical Surface Spline in Tension
# Returns the Green's function for a spherical surface spline in tension,
#     following Wessel and Becker [2008].
# If p == 0 or not given then use minimum curvature solution with dilogarithm
def SSST(x, P=0):
    """
    Calculates the Green's function for a
        spherical surface spline in tension

    Parameters
    ----------
    x: float
        distance between points
    P: float or int, default 0
        Tension parameter

    Returns
    -------
    y: float
        Green's function
    """
    # floating point machine precision
    eps = np.finfo(np.float64).eps
    if P == 0:
        # use dilogarithm (Spence's function) if using splines without tension
        y = np.zeros_like(x)
        if np.any(np.abs(x) < (1.0 - eps)):
            (k,) = np.nonzero(np.abs(x) < (1.0 - eps))
            y[k] = scipy.special.spence(0.5 - 0.5 * x[k])
        # Deal with special cases x == +/- 1
        if np.any(((x + eps) >= 1.0) | ((x - eps) <= -1.0)):
            (k,) = np.nonzero(((x + eps) >= 1.0) | ((x - eps) <= -1.0))
            y[k] = scipy.special.spence(0.5 - 0.5 * np.sign(x[k]))
    else:
        # if in tension
        # calculate tension parameter
        v = (-1.0 + np.lib.scimath.sqrt(1.0 - 4.0 * P**2)) / 2.0
        # Initialize output array
        y = np.zeros_like(x, dtype=v.dtype)
        A = np.pi / np.sin(v * np.pi)
        # Where Pv solution works
        if np.any(np.abs(x) < (1.0 - eps)):
            (k,) = np.nonzero(np.abs(x) < (1.0 - eps))
            y[k] = A * Pv(-x[k], v) - np.log(1.0 - x[k])
        # Approximations where x is close to -1 or 1 using values from
        # "An Atlas of Functions" by Spanier and Oldham, 1987 (590)
        # Deal with special case x == -1
        if np.any((x - eps) <= -1.0):
            (k,) = np.nonzero((x - eps) <= -1.0)
            y[k] = A - np.log(2.0)
        # Deal with special case x == +1
        if np.any((x + eps) >= 1.0):
            (k,) = np.nonzero((x + eps) >= 1.0)
            y[k] = (
                np.pi * (1.0 / np.tan(v * np.pi))
                + 2.0 * (np.euler_gamma + scipy.special.psi(1.0 + v))
                - np.log(2.0)
            )
        # use only the real part (remove insignificant imaginary noise)
        y = np.real(y)
    # return the Green's function
    return y


# Calculate Legendre function of the first kind for arbitrary degree v
def Pv(x, v, method=PvQv_C):
    """Calculate Legendre function of the first kind"""
    P = np.zeros_like(x, dtype=v.dtype)
    for i, val in enumerate(x):
        if val == -1:
            p = np.inf
        else:
            # use compiled Cython version of PvQv as default
            p, q, k = method(val, v)
        P[i] = p
    return P


# Calculate generalized Legendre functions of arbitrary degree v
# Based on recipe in "An Atlas of Functions" by Spanier and Oldham, 1987 (589)
# Pv is the Legendre function of the first kind
# Qv is the Legendre function of the second kind
def PvQv(x, v):
    """Calculate Generalized Legendre function of arbitrary degree"""
    iter = 0
    if x == -1:
        P = -np.inf
        Q = -np.inf
    elif x == +1:
        P = 1.0
        Q = np.inf
    else:
        # set a and R to 1
        a = 1.0
        R = 1.0
        K = 4.0 * np.sqrt(np.abs(v - v**2))
        if (np.abs(1 + v) + np.floor(1 + v.real)) == 0:
            a = 1.0e99
            v = -1.0 - v
        # s and c = sin and cos of (pi*v/2.0)
        s = np.sin(0.5 * np.pi * v)
        c = np.cos(0.5 * np.pi * v)
        w = (0.5 + v) ** 2
        # if v is less than or equal to six (repeat until greater than six)
        while v.real <= 6.0:
            v += 2
            R = R * (v - 1.0) / v
        # calculate X and g and update R
        X = 1.0 / (4.0 + 4.0 * v)
        g = 1.0 + 5 * X * (1.0 - 3.0 * X * (0.35 + 6.1 * X))
        R = R * (1.0 - X * (1.0 - g * X / 2)) / np.sqrt(8.0 * X)
        # set g and u to 2.0*x
        g = 2.0 * x
        u = 2.0 * x
        # set f and t to 1
        f = 1.0
        t = 1.0
        # set k to 1/2
        k = 0.5
        # calculate new X
        X = 1.0 + (1e8 / (1.0 - x**2))
        # update t
        t = t * x**2 * (k**2.0 - w) / ((k + 1.0) ** 2 - 0.25)
        # add 1 to k
        k += 1.0
        # add t to f
        f += t
        # update u
        u = u * x**2 * (k**2 - w) / ((k + 1) ** 2 - 0.25)
        # add 1 to k
        k += 1.0
        # add u to g
        g += u
        # if k is less than K and |Xt| is greater than |f|
        # repeat previous set of operations until valid
        while (k < K) | (np.abs(X * t) > np.abs(f)):
            iter += 1
            t = t * x**2 * (k**2.0 - w) / ((k + 1.0) ** 2 - 0.25)
            k += 1.0
            f += t
            u = u * x**2 * (k**2.0 - w) / ((k + 1.0) ** 2 - 0.25)
            k += 1.0
            g += u
        # update f and g
        f += x**2 * t / (1.0 - x**2)
        g += x**2 * u / (1.0 - x**2)
        # calculate generalized Legendre functions
        P = ((s * g * R) + (c * f / R)) / np.sqrt(np.pi)
        Q = a * np.sqrt(np.pi) * ((c * g * R) - (s * f / R)) / 2.0
    # return P, Q and number of iterations
    return (P, Q, iter)
