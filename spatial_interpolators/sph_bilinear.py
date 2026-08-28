#!/usr/bin/env python
"""
sph_bilinear.py
Written by Tyler Sutterley (05/2022)

Interpolates data over a sphere using bilinear functions

CALLING SEQUENCE:
    zi = sph_bilinear(x, y, z, xi, yi)

INPUTS:
    x: input longitude
    y: input latitude
    z: input data (matrix)
    xi: output longitude
    yi: output latitude

OUTPUTS:
    zi: interpolated data

OPTIONS:
    flattened: input xi, yi are flattened arrays (nlon must equal nlat)
    fill_value: value to use if xi and yi are out of range

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Updated 09/2017: use minimum distances with FLATTENED method
        if indices are out of range: replace with FILL_VALUE
    Updated 03/2016: added FLATTENED option for regional grids to global grids
    Updated 11/2015: made easier to read with data and weight values
    Written 07/2013
"""

import numpy as np


def sph_bilinear(x, y, z, xi, yi, flattened=False, fill_value=-9999.0):
    """
    Spherical interpolation routine for gridded data using
    bilinear interpolation

    Parameters
    ----------
    x: float
        input longitude
    y: float
        input latitude
    z: float
        input data
    xi: float
        output longitude
    yi: float
        output latitude
    flattened: bool, default False
        input xi, yi are flattened arrays
    fill_value: float, default -9999.0
        value to use if xi and yi are out of range

    Returns
    -------
    zi: float
        interpolated data
    """

    # Converting input data into geodetic coordinates in radians
    phi = x * np.pi / 180.0
    th = (90.0 - y) * np.pi / 180.0
    # grid steps for lon and lat
    dlon = np.abs(x[1] - x[0])
    dlat = np.abs(y[1] - y[0])
    # grid steps in radians
    dphi = dlon * np.pi / 180.0
    dth = dlat * np.pi / 180.0
    # input data shape
    nx = len(x)
    ny = len(y)
    # Converting output data into geodetic coordinates in radians
    xphi = xi * np.pi / 180.0
    xth = (90.0 - yi) * np.pi / 180.0
    # check if using flattened array or two-dimensional lat/lon
    if flattened:
        # output array
        ndat = len(xi)
        zi = np.zeros((ndat))
        for i in range(0, ndat):
            # calculating the indices for the original grid
            dx = (x - np.floor(xi[i] / dlon) * dlon) ** 2
            dy = (y - np.floor(yi[i] / dlat) * dlat) ** 2
            iph = np.argmin(dx)
            ith = np.argmin(dy)
            # data is within range of values
            if ((iph + 1) < nx) & ((ith + 1) < ny):
                # corner data values for i,j
                Ia = z[iph, ith]  # (0,0)
                Ib = z[iph + 1, ith]  # (1,0)
                Ic = z[iph, ith + 1]  # (0,1)
                Id = z[iph + 1, ith + 1]  # (1,1)
                # corner weight values for i,j
                Wa = (xphi[i] - phi[iph]) * (xth[i] - th[ith])
                Wb = (phi[iph + 1] - xphi[i]) * (xth[i] - th[ith])
                Wc = (xphi[i] - phi[iph]) * (th[ith + 1] - xth[i])
                Wd = (phi[iph + 1] - xphi[i]) * (th[ith + 1] - xth[i])
                # divisor weight value
                W = (phi[iph + 1] - phi[iph]) * (th[ith + 1] - th[ith])
                # calculate interpolated value for i
                zi[i] = (Ia * Wa + Ib * Wb + Ic * Wc + Id * Wd) / W
            else:
                # replace with fill value
                zi[i] = fill_value
    else:
        # output grid
        nphi = len(xi)
        nth = len(yi)
        zi = np.zeros((nphi, nth))
        for i in range(0, nphi):
            for j in range(0, nth):
                # calculating the indices for the original grid
                iph = np.floor(xphi[i] / dphi)
                jth = np.floor(xth[j] / dth)
                # data is within range of values
                if ((iph + 1) < nx) & ((jth + 1) < ny):
                    # corner data values for i,j
                    Ia = z[iph, jth]  # (0,0)
                    Ib = z[iph + 1, jth]  # (1,0)
                    Ic = z[iph, jth + 1]  # (0,1)
                    Id = z[iph + 1, jth + 1]  # (1,1)
                    # corner weight values for i,j
                    Wa = (xphi[i] - phi[iph]) * (xth[j] - th[jth])
                    Wb = (phi[iph + 1] - xphi[i]) * (xth[j] - th[jth])
                    Wc = (xphi[i] - phi[iph]) * (th[jth + 1] - xth[j])
                    Wd = (phi[iph + 1] - xphi[i]) * (th[jth + 1] - xth[j])
                    # divisor weight value
                    W = (phi[iph + 1] - phi[iph]) * (th[jth + 1] - th[jth])
                    # calculate interpolated value for i,j
                    zi[i, j] = (Ia * Wa + Ib * Wb + Ic * Wc + Id * Wd) / W
                else:
                    # replace with fill value
                    zi[i, j] = fill_value

    # return the interpolated data
    return zi
