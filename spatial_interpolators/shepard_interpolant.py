#!/usr/bin/env python
"""
shepard_interpolant.py
Written by Tyler Sutterley (05/2022)

Evaluates Shepard interpolants to 2D data based on inverse distances
Resultant output will not be as accurate as radial basis functions

CALLING SEQUENCE:
    ZI = shepard_interpolant(xs, ys, zs, XI, YI)
    ZI = shepard_interpolant(xs, ys, zs, XI, YI, modified=True,
        D=25e3, L=500e3)

INPUTS:
    xs: input X data
    ys: input Y data
    zs: input data
    XI: grid X for output ZI
    YI: grid Y for output ZI

OUTPUTS:
    ZI: interpolated grid

OPTIONS:
    power: Power used in the inverse distance weighting (positive real number)
    eps: minimum distance value for valid points (default 1e-7)
    modified: use declustering modified Shepard's interpolants
    D: declustering distance
    L: maximum distance to be included in weights

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org

REFERENCES:
    D. Shepard, A two-dimensional interpolation function for irregularly
        spaced data, ACM68: Proceedings of the 1968 23rd ACM National
        Conference
    Schnell et al., Skill in forecasting extreme ozone pollution episodes with
        a global atmospheric chemistry model. Atmos. Chem. Phys., 14,
        7721-7739, doi:10.5194/acp-14-7721-2014, 2014

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Updated 09/2016: added modified Shepard's interpolants for declustering
        following Schnell et al (2014)
    Written 08/2016
"""

import numpy as np


def shepard_interpolant(
    xs, ys, zs, XI, YI, power=0.0, eps=1e-7, modified=False, D=25e3, L=500e3
):
    """
    Evaluates Shepard interpolants to 2D data based on
    inverse distance weighting :cite:p:`Shepard:1968ef`
    :cite:p:`Schnell:2014if`

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
    power: float, default 0.0
        Power used in the inverse distance weighting
    eps: float, default 1e-7
        minimum distance value for valid points
    modified: bool, default False
        use declustering modified Shepard's interpolants
    D: float, default 25e3
        declustering distance
    L: float, default 500e3
        maximum distance to be included in weights

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
    # number of data points
    npts = len(zs)
    # size of new matrix
    if np.ndim(XI) == 1:
        ni = len(XI)
    else:
        nx, ny = np.shape(XI)
        ni = XI.size

    # Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception("Length of input arrays must be equal")
    if np.shape(XI) != np.shape(YI):
        raise Exception("Size of output arrays must be equal")
    if power < 0:
        raise ValueError("Power parameter must be positive")

    # Modified Shepard interpolants for declustering data
    if modified:
        # calculate number of points within distance D of each point
        M = np.zeros((npts))
        for i, XY in enumerate(zip(xs, ys)):
            # compute radial distance between data point and data coordinates
            Rd = np.sqrt((XY[0] - xs) ** 2 + (XY[1] - ys) ** 2)
            M[i] = np.count_nonzero(Rd <= D)

    # for each interpolated value
    ZI = np.zeros((ni))
    for i, XY in enumerate(zip(XI.flatten(), YI.flatten())):
        # compute the radial distance between point i and data coordinates
        Re = np.sqrt((XY[0] - xs) ** 2 + (XY[1] - ys) ** 2)
        # Modified Shepard interpolants for declustering data
        if modified:
            # calculate weights
            w = np.zeros((npts))
            # find indices of cases
            ind_D = np.nonzero(Re < D)
            ind_M = np.nonzero((Re >= D) & (Re < L))
            ind_L = np.nonzero((Re >= L))
            # declustering of close points (weighted equally)
            w[ind_D] = D ** (-power) / M[ind_D]
            # inverse distance weighting of mid-range points with scaling
            power_inverse_distance = Re[ind_M] ** (-power)
            w[ind_M] = power_inverse_distance / M[ind_M]
            # no weight of distant points
            w[ind_L] = 0.0
            # calculate sum of all weights
            s = np.sum(w)
            # Find 2D interpolated surface
            ZI[i] = np.dot(w / s, zs) if (s > 0.0) else np.nan
        elif (Re < eps).any():
            # if a data coordinate is within the EPS cutoff
            (min_indice,) = np.nonzero(Re < eps)
            ZI[i] = zs[min_indice]
        else:
            # compute the weights based on POWER
            if power == 0.0:
                # weights if POWER is 0
                w = np.ones((npts)) / npts
            else:
                # normalized weights if POWER > 0 (typically between 1 and 3)
                # in the inverse distance weighting
                power_inverse_distance = Re ** (-power)
                s = np.sum(power_inverse_distance)
                w = power_inverse_distance / s
            # Find 2D interpolated surface
            ZI[i] = np.dot(w, zs)

    # reshape to original dimensions
    if np.ndim(XI) != 1:
        ZI = ZI.reshape(nx, ny)
    # return output matrix/array
    return ZI
