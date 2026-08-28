#!/usr/bin/env python
"""
spatial.py
Written by Tyler Sutterley (11/2024)

Utilities for operating on spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

UPDATE HISTORY:
    Updated 11/2024: added function to calculate the altitude and azimuth
    Updated 08/2024: added functions to convert to and from East-North-Up
    Updated 07/2024: added functions to convert to and from DMS
    Updated 03/2024: can calculate polar stereographic distortion for distances
    Updated 04/2023: copy inputs in cartesian to not modify original arrays
        added iterative methods for converting from cartesian to geodetic
    Updated 03/2023: add basic variable typing to function inputs
    Updated 04/2022: docstrings in numpy documentation format
    Updated 03/2022: add option to specify output GDAL driver
    Updated 01/2022: use iteration breaks in convert ellipsoid function
    Updated 10/2021: add pole case in stereographic area scale calculation
    Updated 09/2021: can calculate height differences between ellipsoids
    Updated 07/2021: added function for determining input variable type
    Updated 03/2021: added polar stereographic area scale calculation
        add routines for converting to and from cartesian coordinates
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: added module for converting ellipsoids
    Written 11/2020
"""

from __future__ import annotations

import warnings
import numpy as np

# suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def data_type(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> str:
    """
    Determines input data type based on variable dimensions

    Parameters
    ----------
    x: np.ndarray
        x-dimension coordinates
    y: np.ndarray
        y-dimension coordinates
    t: np.ndarray
        time-dimension coordinates

    Returns
    -------
    string denoting input data type

        - ``'time series'``
        - ``'drift'``
        - ``'grid'``
    """
    xsize = np.size(x)
    ysize = np.size(y)
    tsize = np.size(t)
    if (xsize == 1) and (ysize == 1) and (tsize >= 1):
        return "time series"
    elif (xsize == ysize) & (xsize == tsize):
        return "drift"
    elif (np.ndim(x) > 1) & (xsize == ysize):
        return "grid"
    elif xsize != ysize:
        return "grid"
    else:
        raise ValueError("Unknown data type")


def convert_ellipsoid(
    phi1: np.ndarray,
    h1: np.ndarray,
    a1: float,
    f1: float,
    a2: float,
    f2: float,
    eps: float = 1e-12,
    itmax: int = 10,
):
    """
    Convert latitudes and heights to a different ellipsoid using
    Newton-Raphson :cite:p:`Meeus:1991vh`

    Parameters
    ----------
    phi1: np.ndarray
        latitude of input ellipsoid in degrees
    h1: np.ndarray
        height above input ellipsoid in meters
    a1: float
        semi-major axis of input ellipsoid
    f1: float
        flattening of input ellipsoid
    a2: float
        semi-major axis of output ellipsoid
    f2: float
        flattening of output ellipsoid
    eps: float, default 1e-12
        tolerance to prevent division by small numbers and
        to determine convergence
    itmax: int, default 10
        maximum number of iterations to use in Newton-Raphson

    Returns
    -------
    phi2: np.ndarray
        latitude of output ellipsoid in degrees
    h2: np.ndarray
        height above output ellipsoid in meters
    """
    if len(phi1) != len(h1):
        raise ValueError("phi and h have incompatible dimensions")
    # semiminor axis of input and output ellipsoid
    b1 = (1.0 - f1) * a1
    b2 = (1.0 - f2) * a2
    # initialize output arrays
    npts = len(phi1)
    phi2 = np.zeros((npts))
    h2 = np.zeros((npts))
    # for each point
    for N in range(npts):
        # force phi1 into range -90 <= phi1 <= 90
        if np.abs(phi1[N]) > 90.0:
            phi1[N] = np.sign(phi1[N]) * 90.0
        # handle special case near the equator
        # phi2 = phi1 (latitudes congruent)
        # h2 = h1 + a1 - a2
        if np.abs(phi1[N]) < eps:
            phi2[N] = np.copy(phi1[N])
            h2[N] = h1[N] + a1 - a2
        # handle special case near the poles
        # phi2 = phi1 (latitudes congruent)
        # h2 = h1 + b1 - b2
        elif (90.0 - np.abs(phi1[N])) < eps:
            phi2[N] = np.copy(phi1[N])
            h2[N] = h1[N] + b1 - b2
        # handle case if latitude is within 45 degrees of equator
        elif np.abs(phi1[N]) <= 45:
            # convert phi1 to radians
            phi1r = phi1[N] * np.pi / 180.0
            sinphi1 = np.sin(phi1r)
            cosphi1 = np.cos(phi1r)
            # prevent division by very small numbers
            cosphi1 = np.copy(eps) if (cosphi1 < eps) else cosphi1
            # calculate tangent
            tanphi1 = sinphi1 / cosphi1
            u1 = np.arctan(b1 / a1 * tanphi1)
            hpr1sin = b1 * np.sin(u1) + h1[N] * sinphi1
            hpr1cos = a1 * np.cos(u1) + h1[N] * cosphi1
            # set initial value for u2
            u2 = np.copy(u1)
            # setup constants
            k0 = b2 * b2 - a2 * a2
            k1 = a2 * hpr1cos
            k2 = b2 * hpr1sin
            # perform newton-raphson iteration to solve for u2
            # cos(u2) will not be close to zero since abs(phi1) <= 45
            for i in range(0, itmax + 1):
                cosu2 = np.cos(u2)
                fu2 = k0 * np.sin(u2) + k1 * np.tan(u2) - k2
                fu2p = k0 * cosu2 + k1 / (cosu2 * cosu2)
                if np.abs(fu2p) < eps:
                    break
                else:
                    delta = fu2 / fu2p
                    u2 -= delta
                    if np.abs(delta) < eps:
                        break
            # convert latitude to degrees and verify values between +/- 90
            phi2r = np.arctan(a2 / b2 * np.tan(u2))
            phi2[N] = phi2r * 180.0 / np.pi
            if np.abs(phi2[N]) > 90.0:
                phi2[N] = np.sign(phi2[N]) * 90.0
            # calculate height
            h2[N] = (hpr1cos - a2 * np.cos(u2)) / np.cos(phi2r)
        # handle final case where latitudes are between 45 degrees and pole
        else:
            # convert phi1 to radians
            phi1r = phi1[N] * np.pi / 180.0
            sinphi1 = np.sin(phi1r)
            cosphi1 = np.cos(phi1r)
            # prevent division by very small numbers
            cosphi1 = np.copy(eps) if (cosphi1 < eps) else cosphi1
            # calculate tangent
            tanphi1 = sinphi1 / cosphi1
            u1 = np.arctan(b1 / a1 * tanphi1)
            hpr1sin = b1 * np.sin(u1) + h1[N] * sinphi1
            hpr1cos = a1 * np.cos(u1) + h1[N] * cosphi1
            # set initial value for u2
            u2 = np.copy(u1)
            # setup constants
            k0 = a2 * a2 - b2 * b2
            k1 = b2 * hpr1sin
            k2 = a2 * hpr1cos
            # perform newton-raphson iteration to solve for u2
            # sin(u2) will not be close to zero since abs(phi1) > 45
            for i in range(0, itmax + 1):
                sinu2 = np.sin(u2)
                fu2 = k0 * np.cos(u2) + k1 / np.tan(u2) - k2
                fu2p = -1 * (k0 * sinu2 + k1 / (sinu2 * sinu2))
                if np.abs(fu2p) < eps:
                    break
                else:
                    delta = fu2 / fu2p
                    u2 -= delta
                    if np.abs(delta) < eps:
                        break
            # convert latitude to degrees and verify values between +/- 90
            phi2r = np.arctan(a2 / b2 * np.tan(u2))
            phi2[N] = phi2r * 180.0 / np.pi
            if np.abs(phi2[N]) > 90.0:
                phi2[N] = np.sign(phi2[N]) * 90.0
            # calculate height
            h2[N] = (hpr1sin - b2 * np.sin(u2)) / np.sin(phi2r)

    # return the latitude and height
    return (phi2, h2)


def compute_delta_h(
    lat: np.ndarray, a1: float, f1: float, a2: float, f2: float
):
    """
    Compute difference in elevation for two ellipsoids at a given
    latitude using a simplified empirical relation :cite:p:`Meeus:1991vh`

    Parameters
    ----------
    lat: np.ndarray
        latitudes (degrees north)
    a1: float
        semi-major axis of input ellipsoid
    f1: float
        flattening of input ellipsoid
    a2: float
        semi-major axis of output ellipsoid
    f2: float
        flattening of output ellipsoid

    Returns
    -------
    delta_h: np.ndarray
        difference in elevation for two ellipsoids
    """
    # force latitudes to be within -90 to 90 and convert to radians
    phi = np.clip(lat, -90.0, 90.0) * np.pi / 180.0
    # semi-minor axis of input and output ellipsoid
    b1 = (1.0 - f1) * a1
    b2 = (1.0 - f2) * a2
    # compute differences in semi-major and semi-minor axes
    delta_a = a2 - a1
    delta_b = b2 - b1
    # compute differences between ellipsoids
    # delta_h = -(delta_a * cos(phi)^2 + delta_b * sin(phi)^2)
    delta_h = -(delta_a * np.cos(phi) ** 2 + delta_b * np.sin(phi) ** 2)
    return delta_h


def wrap_longitudes(lon: float | np.ndarray):
    """
    Wraps longitudes to range from -180 to +180

    Parameters
    ----------
    lon: float or np.ndarray
        longitude (degrees east)
    """
    phi = np.arctan2(np.sin(lon * np.pi / 180.0), np.cos(lon * np.pi / 180.0))
    # convert phi from radians to degrees
    return phi * 180.0 / np.pi


def to_dms(d: np.ndarray):
    """
    Convert decimal degrees to degrees, minutes and seconds

    Parameters
    ----------
    d: np.ndarray
        decimal degrees

    Returns
    -------
    degree: np.ndarray
        degrees
    minute: np.ndarray
        minutes (arcminutes)
    second: np.ndarray
        seconds (arcseconds)
    """
    sign = np.sign(d)
    minute, second = np.divmod(np.abs(d) * 3600.0, 60.0)
    degree, minute = np.divmod(minute, 60.0)
    return (sign * degree, minute, second)


def from_dms(degree: np.ndarray, minute: np.ndarray, second: np.ndarray):
    """
    Convert degrees, minutes and seconds to decimal degrees

    Parameters
    ----------
    degree: np.ndarray
        degrees
    minute: np.ndarray
        minutes (arcminutes)
    second: np.ndarray
        seconds (arcseconds)

    Returns
    -------
    d: np.ndarray
        decimal degrees
    """
    sign = np.sign(degree)
    d = np.abs(degree) + minute / 60.0 + second / 3600.0
    return sign * d


def to_cartesian(
    lon: np.ndarray,
    lat: np.ndarray,
    h: float | np.ndarray = 0.0,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
):
    """
    Converts geodetic coordinates to Cartesian coordinates

    Parameters
    ----------
    lon: np.ndarray
        longitude (degrees east)
    lat: np.ndarray
        latitude (degrees north)
    h: float or np.ndarray, default 0.0
        height above ellipsoid (or sphere)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid

        for spherical coordinates set to radius of the Earth
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

        for spherical coordinates set to 0
    """
    # verify axes and copy to not modify inputs
    singular_values = np.ndim(lon) == 0
    lon = np.atleast_1d(np.copy(lon)).astype(np.float64)
    lat = np.atleast_1d(np.copy(lat)).astype(np.float64)
    # fix coordinates to be 0:360
    lon = np.where(lon < 0, lon + 360.0, lon)
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0 * flat - flat**2) * a_axis**2)
    ecc1 = lin_ecc / a_axis
    # convert from geodetic latitude to geocentric latitude
    dtr = np.pi / 180.0
    # geodetic latitude in radians
    latitude_geodetic_rad = lat * dtr
    # prime vertical radius of curvature
    N = a_axis / np.sqrt(1.0 - ecc1**2.0 * np.sin(latitude_geodetic_rad) ** 2.0)
    # calculate X, Y and Z from geodetic latitude and longitude
    X = (N + h) * np.cos(latitude_geodetic_rad) * np.cos(lon * dtr)
    Y = (N + h) * np.cos(latitude_geodetic_rad) * np.sin(lon * dtr)
    Z = (N * (1.0 - ecc1**2.0) + h) * np.sin(latitude_geodetic_rad)
    # return the cartesian coordinates
    # flattened to singular values if necessary
    if singular_values:
        return (X[0], Y[0], Z[0])
    else:
        return (X, Y, Z)


def to_sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Convert from cartesian coordinates to spherical coordinates

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    """
    # verify axes and copy to not modify inputs
    singular_values = np.ndim(x) == 0
    x = np.atleast_1d(np.copy(x)).astype(np.float64)
    y = np.atleast_1d(np.copy(y)).astype(np.float64)
    z = np.atleast_1d(np.copy(z)).astype(np.float64)
    # calculate radius
    rad = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    # calculate angular coordinates
    # phi: azimuthal angle
    phi = np.arctan2(y, x)
    # th: polar angle
    th = np.arccos(z / rad)
    # convert to degrees and fix to 0:360
    lon = 180.0 * phi / np.pi
    lon = np.where(lon < 0, lon + 360.0, lon)
    # convert to degrees and fix to -90:90
    lat = 90.0 - (180.0 * th / np.pi)
    np.clip(lat, -90, 90, out=lat)
    # return longitude, latitude and radius
    # flattened to singular values if necessary
    if singular_values:
        return (lon[0], lat[0], rad[0])
    else:
        return (lon, lat, rad)


def to_geodetic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
    method: str = "bowring",
    eps: float = np.finfo(np.float64).eps,
    iterations: int = 10,
):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using either iterative or closed-form methods

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    method: str, default 'bowring'
        method to use for conversion

            - ``'moritz'``: iterative solution
            - ``'bowring'``: iterative solution
            - ``'zhu'``: closed-form solution
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative methods
    iterations: int, default 10
        maximum number of iterations
    """
    # verify axes and copy to not modify inputs
    singular_values = np.ndim(x) == 0
    x = np.atleast_1d(np.copy(x)).astype(np.float64)
    y = np.atleast_1d(np.copy(y)).astype(np.float64)
    z = np.atleast_1d(np.copy(z)).astype(np.float64)
    # calculate the geodetic coordinates using the specified method
    if method.lower() == "moritz":
        lon, lat, h = _moritz_iterative(
            x, y, z, a_axis=a_axis, flat=flat, eps=eps, iterations=iterations
        )
    elif method.lower() == "bowring":
        lon, lat, h = _bowring_iterative(
            x, y, z, a_axis=a_axis, flat=flat, eps=eps, iterations=iterations
        )
    elif method.lower() == "zhu":
        lon, lat, h = _zhu_closed_form(x, y, z, a_axis=a_axis, flat=flat)
    else:
        raise ValueError(f"Unknown conversion method: {method}")
    # return longitude, latitude and height
    # flattened to singular values if necessary
    if singular_values:
        return (lon[0], lat[0], h[0])
    else:
        return (lon, lat, h)


def _moritz_iterative(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
    eps: float = np.finfo(np.float64).eps,
    iterations: int = 10,
):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the iterative solution of :cite:p:`HofmannWellenhof:2006hy`

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative method
    iterations: int, default 10
        maximum number of iterations
    """
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0 * flat - flat**2) * a_axis**2)
    ecc1 = lin_ecc / a_axis
    # degrees to radians
    dtr = np.pi / 180.0
    # calculate longitude
    lon = np.arctan2(y, x) / dtr
    # set initial estimate of height to 0
    h = np.zeros_like(lon)
    h0 = np.inf * np.ones_like(lon)
    # calculate radius of parallel
    p = np.hypot(x, y)
    # initial estimated value for phi using h=0
    phi = np.arctan(z / (p * (1.0 - ecc1**2)))
    # iterate to tolerance or to maximum number of iterations
    i = 0
    while np.any(np.abs(h - h0) > eps) and (i <= iterations):
        # copy previous iteration of height
        h0 = np.copy(h)
        # calculate radius of curvature
        N = a_axis / np.sqrt(1.0 - ecc1**2 * np.sin(phi) ** 2)
        # estimate new value of height
        h = p / np.cos(phi) - N
        # estimate new value for latitude using heights
        phi = np.arctan(z / (p * (1.0 - ecc1**2 * N / (N + h))))
        # add to iterator
        i += 1
    # return longitude, latitude and height
    return (lon, phi / dtr, h)


def _bowring_iterative(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
    eps: float = np.finfo(np.float64).eps,
    iterations: int = 10,
):
    """
    Convert from cartesian coordinates to geodetic coordinates using
    the iterative solution of :cite:p:`Bowring:1976jh,Bowring:1985du`

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative method
    iterations: int, default 10
        maximum number of iterations
    """
    # semiminor axis of the WGS84 ellipsoid [m]
    b_axis = (1.0 - flat) * a_axis
    # Linear eccentricity
    lin_ecc = np.sqrt((2.0 * flat - flat**2) * a_axis**2)
    # square of first and second numerical eccentricity
    e12 = lin_ecc**2 / a_axis**2
    e22 = lin_ecc**2 / b_axis**2
    # degrees to radians
    dtr = np.pi / 180.0
    # calculate longitude
    lon = np.arctan2(y, x) / dtr
    # calculate radius of parallel
    p = np.hypot(x, y)
    # initial estimated value for reduced parametric latitude
    u = np.arctan(a_axis * z / (b_axis * p))
    # initial estimated value for latitude
    phi = np.arctan(
        (z + e22 * b_axis * np.sin(u) ** 3)
        / (p - e12 * a_axis * np.cos(u) ** 3)
    )
    phi0 = np.inf * np.ones_like(lon)
    # iterate to tolerance or to maximum number of iterations
    i = 0
    while np.any(np.abs(phi - phi0) > eps) and (i <= iterations):
        # copy previous iteration of phi
        phi0 = np.copy(phi)
        # calculate reduced parametric latitude
        u = np.arctan(b_axis * np.tan(phi) / a_axis)
        # estimate new value of latitude
        phi = np.arctan(
            (z + e22 * b_axis * np.sin(u) ** 3)
            / (p - e12 * a_axis * np.cos(u) ** 3)
        )
        # add to iterator
        i += 1
    # calculate final radius of curvature
    N = a_axis / np.sqrt(1.0 - e12 * np.sin(phi) ** 2)
    # estimate final height (Bowring, 1985)
    h = p * np.cos(phi) + z * np.sin(phi) - a_axis**2 / N
    # return longitude, latitude and height
    return (lon, phi / dtr, h)


def _zhu_closed_form(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the closed-form solution of :cite:p:`Zhu:1993ja`

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    """
    # semiminor axis of the WGS84 ellipsoid [m]
    b_axis = (1.0 - flat) * a_axis
    # Linear eccentricity
    lin_ecc = np.sqrt((2.0 * flat - flat**2) * a_axis**2)
    # square of first numerical eccentricity
    e12 = lin_ecc**2 / a_axis**2
    # degrees to radians
    dtr = np.pi / 180.0
    # calculate longitude
    lon = np.arctan2(y, x) / dtr
    # calculate radius of parallel
    w = np.hypot(x, y)
    # allocate for output latitude and height
    lat = np.zeros_like(lon)
    h = np.zeros_like(lon)
    if np.any(w == 0):
        # special case where w == 0 (exact polar solution)
        (ind,) = np.nonzero(w == 0)
        h[ind] = np.sign(z[ind]) * z[ind] - b_axis
        lat[ind] = 90.0 * np.sign(z[ind])
    else:
        # all other cases
        (ind,) = np.nonzero(w != 0)
        l = e12 / 2.0
        m = (w[ind] / a_axis) ** 2.0
        n = ((1.0 - e12) * z[ind] / b_axis) ** 2.0
        i = -(2.0 * l**2 + m + n) / 2.0
        k = (l**2.0 - m - n) * l**2.0
        q = (1.0 / 216.0) * (m + n - 4.0 * l**2) ** 3.0 + m * n * l**2.0
        D = np.sqrt((2.0 * q - m * n * l**2) * m * n * l**2)
        B = i / 3.0 - (q + D) ** (1.0 / 3.0) - (q - D) ** (1.0 / 3.0)
        t = np.sqrt(np.sqrt(B**2 - k) - (B + i) / 2.0) - np.sign(
            m - n
        ) * np.sqrt((B - i) / 2.0)
        wi = w / (t + l)
        zi = (1.0 - e12) * z[ind] / (t - l)
        # calculate latitude and height
        lat[ind] = np.arctan2(zi, ((1.0 - e12) * wi)) / dtr
        h[ind] = np.sign(t - 1.0 + l) * np.sqrt(
            (w - wi) ** 2.0 + (z[ind] - zi) ** 2.0
        )
    # return longitude, latitude and height
    return (lon, lat, h)


def to_ENU(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lon0: float | np.ndarray = 0.0,
    lat0: float | np.ndarray = 0.0,
    h0: float | np.ndarray = 0.0,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
):
    """
    Convert from Earth-Centered Earth-Fixed (ECEF) cartesian coordinates
    to East-North-Up coordinates (ENU)

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    lon0: float or np.ndarray, default 0.0
        reference longitude (degrees east)
    lat0: float or np.ndarray, default 0.0
        reference latitude (degrees north)
    h0: float or np.ndarray, default 0.0
        reference height (meters)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    Returns
    -------
    E: np.ndarray
        east coordinates
    N: np.ndarray
        north coordinates
    U: np.ndarray
        up coordinates
    """
    # degrees to radians
    dtr = np.pi / 180.0
    # verify axes and copy to not modify inputs
    singular_values = np.ndim(x) == 0
    x = np.atleast_1d(np.copy(x)).astype(np.float64)
    y = np.atleast_1d(np.copy(y)).astype(np.float64)
    z = np.atleast_1d(np.copy(z)).astype(np.float64)
    # convert latitude and longitude to ECEF
    X0, Y0, Z0 = to_cartesian(lon0, lat0, h=h0, a_axis=a_axis, flat=flat)
    # calculate the rotation matrix
    R = np.zeros((3, 3))
    R[0, 0] = -np.sin(dtr * lon0)
    R[0, 1] = np.cos(dtr * lon0)
    R[0, 2] = 0.0
    R[1, 0] = -np.sin(dtr * lat0) * np.cos(dtr * lon0)
    R[1, 1] = -np.sin(dtr * lat0) * np.sin(dtr * lon0)
    R[1, 2] = np.cos(dtr * lat0)
    R[2, 0] = np.cos(dtr * lat0) * np.cos(dtr * lon0)
    R[2, 1] = np.cos(dtr * lat0) * np.sin(dtr * lon0)
    R[2, 2] = np.sin(dtr * lat0)
    # calculate the ENU coordinates
    E, N, U = np.dot(R, np.vstack((x - X0, y - Y0, z - Z0)))
    # return the ENU coordinates
    # flattened to singular values if necessary
    if singular_values:
        return (E[0], N[0], U[0])
    else:
        return (E, N, U)


def from_ENU(
    E: np.ndarray,
    N: np.ndarray,
    U: np.ndarray,
    lon0: float | np.ndarray = 0.0,
    lat0: float | np.ndarray = 0.0,
    h0: float | np.ndarray = 0.0,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
):
    """
    Convert from East-North-Up coordinates (ENU) to
    Earth-Centered Earth-Fixed (ECEF) cartesian coordinates

    Parameters
    ----------
    E, np.ndarray
        east coordinates
    N, np.ndarray
        north coordinates
    U, np.ndarray
        up coordinates
    lon0: float or np.ndarray, default 0.0
        reference longitude (degrees east)
    lat0: float or np.ndarray, default 0.0
        reference latitude (degrees north)
    h0: float or np.ndarray, default 0.0
        reference height (meters)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    Returns
    -------
    x, float
        cartesian x-coordinates
    y, float
        cartesian y-coordinates
    z, float
        cartesian z-coordinates
    """
    # degrees to radians
    dtr = np.pi / 180.0
    # verify axes and copy to not modify inputs
    singular_values = np.ndim(E) == 0
    E = np.atleast_1d(np.copy(E)).astype(np.float64)
    N = np.atleast_1d(np.copy(N)).astype(np.float64)
    U = np.atleast_1d(np.copy(U)).astype(np.float64)
    # convert latitude and longitude to ECEF
    X0, Y0, Z0 = to_cartesian(lon0, lat0, h=h0, a_axis=a_axis, flat=flat)
    # calculate the rotation matrix
    R = np.zeros((3, 3))
    R[0, 0] = -np.sin(dtr * lon0)
    R[1, 0] = np.cos(dtr * lon0)
    R[2, 0] = 0.0
    R[0, 1] = -np.sin(dtr * lat0) * np.cos(dtr * lon0)
    R[1, 1] = -np.sin(dtr * lat0) * np.sin(dtr * lon0)
    R[2, 1] = np.cos(dtr * lat0)
    R[0, 2] = np.cos(dtr * lat0) * np.cos(dtr * lon0)
    R[1, 2] = np.cos(dtr * lat0) * np.sin(dtr * lon0)
    R[2, 2] = np.sin(dtr * lat0)
    # calculate the ECEF coordinates
    x, y, z = np.dot(R, np.vstack((E, N, U)))
    # add reference coordinates
    x += X0
    y += Y0
    z += Z0
    # return the ECEF coordinates
    # flattened to singular values if necessary
    if singular_values:
        return (x[0], y[0], z[0])
    else:
        return (x, y, z)


def to_horizontal(
    E: np.ndarray,
    N: np.ndarray,
    U: np.ndarray,
):
    """
    Convert from East-North-Up coordinates (ENU) to a
    celestial horizontal coordinate system (alt-az)

    Parameters
    ----------
    E: np.ndarray
        east coordinates
    N: np.ndarray
        north coordinates
    U: np.ndarray
        up coordinates

    Returns
    -------
    alpha: np.ndarray
        altitude (elevation) angle in degrees
    phi: np.ndarray
        azimuth angle in degrees
    D: np.ndarray
        distance from observer to object in meters
    """
    # calculate distance to object
    # convert coordinates to unit vectors
    D = np.sqrt(E**2 + N**2 + U**2)
    # altitude (elevation) angle in degrees
    alpha = np.arcsin(U / D) * 180.0 / np.pi
    # azimuth angle in degrees (fixed to 0 to 360)
    phi = np.mod(np.arctan2(E / D, N / D) * 180.0 / np.pi, 360.0)
    return (alpha, phi, D)


def to_zenith(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lon0: float | np.ndarray = 0.0,
    lat0: float | np.ndarray = 0.0,
    h0: float | np.ndarray = 0.0,
    a_axis: float = 6378137.0,
    flat: float = 1.0 / 298.257223563,
):
    """
    Calculate zenith angle of an object from Earth-Centered
    Earth-Fixed (ECEF) cartesian coordinates

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    lon0: float or np.ndarray, default 0.0
        reference longitude (degrees east)
    lat0: float or np.ndarray, default 0.0
        reference latitude (degrees north)
    h0: float or np.ndarray, default 0.0
        reference height (meters)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    Returns
    -------
    zenith: np.ndarray
        zenith angle of object in degrees
    """
    # convert from ECEF to ENU
    E, N, U = to_ENU(
        x, y, z, lon0=lon0, lat0=lat0, h0=h0, a_axis=a_axis, flat=flat
    )
    # convert from ENU to horizontal coordinates
    alpha, phi, D = to_horizontal(E, N, U)
    # calculate zenith angle in degrees
    zenith = 90.0 - alpha
    # return zenith angle
    return zenith


def scale_factors(
    lat: np.ndarray,
    flat: float = 1.0 / 298.257223563,
    reference_latitude: float = 70.0,
    metric: str = "area",
):
    """
    Calculates scaling factors to account for polar stereographic
    distortion including special case of at the exact pole
    :cite:p:`Snyder:1982gf`

    Parameters
    ----------
    lat: np.ndarray
        latitude (degrees north)
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    reference_latitude: float, default 70.0
        reference latitude (true scale latitude)
    metric: str, default 'area'
        metric to calculate scaling factors

            - ``'distance'``: scale factors for distance
            - ``'area'``: scale factors for area

    Returns
    -------
    scale: np.ndarray
        scaling factors at input latitudes
    """
    assert metric.lower() in ["distance", "area"], "Unknown metric"
    # convert latitude from degrees to positive radians
    theta = np.abs(lat) * np.pi / 180.0
    # convert reference latitude from degrees to positive radians
    theta_ref = np.abs(reference_latitude) * np.pi / 180.0
    # square of the eccentricity of the ellipsoid
    # ecc2 = (1-b**2/a**2) = 2.0*flat - flat^2
    ecc2 = 2.0 * flat - flat**2
    # eccentricity of the ellipsoid
    ecc = np.sqrt(ecc2)
    # calculate ratio at input latitudes
    m = np.cos(theta) / np.sqrt(1.0 - ecc2 * np.sin(theta) ** 2)
    t = np.tan(np.pi / 4.0 - theta / 2.0) / (
        (1.0 - ecc * np.sin(theta)) / (1.0 + ecc * np.sin(theta))
    ) ** (ecc / 2.0)
    # calculate ratio at reference latitude
    mref = np.cos(theta_ref) / np.sqrt(1.0 - ecc2 * np.sin(theta_ref) ** 2)
    tref = np.tan(np.pi / 4.0 - theta_ref / 2.0) / (
        (1.0 - ecc * np.sin(theta_ref)) / (1.0 + ecc * np.sin(theta_ref))
    ) ** (ecc / 2.0)
    # distance scaling
    k = (mref / m) * (t / tref)
    kp = (
        0.5
        * mref
        * np.sqrt(((1.0 + ecc) ** (1.0 + ecc)) * ((1.0 - ecc) ** (1.0 - ecc)))
        / tref
    )
    if metric.lower() == "distance":
        # distance scaling
        scale = np.where(np.isclose(theta, np.pi / 2.0), 1.0 / kp, 1.0 / k)
    elif metric.lower() == "area":
        # area scaling
        scale = np.where(
            np.isclose(theta, np.pi / 2.0), 1.0 / (kp**2), 1.0 / (k**2)
        )
    return scale
