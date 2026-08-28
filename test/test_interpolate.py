#!/usr/bin/env python
"""
test_interpolate.py (01/2026)
Test the interpolation and extrapolation routines

UPDATE HISTORY:
    Updated 01/2026: xfail tests on HTTPError exceptions
    Updated 08/2025: added 1d interpolation routine test
        added inpaint interpolation test based on 2D franke function
    Updated 04/2023: test geodetic conversion additionally as arrays
        using pathlib to define and expand paths
    Updated 12/2022: refactored interpolation routines into new module
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Written 03/2021
"""

import pytest
import inspect
import pathlib
import posixpath
import numpy as np
import scipy.io
import urllib.request
import spatial_interpolators as spi

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent


# PURPOSE: Download max determinant nodes from spherepts
# https://github.com/gradywright/spherepts
@pytest.fixture(scope="module", autouse=True)
def download_nodes(N=324, cleanup=False):
    # attempt to download the node file
    try:
        matfile = spi.datasets.fetch_test_data(directory=filepath, N=N)
    except urllib.request.URLError as exc:
        pytest.xfail(exc.reason)
    except urllib.request.HTTPError as exc:
        pytest.xfail(exc.reason)
    # run tests
    yield
    # remove the node file
    if cleanup:
        matfile.unlink(missing_ok=True)


# Franke's 2D evaluation function
def franke_2d(x, y):
    F1 = 0.75 * np.exp(-((9.0 * x - 2.0) ** 2 + (9.0 * y - 2.0) ** 2) / 4.0)
    F2 = 0.75 * np.exp(-((9.0 * x + 1.0) ** 2 / 49.0 - (9.0 * y + 1.0) / 10.0))
    F3 = 0.5 * np.exp(-((9.0 * x - 7.0) ** 2 + (9.0 * y - 3.0) ** 2) / 4.0)
    F4 = 0.2 * np.exp(-((9.0 * x - 4.0) ** 2 + (9.0 * y - 7.0) ** 2))
    F = F1 + F2 + F3 - F4
    return F


# Franke's 3D evaluation function
def franke_3d(x, y, z):
    F1 = 0.75 * np.exp(
        -((9.0 * x - 2.0) ** 2 + (9.0 * y - 2.0) ** 2 + (9.0 * z - 2.0) ** 2)
        / 4.0
    )
    F2 = 0.75 * np.exp(
        -(
            (9.0 * x + 1.0) ** 2 / 49.0
            + (9.0 * y + 1.0) / 10.0
            + (9.0 * z + 1.0) / 10.0
        )
    )
    F3 = 0.5 * np.exp(
        -((9.0 * x - 7.0) ** 2 + (9.0 * y - 3.0) ** 2 + (9.0 * z - 5) ** 2)
        / 4.0
    )
    F4 = 0.2 * np.exp(
        -((9.0 * x - 4.0) ** 2 + (9.0 * y - 7.0) ** 2 + (9.0 * z - 5.0) ** 2)
    )
    F = F1 + F2 + F3 - F4
    return F


# use max determinant nodes from spherepts
def test_cartesian(N=324):
    # read the node file
    matfile = f"md{N:05d}.mat"
    xd = scipy.io.loadmat(filepath.joinpath(matfile))
    x, y, z = (xd["x"][:, 0], xd["x"][:, 1], xd["x"][:, 2])
    # convert from cartesian to sphere
    lon, lat, r = spi.spatial.to_sphere(x, y, z)
    X, Y, Z = spi.spatial.to_cartesian(lon, lat, a_axis=r, flat=0.0)
    # verify that coordinates are within tolerance
    assert np.allclose(x, X)
    assert np.allclose(y, Y)
    assert np.allclose(z, Z)


# use max determinant nodes from spherepts
def test_geodetic(N=324):
    # read the node file
    matfile = f"md{N:05d}.mat"
    xd = scipy.io.loadmat(filepath.joinpath(matfile))
    # convert from cartesian to sphere
    ln, lt, _ = spi.spatial.to_sphere(
        xd["x"][:, 0], xd["x"][:, 1], xd["x"][:, 2]
    )
    # convert from sphere to cartesian
    X, Y, Z = spi.spatial.to_cartesian(ln, lt)
    # convert from cartesian to geodetic
    lon = np.zeros((N))
    lat = np.zeros((N))
    h = np.zeros((N))
    for i in range(N):
        lon[i], lat[i], h[i] = spi.spatial.to_geodetic(X[i], Y[i], Z[i])
    # fix coordinates to be 0:360
    lon[lon < 0] += 360.0
    # verify that coordinates are within tolerance
    assert np.allclose(ln, lon)
    assert np.allclose(lt, lat)
    # convert from cartesian to geodetic as arrays
    lon, lat, h = spi.spatial.to_geodetic(X, Y, Z)
    # fix coordinates to be 0:360
    lon[lon < 0] += 360.0
    # verify that coordinates are within tolerance
    assert np.allclose(ln, lon)
    assert np.allclose(lt, lat)


# PURPOSE: test gap-filling over a 2D grid
def test_gap_fill(nx=250, ny=250, percent=30, N=100):
    # normalized coordinates
    xpts = np.arange(nx) / np.float64(nx)
    ypts = np.arange(ny) / np.float64(ny)
    XI, YI = np.meshgrid(xpts, ypts)
    # calculate values at grid points
    val = franke_2d(XI, YI)
    # create masked array
    ZI = np.ma.MaskedArray(val.copy())
    ZI.mask = np.zeros((ny, nx), dtype=bool)
    # number of points to be removed
    size = int(percent * nx * ny / 100.0)
    # create random points to be removed from the grid
    rng = np.random.default_rng(0)
    indx = rng.integers(0, nx, size=size)
    indy = rng.integers(0, ny, size=size)
    ZI.mask[indy, indx] = True
    # replace masked points with NaN
    ZI = ZI.filled(np.nan)
    # calculate gap-filled values with inpainting
    test = spi.inpaint(xpts, ypts, ZI, n=N)
    # verify that coordinates are within tolerance
    assert np.allclose(val, test, atol=0.01)
