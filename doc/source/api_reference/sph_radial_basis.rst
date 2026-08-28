==========================================
``spatial_interpolators.sph_radial_basis``
==========================================

- Interpolates data over a sphere using radial basis functions
- QR factorization option to eliminate ill-conditioning

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    output = spi.sph_radial_basis(lon, lat, data, longitude, latitude, method='inverse')

`Source code`__

.. __: https://github.com/polargeodesy/spatial-interpolators/blob/main/spatial_interpolators/sph_radial_basis.py

.. autofunction:: spatial_interpolators.sph_radial_basis
