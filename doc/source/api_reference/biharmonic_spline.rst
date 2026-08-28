===========================================
``spatial_interpolators.biharmonic_spline``
===========================================

- Interpolates data using 2-dimensional biharmonic splines
- Can use surface splines in tension
- Can use regularized surface splines

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.biharmonic_spline(xs, ys, zs, XI, YI, tension=0.5)

`Source code`__

.. __: https://github.com/polargeodesy/spatial-interpolators/blob/main/spatial_interpolators/biharmonic_spline.py

.. autofunction:: spatial_interpolators.biharmonic_spline
