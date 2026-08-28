=============================================
``spatial_interpolators.shepard_interpolant``
=============================================

- Interpolates data by evaluating Shepard Interpolants based on inverse distances

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.shepard_interpolant(xs, ys, zs, XI, YI, power=2.0)

`Source code`__

.. __: https://github.com/polargeodesy/spatial-interpolators/blob/main/spatial_interpolators/shepard_interpolant.py

.. autofunction:: spatial_interpolators.shepard_interpolant
