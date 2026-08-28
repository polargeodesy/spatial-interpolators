==================================
``spatial_interpolators.legendre``
==================================

- Computes associated Legendre functions of degree ``l`` evaluated for elements ``x``
- ``l`` must be a scalar integer and ``x`` must contain real values ranging -1 <= ``x`` <= 1
- Unnormalized associated Legendre function values will overflow for ``l`` > 150

Calling Sequence
################

.. code-block:: python

    from spatial_interpolators.legendre import legendre
    Pl = legendre(l, x)

`Source code`__

.. __: https://github.com/polargeodesy/spatial-interpolators/blob/main/spatial_interpolators/legendre.py

.. autofunction:: spatial_interpolators.legendre

