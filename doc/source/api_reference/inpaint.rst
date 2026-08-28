=================================
``spatial_interpolators.inpaint``
=================================

- Inpaint over missing data in a two-dimensional array using a penalized least square method based on discrete cosine transforms

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    output = spi.inpaint(xs, ys, zs)

`Source code`__

.. __: https://github.com/polargeodesy/spatial-interpolators/blob/main/spatial_interpolators/inpaint.py

.. autofunction:: spatial_interpolators.inpaint

.. autofunction:: spatial_interpolators.inpaint.nearest_neighbors
