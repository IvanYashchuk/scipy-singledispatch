from scipy_dispatch import ndimage
import cupy
import numpy

from pytest_check import check

# Tests are order dependent.


def test_registry():
    from scipy_dispatch import ndimage

    registry = ndimage.gaussian_filter.registry
    # At the moment, there are no specialized functions registered.
    with check:
        assert len(registry) == 1

    # The only function is the default one.
    with check:
        assert object in registry


def test_registry_cupy():
    import scipy_dispatch.cupy_backend.ndimage

    registry = ndimage.gaussian_filter.registry
    # Now there are two specialized functions registered.
    with check:
        assert len(registry) == 2
    with check:
        assert cupy.ndarray in registry


def test_dispatch():
    a_np = numpy.random.random((5, 5))
    a_cp = cupy.random.random((5, 5))

    out_np = ndimage.gaussian_filter(a_np, sigma=1)
    out_cp = ndimage.gaussian_filter(a_cp, sigma=1)

    with check:
        assert isinstance(out_np, numpy.ndarray)
    with check:
        assert isinstance(out_cp, cupy.ndarray)
