from scipy_dispatch import fft
import cupy
import numpy

from pytest_check import check

# Tests are order dependent.


def test_registry():
    from scipy_dispatch import fft

    registry = fft.fft.registry
    # At the moment, there are no specialized functions registered.
    with check:
        assert len(registry) == 1

    # The only function is the default one.
    with check:
        assert object in registry


def test_registry_cupy():
    import scipy_dispatch.cupy_backend.fft

    registry = fft.fft.registry
    # Now there are two specialized functions registered.
    with check:
        assert len(registry) == 2
    with check:
        assert cupy.ndarray in registry


def test_dispatch():
    a_np = numpy.random.random((5, 5))
    a_cp = cupy.random.random((5, 5))

    out_np = fft.fft(a_np)
    out_cp = fft.fft(a_cp)

    with check:
        assert isinstance(out_np, numpy.ndarray)
    with check:
        assert isinstance(out_cp, cupy.ndarray)
