from scipy_dispatch import special
import cupy
import numpy

from pytest_check import check

# Tests are order dependent.


def test_registry():
    from scipy_dispatch import special

    registry = special.zeta.registry
    # At the moment, there are no specialized functions registered.
    with check:
        assert len(registry) == 1

    # The only function is the default one.
    with check:
        assert object in registry


def test_registry_cupy():
    import scipy_dispatch.cupy_backend.special

    registry = special.zeta.registry
    # Now there are two specialized functions registered.
    with check:
        assert len(registry) == 2
    with check:
        assert cupy.ndarray in registry


def test_dispatch():
    a_np = numpy.random.random((5, 5))
    b_np = numpy.random.random((5, 5))
    a_cp = cupy.random.random((5, 5))
    b_cp = cupy.random.random((5, 5))

    out_np = special.zeta(a_np, b_np)
    out_cp = special.zeta(a_cp, b_cp)

    with check:
        assert isinstance(out_np, numpy.ndarray)
    with check:
        assert isinstance(out_cp, cupy.ndarray)


def test_dispatch_digamma():
    a_np = numpy.random.random((5, 5))
    a_cp = cupy.random.random((5, 5))

    out_np = special.digamma(a_np)
    out_cp = special.digamma(a_cp)

    with check:
        assert isinstance(out_np, numpy.ndarray)
    with check:
        assert isinstance(out_cp, cupy.ndarray)
