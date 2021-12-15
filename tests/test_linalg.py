from scipy_dispatch import linalg
import cupy
import numpy

from pytest_check import check

# Tests are order dependent.


def test_registry():
    from scipy_dispatch import linalg

    registry = linalg.lu_factor.registry
    # At the moment, there are no specialized functions registered.
    with check:
        assert len(registry) == 1

    # The only function is the default one.
    with check:
        assert object in registry


def test_registry_cupy():
    import scipy_dispatch.cupy_backend.linalg

    registry = linalg.lu_factor.registry
    # Now there are two specialized functions registered.
    with check:
        assert len(registry) == 2
    with check:
        assert cupy.ndarray in registry


def test_dispatch():
    a_np = numpy.random.random((5, 5))
    a_cp = cupy.random.random((5, 5))

    lu_np, piv_np = linalg.lu_factor(a_np)
    lu_cp, piv_cp = linalg.lu_factor(a_cp)

    with check:
        assert isinstance(lu_np, numpy.ndarray)
    with check:
        assert isinstance(piv_np, numpy.ndarray)
    with check:
        assert isinstance(lu_cp, cupy.ndarray)
    with check:
        assert isinstance(piv_cp, cupy.ndarray)
