from scipy_dispatch import linalg
import cupy
import numpy

try:
    import cupy.array_api
    CUPY_ARRAY_API_AVAILABLE = True
except ImportError:
    CUPY_ARRAY_API_AVAILABLE = False

try:
    import numpy.array_api
    NUMPY_ARRAY_API_AVAILABLE = True
except ImportError:
    NUMPY_ARRAY_API_AVAILABLE = False

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
        expected = 2
        if CUPY_ARRAY_API_AVAILABLE or NUMPY_ARRAY_API_AVAILABLE:
            expected = 3
        if CUPY_ARRAY_API_AVAILABLE and NUMPY_ARRAY_API_AVAILABLE:
            expected = 4
        assert len(registry) == expected
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

if CUPY_ARRAY_API_AVAILABLE:
    def test_dispatch_cupy_array_api():
        xp = cupy.array_api
        a = xp.asarray(cupy.random.random((5, 5)))
        lu, piv = linalg.lu_factor(a)

        with check:
            assert isinstance(lu, cupy.array_api._array_object.Array)
        with check:
            assert isinstance(piv, cupy.array_api._array_object.Array)

if NUMPY_ARRAY_API_AVAILABLE:
    def test_dispatch_cupy_array_api():
        xp = numpy.array_api
        a = xp.asarray(numpy.random.random((5, 5)))
        lu, piv = linalg.lu_factor(a)

        with check:
            assert isinstance(lu, numpy.array_api._array_object.Array)
        with check:
            assert isinstance(piv, numpy.array_api._array_object.Array)
