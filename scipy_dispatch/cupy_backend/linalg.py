import cupyx.scipy.linalg as _cupy_linalg
import cupy
from functools import wraps

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

import scipy_dispatch.linalg as linalg

cupy_linalg_functions = (
    getattr(_cupy_linalg, x)
    for x in dir(_cupy_linalg)
    if callable(getattr(_cupy_linalg, x)) and not x.startswith("_")
)

def make_array_api_wrapper(function, Array):
    @wraps(function)
    def array_api_wrapper(*args, **kwargs):
        # assumption is that only positional arguments are arrays
        def convert_array(array):
            if isinstance(array, Array):
                return array._array
            return array
        args = tuple(map(convert_array, args))
        result = function(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(map(Array._new, result))
        return Array._new(result)
    return array_api_wrapper

for function in cupy_linalg_functions:
    scipy_function = getattr(linalg, function.__name__)
    scipy_function.register(cupy.ndarray, function)

    # Array API wrappers
    if CUPY_ARRAY_API_AVAILABLE:
        cupy_array_type = cupy.array_api._array_object.Array
        scipy_function.register(cupy_array_type, make_array_api_wrapper(function, cupy_array_type))
    if NUMPY_ARRAY_API_AVAILABLE:
        numpy_array_type = numpy.array_api._array_object.Array
        scipy_function.register(numpy_array_type, make_array_api_wrapper(function, numpy_array_type))
