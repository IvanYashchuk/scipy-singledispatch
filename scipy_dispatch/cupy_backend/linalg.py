import cupyx.scipy.linalg as _cupy_linalg
import cupy

import scipy_dispatch.linalg as linalg

cupy_linalg_functions = (
    getattr(_cupy_linalg, x)
    for x in dir(_cupy_linalg)
    if callable(getattr(_cupy_linalg, x)) and not x.startswith("_")
)

for function in cupy_linalg_functions:
    scipy_function = getattr(linalg, function.__name__)
    scipy_function.register(cupy.ndarray, function)
