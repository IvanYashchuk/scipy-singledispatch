import cupyx.scipy.special as _cupy_special
import cupy

import scipy_dispatch.special as special

cupy_special_functions = (
    getattr(_cupy_special, x)
    for x in dir(_cupy_special)
    if callable(getattr(_cupy_special, x)) and not x.startswith("_")
)

for function in cupy_special_functions:
    name = (
        function.__name__
        if not isinstance(function, cupy.ufunc)
        else function.__name__.replace("cupyx_scipy_", "")
    )
    name = name.replace("special_", "")
    scipy_function = getattr(special, name)
    scipy_function.register(cupy.ndarray, function)
