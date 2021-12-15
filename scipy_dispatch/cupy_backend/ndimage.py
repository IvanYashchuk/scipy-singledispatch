import cupyx.scipy.ndimage as _cupy_ndimage
import cupy

import scipy_dispatch.ndimage as ndimage

cupy_ndimage_functions = (
    getattr(_cupy_ndimage, x)
    for x in dir(_cupy_ndimage)
    if callable(getattr(_cupy_ndimage, x)) and not x.startswith("_")
)

for function in cupy_ndimage_functions:
    scipy_function = getattr(ndimage, function.__name__)
    scipy_function.register(cupy.ndarray, function)
