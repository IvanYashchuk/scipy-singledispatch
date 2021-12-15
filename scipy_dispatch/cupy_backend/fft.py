import cupyx.scipy.fft as _cupy_fft
import cupy

import scipy_dispatch.fft as _fft

cupy_fft_functions = (
    getattr(_cupy_fft, x)
    for x in dir(_cupy_fft)
    if callable(getattr(_cupy_fft, x)) and not x.startswith("_")
)

for function in cupy_fft_functions:
    if function.__name__ in ["next_fast_len", "get_fft_plan"]:
        continue
    scipy_function = getattr(_fft, function.__name__)
    scipy_function.register(cupy.ndarray, function)
