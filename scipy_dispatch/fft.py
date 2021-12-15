import scipy.fft

from functools import singledispatch, wraps

_fft_functions = (
    getattr(scipy.fft, x) for x in scipy.fft.__all__ if callable(getattr(scipy.fft, x))
)

for function in _fft_functions:
    fname = function.__name__
    exec(
        f"""
@wraps(scipy.fft.{fname})
@singledispatch
def {fname}(*args, **kwargs):
    return scipy.fft.{fname}(*args, **kwargs)
"""
    )
