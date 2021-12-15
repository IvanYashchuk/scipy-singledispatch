from scipy import ndimage

from functools import singledispatch, wraps

ndimage_functions = (
    getattr(ndimage, x) for x in ndimage.__all__ if callable(getattr(ndimage, x))
)

for function in ndimage_functions:
    fname = function.__name__
    exec(
        f"""
@wraps(ndimage.{fname})
@singledispatch
def {fname}(*args, **kwargs):
    return ndimage.{fname}(*args, **kwargs)
"""
    )
