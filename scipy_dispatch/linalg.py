from scipy import linalg

from functools import singledispatch, wraps

linalg_functions = (
    getattr(linalg, x) for x in linalg.__all__ if callable(getattr(linalg, x))
)

for function in linalg_functions:
    fname = function.__name__
    exec(
        f"""
@wraps(linalg.{fname})
@singledispatch
def {fname}(*args, **kwargs):
    return linalg.{fname}(*args, **kwargs)
"""
    )
