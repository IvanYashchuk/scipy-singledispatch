from scipy import special

from functools import singledispatch, wraps

special_functions = (
    getattr(special, x) for x in special.__all__ if callable(getattr(special, x))
)

for function in special_functions:
    fname = function.__name__
    if "digamma" == fname:
        print(fname)
    exec(
        f"""
@wraps(special.{fname})
@singledispatch
def {fname}(*args, **kwargs):
    return special.{fname}(*args, **kwargs)
"""
    )

# digamma has __name__ psi therefore it's not yet defined
@wraps(special.digamma)
@singledispatch
def digamma(*args, **kwargs):
    return special.digamma(*args, **kwargs)
