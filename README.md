# scipy-singledispatch &middot;

This package demonstrates the use of SciPy with [`functools.singledispatch`](https://docs.python.org/3/library/functools.html#functools.singledispatch) for registering functions with different array types than NumPy.

Use this package as drop-in replacement for `scipy.module` as `scipy_dispatch.module`. Currently, only a CuPy backend is supported.
The CuPy backend can be enabled by importing `scipy_dispatch.cupy_backend.module`.

## Example

```python
import scipy_dispatch.cupy_backend.linalg
from scipy_dispatch import linalg
import cupy

A = cupy.random.random((5, 5))
lu, piv = linalg.lu_factor(A)
assert type(lu) == cupy.ndarray
assert type(piv) == cupy.ndarray
```

## Installation
There are no additional dependencies except SciPy and CuPy. Once they are installed, the package can be installed with:

    python -m pip install git+https://github.com/IvanYashchuk/scipy-singledispatch.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/scipy-singledispatch/issues/new

## Asking questions and general discussion

If you have a question or anything else, create a new [discussion]. Using issues is also fine!

[discussion]: https://github.com/IvanYashchuk/scipy-singledispatch/discussions/new
