import collections
from numbers import Integral
import numpy as np


def print_options():
    """This context manager set numpy print options temporarily

    ODL sets the numpy printoptions globally, which makes it difficult
    to ensure that the representation of geometries is consistent over
    different installations. Specifically, we want to ensure that
    testing succeeds regardless of the specific version of ODL.

    :returns:
    :rtype:

    """
    # Set printing line width to 71 to allow method docstrings to not extend
    # beyond 79 characters (2 times indent of 4)
    return np.printoptions(
        edgeitems=3,
        threshold=1000,
        floatmode="maxprec",
        precision=8,
        suppress=False,
        linewidth=71,
        nanstr="nan",
        infstr="inf",
        sign="-",
        formatter=None,
        legacy=False,
    )


def up_tuple(x, n):
    if isinstance(x, collections.Iterator) or isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) == 1:
            return (x[0],) * n
        if len(x) != n:
            raise ValueError(f"Expected tuple with {n} elements.")
        return x
    # Make tuple
    return (x,) * n


def to_shape(shape, dim=3, allow_zeros=True):
    shape = up_tuple(shape, dim)
    for s in shape:
        if not isinstance(s, Integral):
            raise TypeError(
                f"Shape must contain only integers. Got {shape} with type {type(s)}. "
            )
    shape = tuple(map(int, shape))
    if allow_zeros:
        if not all(s >= 0 for s in shape):
            raise ValueError(f"Shape must be non-negative. Got {shape}.")
    else:
        if not all(s >= 1 for s in shape):
            raise ValueError(f"Shape must be positive. Got {shape}.")

    return shape


def to_pos(pos, dim=3):
    if np.isscalar(pos) and pos == 0.0:
        pos = up_tuple(pos, dim)
    elif np.isscalar(pos):
        raise ValueError(
            f"Position {pos} is not automatically converted to {up_tuple(pos, dim)}. "
            f"Provide a tuple instead. "
        )

    # TODO: Check if pos is a vector and nicely shaped.
    return pos


def to_size(size, dim=3):
    if np.isscalar(size):
        size = up_tuple(size, dim)
    if not all(np.ndim(s) > 0 or s >= 0.0 for s in size):
        raise ValueError(f"Size must be non-negative. Got {size}")
    # TODO: Check if size is a vector and nicely shaped.
    return size


def up_slice(key):
    if isinstance(key, Integral):
        if key == -1:
            key = slice(key, None)
        else:
            key = slice(key, key + 1)

    return key


def slice_interval(left, right, length, key):
    """Slice a one-dimensional interval

    The line extends from `left` to `right` and is divided into
    `length` parts. It is indexed by a `key` which can be an `int` or
    a `slice`.

    If the step size of `key` equals one, then `slice_interval` should
    works exactly as expected.

    When the step size is greater than one, there are two equally
    valid approaches to slicing:

    1) Take the left-most and right-most pixel of the slice and set
       the size of the resulting interval to be equal to the
       difference between their left- and right-most edge,
       respectively.

    2) Multiply the pixel size by the step size and ensure the size of
       the new interval equals new_len * new_pixel_size.

    This function implements the second approach.

    The first approach has the disadvantage that the new pixel size is
    not an integer multiple of the original pixel size. Moreover, it
    is not compatible with the way that we want to use this function
    as a way to bin detector pixels.

    The second approach has the disadvantage that the new values for
    `left` and `right` might fall outside the original interval, which
    is counter-intuitive. Nonetheless, it is compatible with detector
    and volume binning.

    Example. Suppose we have the interval [0, 4] divided into four
    pixels. An illustration of the interval is shown below, with `|`
    denoting the edges of the pixels. As you can see, with a step size
    of two, the pixels become twice as large, and depending on the
    start of the slice, the interval is shifted to the left or
    right. With a step size of 3, the resulting interval is larger
    than the original interval.

       |x|x|x|x|   [0:4:1]
      | x | x |    [0:4:2]
        | x | x |  [1:4:2]
     |  x  |  x  | [0:4:3]

    :param left: `scalar` or `np.array`
    :param right: `scalar` or `np.array`
    :param length: `int`
    :param key: `slice` or `int`
    :returns:
        A tuple containing the new left and right extent of the line,
        the number of "pixels", and the new "pixel" size.
    :rtype: `(scalar, scalar, int, scalar)`

    """
    # Prevent division by zero
    pixel_size = 1 if length == 0 else (right - left) / length
    key = up_slice(key)
    start, stop, step = key.indices(length)
    # `(stop - start)` is not necessarily divisible by `step`. We have
    # to calculate new_len in this convoluted way:
    new_len = max(0, (stop - start + step - 1) // step)
    stop = max(start, start + (new_len - 1) * step + 1)

    new_pixel_size = pixel_size * step
    L = left + start * pixel_size + 0.5 * pixel_size * (1 - step)
    R = left + stop * pixel_size + 0.5 * pixel_size * (step - 1)

    # Prevent division by zero
    return (L, R, new_len, new_pixel_size)
