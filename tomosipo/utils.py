import collections
from numbers import Integral


def up_tuple(x, n):
    if isinstance(x, collections.Iterator) or isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) == 1:
            return (x[0],) * n
        if len(x) != n:
            raise ValueError(f"Expected container with {n} elements.")
        return x
    # Make tuple
    return (x,) * n


def index_one_dim(left, right, length, key):
    """Index into a one-dimensional line with key

    The line extends from `left` to `right` and is divided into
    `length` parts. It is indexed by a `key` which can be an `int` or
    a `slice`.

    :param left: `scalar`
    :param right: `scalar`
    :param length: `int`
    :param key: `slice` or `int`
    :returns:
        A tuple containing the new left and right extent of the line,
        the number of "pixels", and the new "pixel" size.
    :rtype: `(scalar, scalar, int, scalar)`

    """
    # Prevent division by zero
    pixel_size = 1 if length == 0 else (right - left) / length
    if isinstance(key, Integral):
        if key < 0:
            key = slice(key, None)
        else:
            key = slice(key, key + 1)

    start, stop, step = key.indices(length)

    L = left + start * pixel_size
    R = left + stop * pixel_size

    # Prevent division by zero
    new_len = (stop - start) // step
    new_pixel_size = 1 if start == stop else (R - L) / new_len
    return (L, R, new_len, new_pixel_size)
