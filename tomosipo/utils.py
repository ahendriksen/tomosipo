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
