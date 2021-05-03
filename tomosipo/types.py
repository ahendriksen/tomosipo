import tomosipo as ts
from typing import Collection, Union, Tuple, Iterable, Any, TypeVar
from collections import abc
from numbers import Integral
import numpy as np

###############################################################################
#                                    Types                                    #
###############################################################################

T = TypeVar('T')

Shape2D = Tuple[int, int]
Shape3D = Tuple[int, int, int]
Size2D = Tuple[float, float]
Size3D = Tuple[float, float, float]

ToShape2D = Union[int, Tuple[int, int], Iterable[int]]
ToShape3D = Union[int, Tuple[int, int, int], Iterable[int]]
ToSize2D = Union[float, Tuple[float, float], Iterable[float]]
ToSize3D = Union[float, Tuple[float, float, float], Iterable[float]]


Pos = Tuple[float, float, float]
ToPos = Union[float, Collection[float]]

# Used for angles and stuff
Scalars = np.ndarray
ToScalars = Union[float, Collection[float], np.ndarray]


Vec = np.ndarray
HomogeneousVec = np.ndarray

ToVec = Union[
    Tuple[float, float, float],
    Iterable[Tuple[float, float, float]],
    np.ndarray
]

ToHomogeneousVec = Union[
    ToVec,
    Tuple[float, float, float, float],
    Iterable[Tuple[float, float, float, float]],
]


###############################################################################
#                               Shapes and Sizes                              #
###############################################################################

def to_tuple(val: Union[Iterable[T], T], n: int) -> Tuple[T]:
    """Convert value to tuple of length `n`

    >>> to_tuple(1, 2)
    (1, 1)

    >>> to_tuple(1, n=2)
    (1, 1)

    >>> to_tuple((1, 2), n=2)
    (1, 2)

    >>> to_tuple((1, 2, 3), n=2)
    Traceback (most recent call last):
    ...
    TypeError: Expected tuple with 2 elements. Got (1, 2, 3).

    """
    n = int(n)
    if isinstance(val, abc.Iterable):
        shape = tuple(val)
        if len(shape) != n:
            raise TypeError(
                f"Expected tuple with {n} elements. Got {repr(val)}."
            )
        else:
            return val
    else:
        return (val, ) * n


def to_float_tuple(val, n, var_name='value'):
    """Convert value to tuple of `n` floats

    >>> to_float_tuple(1, 2)
    (1.0, 1.0)

    >>> to_float_tuple(1, n=2)
    (1.0, 1.0)

    >>> to_float_tuple((1.0, 2.0), n=2)
    (1.0, 2.0)

    >>> to_tuple((1, 2, 3), n=2)
    Traceback (most recent call last):
    ...
    TypeError: Expected tuple with 2 elements. Got (1, 2, 3).

    >>> to_float_tuple(('a', 0), n=2)
    Traceback (most recent call last):
    ...
    TypeError: value must contain only floats. Got ('a', 0).

    """
    val = to_tuple(val, n)
    try:
        val = tuple(map(float, val))
    except ValueError:
        raise TypeError(
            f"{var_name} must contain only floats. Got {repr(val)}."
        )
    return val


def to_shape_nd(shape, n):
    """ Create N-dimensional shape

    :meta private:
    """
    shape = to_tuple(shape, n)
    for s in shape:
        if not isinstance(s, Integral):
            raise TypeError(
                f"Shape must contain only integers. Got {shape} with type {type(s)}."
            )
    shape = tuple(map(int, shape))

    if not all(s >= 1 for s in shape):
        raise TypeError(f"Shape must be positive. Got {shape}.")

    return shape


def to_shape2d(shape: ToShape2D) -> Shape2D:
    """Create a 2D shape tuple

    >>> to_shape2d(1)
    (1, 1)
    >>> to_shape2d((5, 3))
    (5, 3)
    >>> to_shape2d((5.0, 3))
    Traceback (most recent call last):
    ...
    TypeError: Shape must contain only integers. Got (5.0, 3) with type <class 'float'>.
    """
    return to_shape_nd(shape, 2)


def to_shape3d(shape: ToShape3D) -> Shape3D:
    """Create a 3D shape tuple

    >>> to_shape3d(1)
    (1, 1, 1)
    >>> to_shape3d((5, 3, 2))
    (5, 3, 2)
    >>> to_shape3d((5.0, 3))
    Traceback (most recent call last):
    ...
    TypeError: Expected tuple with 3 elements. Got (5.0, 3).
    """
    return to_shape_nd(shape, 3)


def to_size_nd(size, n):
    """ Create N-dimensional shape

    :meta private:
    """
    size = to_float_tuple(size, n, var_name="Size")

    if not all(s >= -ts.epsilon for s in size):
        raise TypeError(f"Size must be non-negative. Got {size}.")

    return size


def to_size2d(size: ToSize2D) -> Size2D:
    """Create a 2D size tuple

    >>> to_size2d(1)
    (1.0, 1.0)
    >>> to_size2d((5, 3))
    (5.0, 3.0)
    >>> to_size2d((3, 2, 1))
    Traceback (most recent call last):
    ...
    TypeError: Expected tuple with 2 elements. Got (3, 2, 1).
    """
    return to_size_nd(size, 2)


def to_size3d(size: ToSize3D) -> Size3D:
    """Create a 3D size tuple

    >>> to_size3d(1)
    (1.0, 1.0, 1.0)
    >>> to_size3d((3, 2, 1))
    (3.0, 2.0, 1.0)
    >>> to_size3d((2, 1))
    Traceback (most recent call last):
    ...
    TypeError: Expected tuple with 3 elements. Got (2, 1).
    """
    return to_size_nd(size, 3)


def to_pos(pos: ToPos) -> Pos:
    """Create a 3D position tuple

    >>> to_pos(0)
    (0.0, 0.0, 0.0)
    >>> to_pos(0.0)
    (0.0, 0.0, 0.0)
    >>> to_pos((3, 2, 1))
    (3.0, 2.0, 1.0)
    >>> to_pos((2, 1))
    Traceback (most recent call last):
    ...
    TypeError: Expected tuple with 3 elements. Got (2, 1).
    """
    if np.isscalar(pos) and pos == 0.0:
        return (0.0, 0.0, 0.0)
    if isinstance(pos, abc.Iterable):
        return to_float_tuple(pos, 3, "Position")
    else:
        raise TypeError(
            "Cannot convert value to position. Expected (float, float, float). Got {pos}. "
        )


###############################################################################
#                                   Scalars                                   #
###############################################################################

def to_scalars(s: ToScalars, var_name='scalars', accept_empty=False) -> Scalars:
    """Create an array of scalars

    Parameters:
    s
        A single float or collection of floats.
    var_name
        The name to use in error messages.

    Returns
    -------
    np.ndarray
        A 1-dimensional array contains floats.

    Examples
    --------

    >>> to_scalars((1, 1, 1))
    array([1., 1., 1.])

    >>> to_scalars(1).shape
    (1,)

    >>> import numpy as np
    >>> to_scalars(np.ones((1, 3))).shape
    Traceback (most recent call last):
    ...
    TypeError: Value cannot be converted to scalars. Expected shape: (N,). Got shape: (1, 3).

    >>> import numpy as np
    >>> to_scalars([])
    Traceback (most recent call last):
    ...
    TypeError: Value cannot be converted to scalars. Expected shape: (N,). Got shape: (0,).

    >>> to_scalars("string")
    Traceback (most recent call last):
    ...
    TypeError: Could not convert scalars to np.array. Got: 'string'.
    >>> to_scalars(None)
    Traceback (most recent call last):
    ...
    TypeError: Could not convert to array of scalars: array contains NaN.
    """
    try:
        s = np.array(s, dtype=np.float64, ndmin=1, copy=False)
    except ValueError:
        raise TypeError(
            f"Could not convert {var_name} to np.array. Got: {repr(s)}."
        )
    if np.any(np.isnan(s)):
        raise TypeError(
            f"Could not convert to array of scalars: array contains NaN."
        )

    shape = s.shape

    length_acceptable = accept_empty or len(s) > 0
    if s.ndim == 1 and length_acceptable:
        return s

    raise TypeError(
        f"Value cannot be converted to {var_name}. "
        f"Expected shape: (N,). Got shape: {shape}."
    )


###############################################################################
#                            Vectors and Positions                            #
###############################################################################



def to_vec(vec: ToVec, var_name='vector') -> Vec:
    """Create an array of vectors

    Examples
    --------

    >>> to_vec((1, 1, 1))
    array([[1., 1., 1.]])

    >>> to_vec((1, 1, 1)).shape
    (1, 3)

    >>> to_vec([(1, 1, 1), (2, 2, 2)]).shape
    (2, 3)

    >>> import numpy as np
    >>> to_vec(np.ones((1, 3))).shape
    (1, 3)

    >>> import numpy as np
    >>> to_vec(np.ones((1, 2))).shape
    Traceback (most recent call last):
    ...
    TypeError: Value cannot be converted to vector. Expected shape: (3,) or (N, 3). Got shape: (1, 2).

    >>> to_vec("string")
    Traceback (most recent call last):
    ...
    TypeError: Could not convert vector to np.array. Got: 'string'.
    """
    try:
        vec = np.array(vec, dtype=np.float64, copy=False)
    except ValueError:
        raise TypeError(
            f"Could not convert {var_name} to np.array. Got: {repr(vec)}."
        )
    shape = vec.shape
    vec = np.array(vec, dtype=np.float64, copy=False, ndmin=2)

    if vec.ndim == 2 and vec.shape[1] == 3:
        return vec

    raise TypeError(
        f"Value cannot be converted to {var_name}. "
        f"Expected shape: (3,) or (N, 3). Got shape: {shape}."
    )


def to_homogeneous(vec, s) -> HomogeneousVec:
    """Create homogeneous vector or position

    :meta private:
    """
    s = float(s)
    try:
        vec = np.array(vec, dtype=np.float64, copy=False)
    except ValueError:
        raise TypeError(
            f"Could not convert value to np.array. Got: {repr(vec)}."
        )
    shape = vec.shape
    vec = np.array(vec, dtype=np.float64, copy=False, ndmin=2)

    if vec.ndim == 2:
        if vec.shape[1] == 4:
            return vec
        if vec.shape[1] == 3:
            return np.append(
                vec,
                s * np.ones((vec.shape[0], 1)),
                axis=1
            )

    raise TypeError(
        f"Value cannot be converted to homogeneous coordinates. "
        f"Expected shape: (3,), (4,), (N, 3), or (N, 4). Got shape: {shape}. "
    )


def to_homogeneous_vec(vec: ToVec) -> HomogeneousVec:
    """Create an array of vectors

    Examples
    --------

    >>> to_vec((1, 1, 1))
    array([[1., 1., 1.]])

    >>> to_vec((1, 1, 1)).shape
    (1, 3)

    >>> to_vec([(1, 1, 1), (2, 2, 2)]).shape
    (2, 3)

    >>> import numpy as np
    >>> to_vec(np.ones((1, 3))).shape
    (1, 3)

    >>> import numpy as np
    >>> to_vec(np.ones((1, 2))).shape
    Traceback (most recent call last):
    ...
    TypeError: Value cannot be converted to vector. Expected shape: (3,) or (N, 3). Got shape: (1, 2).

    >>> to_vec("string")
    Traceback (most recent call last):
    ...
    TypeError: Could not convert vector to np.array. Got: 'string'.
    """
    return to_homogeneous(vec, 0.0)


def to_homogeneous_pos(vec: ToVec) -> HomogeneousVec:
    """Create an array of vectors

    Parameters
    ----------
    vec : array_like
        The array that is to be converted to a homogeneous position.
        Must be of shape `(3,), (4,), (N, 3)`, or `(N, 4)`.

    Returns
    -------
    np.ndarray
        An array with shape (N, 4)

    Examples
    --------

    >>> to_vec((1, 1, 1))
    array([[1., 1., 1.]])

    >>> to_vec((1, 1, 1)).shape
    (1, 3)

    >>> to_vec([(1, 1, 1), (2, 2, 2)]).shape
    (2, 3)

    >>> import numpy as np
    >>> to_vec(np.ones((1, 3))).shape
    (1, 3)

    >>> import numpy as np
    >>> to_vec(np.ones((1, 2))).shape
    Traceback (most recent call last):
    ...
    TypeError: Value cannot be converted to vector. Expected shape: (3,) or (N, 3). Got shape: (1, 2).

    >>> to_vec("string")
    Traceback (most recent call last):
    ...
    TypeError: Could not convert vector to np.array. Got: 'string'.
    """
    return to_homogeneous(vec, 1.0)
