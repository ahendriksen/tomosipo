import collections


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


def check_same_shapes(*args):
    shapes = [x.shape for x in args]
    if min(shapes) != max(shapes):
        raise ValueError(f"Not all arguments are the same shape. Got: {shapes}")
