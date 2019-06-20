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
