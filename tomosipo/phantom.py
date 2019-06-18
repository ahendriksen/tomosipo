import numpy as np


def hollow_box(vd):
    """Fills a volume dataset with a hollow box phantom

    :param vd: `ts.Data`
        A volume dataset.
    :returns:
    :rtype: ts.Data

    """
    shape = np.array(vd.data.shape)

    s20 = shape * 20 // 100
    s40 = shape * 40 // 100

    box_slices = tuple(slice(a, l - a) for (a, l) in zip(s20, shape))
    hollow_slices = tuple(slice(a, l - a) for (a, l) in zip(s40, shape))

    vd.data[:] = 0.0
    vd.data[box_slices] = 1.0
    vd.data[hollow_slices] = 0.0

    return vd
