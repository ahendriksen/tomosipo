"""Example of reconstructing with a helical geometry

Requirements:

! conda install -c astra-toolbox/label/dev pyqtgraph pyqt5 astra-toolbox pyopengl numpy
! pip install -e /path/to/tomosipo


This file explains how you can obtain a helical geometry from a
standard circular cone beam geometry using tomosipo.

In the process,

"""
import numpy as np
import tomosipo as ts

# Set up some parameters for the helical scanning geometry.
helix_height = 2
rotations = 2
num_angles = 2000
detector_size = (.8, 3)
detector_shape = (200, 750)
source_detector_distance = 2

# First, we create a conventional circular cone beam geometry with the
# specified sizes. We will transform it into a helical scan later.
# For the volume, take a unit cube on the origin with 128x128x128 voxels.
angles = np.linspace(0, rotations * 2 * np.pi, num_angles)
pg = ts.cone(
    angles,
    detector_size,
    detector_shape,
    src_obj_dist=source_detector_distance / 2,
    src_det_dist=source_detector_distance,
)
vg = ts.volume(128)

# We can easily inspect what we have created by printing to the
# console, or by visualizing the geometries.
print(pg)
print(vg)
ts.display(pg, vg)

# Now that we have a circular cone beam scan, let's transform it into
# a helical scan. First create a translation that first moves the
# source-detector pair up by `helix_height / 2` and the continuously
# moves it down to `-helix_height / 2`. Note that the order of the
# axes in tomosipo is always Z, Y, X. Since we are creating a
# translation for each time step (angle), we create an array with
# shape `(num_angles, 3)`, where the 3 columns are used to store the
# translation in the Z, Y, and X axes respectively.
time = np.linspace(0, 1, num_angles)
t = np.stack([helix_height / 2 - 2 * time, 0 * time, 0 * time], axis=1)
print(t.shape)
T = ts.translate(t)

# Once we have obtained a translation, we can apply it to the
# projection geometry we already have and visualize the result:
helical_pg = T * pg
ts.display(helical_pg, vg)

# XXX: Why do we want to change everything into boxes now?
# We can also visualize the geometry from the perspective of the
# detector.
src_box, det_box = helical_pg.to_box()
vol_box = vg.to_box()
ts.display(src_box, vol_box, det_box)

# This is the same as what we have seen before. However, we may also
# look at it from the perspective of the detector!
S = ts.from_perspective(box=det_box)
ts.display(S * src_box, S * vol_box, S * det_box)

# What if the detector is actually a bit tilted? Let's say 10 degrees.
rot10 = ts.rotate(det_box.pos, det_box.v, deg=10)
# This is what it looks like:
ts.display(src_box, vol_box, rot10 * det_box)

# And from the perspective of the detector:
S = ts.from_perspective(box=rot10 * det_box)
ts.display(S * src_box, S * vol_box, S * det_box)

# Alright. That looks like our acquisition geometry
ts.display(vg, rot10 * helical_pg)

# Now let's do some tomography! First generate data.
vd = ts.data(vg)
pd = ts.data(rot10 * helical_pg)

# Let's put a hollow box phantom in the volume and visualize it.
ts.phantom.hollow_box(vd)
ts.display(vd)

# Forward project and visualize.
ts.forward(vd, pd)
ts.display(pd)

# TODO: the joy of non-uniform pixel sizes..

# To do a reconstruction, we use SIRT. But to use SIRT, we need C and
# R matrices. This is what we are going to calculate with two
# temporary data objects, helpfully named vd_tmp and pd_tmp.
with ts.data(vg) as vd_tmp, ts.data(rot10 * helical_pg) as pd_tmp:
    # Calculate R
    vd_tmp.data[:] = 1.0
    ts.forward(vd_tmp, pd_tmp)
    pd_tmp.data[pd_tmp.data < ts.epsilon] = np.Inf
    np.reciprocal(pd_tmp.data, out=pd_tmp.data)
    R = np.copy(pd_tmp.data)  # <---- copy

    # Calculate C
    pd_tmp.data[:] = 1.0
    ts.backward(vd_tmp, pd_tmp)
    vd_tmp.data[vd_tmp.data < ts.epsilon] = np.Inf
    np.reciprocal(vd_tmp.data, out=vd_tmp.data)
    C = np.copy(vd_tmp.data)  # <---- copy

# After the with statement, all the data has been cleaned up. That is
# why we had to copy R and C to new separate numpy arrays.

# We can also try to do naive iteration, like so:
def iter_naive(vd, pd):
    with vd.clone() as vdiff, pd.clone() as pdiff:
        ts.forward(vd, pdiff)
        pdiff.data[:] -= pd.data
        print(np.square(pdiff.data).mean())
        ts.backward(vdiff, pdiff)
        vd.data[:] -= vdiff.data


# Let us first clear vd, since it still contains the original phantom.
vd.data[:] = 0.0

# Then do a single iterative iteration
iter_naive(vd, pd)
ts.display(vd)
ts.display(pd)
# hmm.. not there yet.

for _ in range(100):
    iter_naive(vd, pd)
ts.display(vd)

# Let's see if we can make it converge faster :)


def iter_SIRT(vd, pd, R, C, iterations=1):
    with vd.clone() as vdiff, pd.clone() as pdiff:
        for _ in range(iterations):
            ts.forward(vd, pdiff)
            pdiff.data[:] -= pd.data
            pdiff.data[:] *= R
            print(np.square(pdiff.data).mean())
            ts.backward(vdiff, pdiff)
            vdiff.data[:] *= C
            vd.data[:] -= vdiff.data


# Clear the volume again.. No Cheating!
vd.data[:] = 0
iter_SIRT(vd, pd, R, C, 1)
ts.display(vd)

iter_SIRT(vd, pd, R, C, 100)
ts.display(vd)

# There are some distinct artifacts. In theory, they shouldn't be
# there and you can see that there is still a difference on the
# forward projection:
with pd.clone() as tmp:
    ts.forward(vd, tmp)
    tmp.data[:] *= -1
    tmp.data[:] += pd.data
    ts.display(tmp)

# This whole business could actually be done quicker:
A = ts.operator(vg, rot10 * helical_pg)
with A(vd) as tmp:
    tmp.data[:] -= pd.data
    ts.display(tmp)
