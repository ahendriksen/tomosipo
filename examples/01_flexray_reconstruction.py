import tomosipo as ts
import tifffile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from contextlib import suppress

###############################################################################
#                                  Parameters                                 #
###############################################################################
binning = 4
path = "/export/scratch2/hendriks/datasets/oatmeal/zoom1/"
path = Path(path).expanduser().resolve()

###############################################################################
#                                  Load data                                  #
###############################################################################


# We use the following function to load a stack of images:
def load_stack(paths, binning=1):
    """Load a stack of tiff files.

    :param paths: paths to tiff files
    :param binning: whether angles and projection images should be binned.
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    # Read first image for shape and dtype information
    paths = list(paths)
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    dtype = img0.dtype
    # Create empty numpy array to hold result
    imgs = np.empty((len(paths), *img0.shape), dtype=dtype)
    for i, p in tqdm(enumerate(paths)):
        imgs[i] = tifffile.imread(str(p))[::binning, ::binning]
    return imgs


# Only read every `binning' image:
proj_paths = sorted(path.glob("scan_*.tif"))[::binning]  # bin angles
dark = load_stack(path.glob("di00*.tif"), binning)
flat = load_stack(path.glob("io00*.tif"), binning)
proj = load_stack(proj_paths, binning)

###############################################################################
#                                Flat fielding                                #
###############################################################################

# Do flat fielding.  You typically want to do this with the "numexpr"
# package, because it uses all cpu cores. See:
# https://numexpr.readthedocs.io/en/latest/intro.html#expected-performance
flat = (flat - dark).mean(axis=0)
proj = (proj - dark) / flat[None, :, :]
proj = -np.log(proj).astype("float32")

# Convert to sinogram data that is compatible with Astra:
proj = np.transpose(
    proj, [1, 0, 2]
)  # Angles in the middle, "up" in front, "right" at the back.
proj = np.flipud(proj)  # Flip image in the vertical direction
proj = np.ascontiguousarray(proj)  # Ensure contiguous layout in memory

###############################################################################
#                              Create geometries                              #
###############################################################################


def parse_flexray_log(path):
    # This function parses a flexraylog. I have tried to keep it
    # short but readable. It is a bit of a code golf, see:
    # https://en.wikipedia.org/wiki/Code_golf
    settings = Path(path).expanduser().resolve().read_text()
    ls = settings.strip().lower().splitlines()
    ls = [l for l in ls if ":" in l]

    names, vars = zip(*[l.split(":")[:2] for l in ls])
    names = [str.replace(s.strip(), " ", "_") for s in names]
    names = [str.replace(s, "roi_(ltrb)", "roi_ltrb") for s in names]
    d = dict(zip(names, vars))
    for k, v in d.items():
        if k == "roi_ltrb":
            l, t, r, b = map(int, v.strip().split(","))
            d[k] = (l, t, r, b)
        else:
            with suppress(ValueError, IndexError):
                d[k] = float(v.split()[0])

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    return AttrDict(d)


s = parse_flexray_log(path / "scan settings.txt")
for k, v in s.items():
    print(f"{k:<20}: {v}")

num_angles = proj.shape[1]
original_shape = np.array((1536, 1944))
s.original_pixel_size = 0.074800  # Sometimes the file is wrong
det_size = s.original_pixel_size * original_shape

# Note: mag_tube is always zero
# Note: ver_obj can be really big, so use ver_tube instead.
# Note: Z, Y, X coordinate order
src_pos = s.ver_tube, 0, s.tra_tube
obj_pos = s.ver_tube, s.mag_obj, s.tra_obj
det_pos = s.ver_det, s.mag_det, s.tra_det

# The object rotates clockwise:
angles = -np.linspace(0, 2 * np.pi, num_angles, endpoint=True)
src = ts.box(size=det_size[0] / 10, pos=src_pos)
det = ts.box((det_size[0], 0, det_size[1]), pos=det_pos)
R = ts.rotate(obj_pos, (1, 0, 0), rad=angles)
vol = R(ts.box(size=det_size[0] / 5, pos=obj_pos))

# We can visualize the geometry. NB: Do zoom out to see the whole
# scene.
ts.display(src, vol, det)

# Apply correction profile 'cwi-flexray-2019-04-24':
det_tan = ts.translate((0, 0, 24))  # 'det_tan': 24,
src_ort = ts.translate((-7, 0, 0))  # 'src_ort': -7,
axs_tan = ts.translate((0, 0, -0.5))  # 'axs_tan': -0.5,
src_cor = src_ort(src)
vol_cor = axs_tan(vol)
det_cor = det_tan(det)

# We can visualize the corrections by displaying the original
# configuration alongside the corrected configuration. Note that you
# can recognize the original configuration by the fact that their
# boxes have the same color as they did in the previous visualization.
ts.display(src, vol, det, src_cor, vol_cor, det_cor)

# We can also visualize the acquisition geometry from the perspective
# of the volume. This should look familiar to those accustomed to
# Astra projection geometries.
P = ts.from_perspective(box=vol_cor)
ts.display(P(src_cor), P(vol_cor), P(det_cor))

# We create projection geometry using the positions that we have
# determined above:
pg = ts.cone_vec(
    shape=original_shape,
    source_positions=P(src_cor).pos,
    detector_positions=P(det_cor).pos,
    # w is the upward pointing vector of the detector box.
    detector_vs=s.original_pixel_size * P(det_cor).w,
    detector_us=s.original_pixel_size * P(det_cor).u,
)
# For most reasonable geometries, tomosipo can deduce the size and
# position of a reasonable volume geometry for us.
vg = ts.volume_from_projection_geometry(pg)
voxel_size = s.sod / s.sdd * s.original_pixel_size * s.binning_value * binning
vg = vg.with_voxel_size(voxel_size)
ts.display(pg, vg)

# Sometimes, only a region of interest is acquired on the detector. In
# that case, we can adjust the projection geometry to take this into
# account. The `t` and `b`, referring to top and bottom, are the
# "wrong way around", since t < b. Hence, we swap them
# below. Moreover, all values are indices, so we must add 1 to the
# right hand size of the slices.
l, t, r, b = s["roi_ltrb"]
pg = pg[:, t : b + 1, l : r + 1]

# When the flexray scanner has already binned the detector values, we
# must increase the detector pixel size. This does not alter the
# dimensions of the detector!
pg = pg.rescale_det(s.binning_value)

# The projection data has also been binned in the loading script
# above, so we obtain a subgeometry with the same binning applied:
pg = pg[:, ::binning, ::binning]


###############################################################################
#                                Reconstruction                               #
###############################################################################

vg_slice = vg[vg.shape[0] // 2 : vg.shape[0] // 2 + 10]
vd = ts.data(vg_slice)
pd = ts.data(pg, proj)

ts.fdk(vd, pd)

###############################################################################
#                            Display reconstruction                           #
###############################################################################

ts.display(vd)
ts.display(pd)
