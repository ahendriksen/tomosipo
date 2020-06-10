"""
This example requires numexpr:

conda install numexpr tifffile pyqtgraph tqdm -c defaults -c conda-forge
pip install git+https://github.com/ahendriksen/tomosipo.git@develop

"""
import tomosipo as ts
import tifffile
import numexpr as ne
from pathlib import Path
from tqdm import tqdm
import numpy as np
from contextlib import suppress
import pyqtgraph as pq
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree


###############################################################################
#                                  Parameters                                 #
###############################################################################
initial_binning = 1
path = "/export/scratch1/hendriks/noisy"
path = Path(path).expanduser().resolve()

###############################################################################
#                                  Load data                                  #
###############################################################################


def load_sino(paths, binning=1, dtype=None):
    """Load a stack of tiff files into a sinogram

    :param paths: paths to tiff files
    :param binning: whether angles and projection images should be binned.
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    # Read first image for shape and dtype information
    paths = list(paths)
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    if dtype is None:
        dtype = img0.dtype
    # Create empty numpy array to hold result
    imgs = np.empty((img0.shape[0], len(paths), img0.shape[1]), dtype=dtype)
    for i, p in tqdm(enumerate(paths)):
        # Angles in the middle, "up" in front, "right" at the back.
        # Flip in the vertical direction
        imgs[:, i, :] = tifffile.imread(str(p))[::-binning, ::binning]
    return imgs


# Only read every `binning' image:
proj_paths = sorted(path.glob("scan_*.tif"))[::initial_binning]  # bin angles
dark = load_sino(path.glob("di00*.tif"), initial_binning, dtype=np.float32)
flat = load_sino(path.glob("io00*.tif"), initial_binning, dtype=np.float32)
proj = load_sino(proj_paths, initial_binning, dtype=np.float32)

###############################################################################
#                                Flat fielding                                #
###############################################################################

# Do flat fielding.  You typically want to do this with the "numexpr"
# package, because it uses all cpu cores. See:
# https://numexpr.readthedocs.io/en/latest/intro.html#expected-performance
flat_mean = (flat - dark).mean(axis=1, dtype=np.float32)[:, None, :]

# proj = (proj - dark) / flat[None, :, :]
# proj = -np.log(proj).astype('float32')
ne.evaluate("-log((proj - dark) / ((flat_mean - dark)))", out=proj)

###############################################################################
#                              Parse FlexRay log                              #
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

###############################################################################
#                           Reconstruction function                           #
###############################################################################


def reconstruct(proj, param_dict):
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
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=True)
    angles += param_dict["Angle offset"]
    if param_dict["Rotate clockwise"]:
        angles *= -1
    src = ts.box(size=1, pos=src_pos)
    det = ts.box((det_size[0], 0, det_size[1]), pos=det_pos)
    R = ts.rotate(obj_pos, (1, 0, 0), rad=angles)
    vol = R * ts.box(size=det_size[0] / 5, pos=obj_pos)

    axes = ["Z", "Y", "X"]
    src_coords = tuple(param_dict[" ".join(["Src", ax])] for ax in axes)
    vol_coords = tuple(param_dict[" ".join(["Vol", ax])] for ax in axes)
    det_coords = tuple(param_dict[" ".join(["Det", ax])] for ax in axes)

    src_tr = ts.translate(src_coords)
    vol_tr = ts.translate(vol_coords)
    det_tr = ts.translate(det_coords)
    src_cor = src_tr * src
    vol_cor = vol_tr * vol
    det_cor = det_tr * det

    P = ts.from_perspective(box=vol_cor)
    # We create projection geometry using the positions that we have
    # determined above:
    pg = ts.cone_vec(
        shape=original_shape,
        src_pos=(P * src_cor).pos,
        det_pos=(P * det_cor).pos,
        # w is the upward pointing vector of the detector box.
        det_v=s.original_pixel_size * (P * det_cor).w,
        det_u=s.original_pixel_size * (P * det_cor).u,
    )
    # Determine volume geometry
    vol_ver_voxels = param_dict["Vol vertical voxels"]
    vol_hor_voxels = param_dict["Vol horizontal voxels"]
    vol_shape = np.array((vol_ver_voxels, vol_hor_voxels, vol_hor_voxels))
    voxel_size = (
        s.sod / s.sdd * s.original_pixel_size * s.binning_value * initial_binning
    )
    vol_size = voxel_size * vol_shape
    vol_center = (
        param_dict["Vol Z"],
        param_dict["Vol Y"],
        param_dict["Vol X"],
    )
    vg = ts.volume(shape=vol_shape, center=vol_center, size=vol_size)
    # vg = ts.volume_from_projection_geometry(pg)
    # vg = vg.with_voxel_size(voxel_size)

    # Take region of interest on detector plane
    l, t, r, b = s["roi_ltrb"]
    pg = pg[:, t : b + 1, l : r + 1]
    # Apply hardware binning
    pg = pg.rescale_det(s.binning_value)
    # Apply initial software binning (see loading data above)
    pg = pg[:, ::initial_binning, ::initial_binning]
    # Apply additional software binning:
    additional_binning = param_dict["Additional binning"]
    pg = pg[::additional_binning, ::additional_binning, ::additional_binning]

    print(pg)
    print("det_pos", pg.det_pos)
    print("src_pos", pg.src_pos)
    print("vol_pos", vg.get_center())
    ###############################################################################
    #                                Reconstruction                               #
    ###############################################################################

    num_slices = param_dict["Num slices"]
    v_xy = ts.data(vg[vg.shape[0] // 2 : vg.shape[0] // 2 + num_slices, :, :])
    v_zx = ts.data(vg[:, vg.shape[0] // 2 : vg.shape[0] // 2 + num_slices, :])
    v_zy = ts.data(vg[:, :, vg.shape[0] // 2 : vg.shape[0] // 2 + num_slices])

    pd = ts.data(
        pg, proj[::additional_binning, ::additional_binning, ::additional_binning]
    )

    with v_xy, v_zx, v_zy, pd:
        for v, opt in zip([v_xy, v_zx, v_zy], ["Show XY", "Show ZX", "Show ZY"]):
            if param_dict[opt]:
                ts.fdk(v, pd)

        return (np.copy(v_xy.data), np.copy(v_zx.data), np.copy(v_zy.data))


###############################################################################
#                                     GUI                                     #
###############################################################################

app = QtGui.QApplication([])
# Create window with two columns
win = QtGui.QMainWindow()
win.resize(800, 800)
win.setWindowTitle("Example 04: FleX-ray Calibration")
root = QtGui.QWidget()
win.setCentralWidget(root)
grid_layout = QtGui.QGridLayout()
root.setLayout(grid_layout)


# ImageView
img_view_xy = pq.ImageView()
img_view_zx = pq.ImageView()
img_view_zy = pq.ImageView()


# Parameter Tree

params = [
    {
        "name": "Geometry",
        "type": "group",
        "children": [
            {"name": "Src Z", "type": "float", "value": -7.0, "step": 0.1},
            {"name": "Src Y", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Src X", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Vol Z", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Vol Y", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Vol X", "type": "float", "value": -0.5, "step": 0.1},
            {"name": "Angle offset", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Vol vertical voxels", "type": "int", "value": 1600, "step": 1},
            {"name": "Vol horizontal voxels", "type": "int", "value": 1600, "step": 1},
            {"name": "Det Z", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Det Y", "type": "float", "value": 0.0, "step": 0.1},
            {"name": "Det X", "type": "float", "value": 24.0, "step": 0.1},
            {"name": "Rotate clockwise", "type": "bool", "value": True},
            {"name": "Additional binning", "type": "int", "value": 4, "step": 1},
            {"name": "Show XY", "type": "bool", "value": True},
            {"name": "Show ZX", "type": "bool", "value": False},
            {"name": "Show ZY", "type": "bool", "value": False},
            {"name": "Num slices", "type": "int", "value": 1},
        ],
    }
]

# Create tree of Parameter objects
parameter = Parameter.create(name="params", type="group", children=params)


def make_param_dict():
    d = {}
    axes = ["Z", "Y", "X"]
    for obj in ["Src", "Vol", "Det"]:
        for ax in axes:
            d[" ".join([obj, ax])] = 0.0

    d["Angle offset"] = 0.0
    d["Vol vertical voxels"] = 100
    d["Vol horizontal voxels"] = 100
    d["Rotate clockwise"] = True
    d["Additional binning"] = 4
    d["Show XY"] = True
    d["Show ZX"] = False
    d["Show ZY"] = False
    d["Num slices"] = 1
    return d


def change(param, changes):
    d = make_param_dict()
    for k in d.keys():
        d[k] = parameter.children()[0][k]

    for k, v in d.items():
        print(f"{k:<10}: {v}")

    v_xy, v_zx, v_zy = reconstruct(proj, d)
    img_view_xy.setImage(v_xy, scale=(1, -1), axes=dict(zip("tyx", range(3))))
    img_view_zx.setImage(v_zx, scale=(1, -1), axes=dict(zip("ytx", range(3))))
    img_view_zy.setImage(v_zy, scale=(1, -1), axes=dict(zip("yxt", range(3))))


parameter.sigTreeStateChanged.connect(change)

parameter_tree = ParameterTree()
parameter_tree.setParameters(parameter, showTop=False)

# Add widgets to layout
grid_layout.addWidget(parameter_tree, 0, 0, 2, 1)
grid_layout.addWidget(img_view_xy, 0, 1, 1, 1)
grid_layout.addWidget(img_view_zx, 0, 2, 1, 1)
grid_layout.addWidget(img_view_zy, 1, 1, 1, 1)
grid_layout.setColumnStretch(1, 5)
grid_layout.setColumnStretch(2, 5)

# Do initial reconstruction:
change(None, None)

# Show window
win.show()
app.exec_()
