"""ASTRA conversion and projection code

There are two geometry conversion methods:

- from_astra
- to_astra

An important method is `create_astra_projector`, which creates an ASTRA
projector from a pair of geometries.

Moreover, there is projection code that is centered around the following
ASTRA APIs:

1. astra.experimental.direct_FPBP3D (modern)
2. astra.experimental.do_composite (legacy)

The first is used in modern tomosipo code: it takes an existing ASTRA projector
and a link to a numpy or gpu array.

The second is a legacy approach that is kept for debugging and testing purposes.
It takes multiple Data objects describing volumes (both data and geometry) and
projection geometries (both data and geometry). On this basis, it creates a
projector and passes it to ASTRA, which performs an all-to-all (back)projection.

"""
import astra
import tomosipo as ts


###############################################################################
#                              Convert geometries                             #
###############################################################################


def from_astra(astra_geom):
    """Import ASTRA geometry to tomosipo

    Parameters
    ----------
    astra_geom:
        A 3D ASTRA  volume or projection geometry.

    Returns
    -------
        A tomosipo 3D volume or projection geometry.
    """
    if not isinstance(astra_geom, dict):
        raise TypeError(
            f"Currently, tomosipo only supports importing ASTRA geometries. "
            f"Objects of type {type(astra_geom)} are not supported. "
            f"Perhaps you meant to use `ts.to_astra'? "
        )
    if "GridSliceCount" in astra_geom:
        return ts.geometry.volume.from_astra(astra_geom)
    else:
        return ts.geometry.conversion.from_astra_projection_geometry(astra_geom)


def to_astra(x):
    """Convert tomosipo geometry to ASTRA

    Parameters
    ----------
    x:
        A 3D tomosipo volume or projection geometry.

    Returns
    -------
        A 3D ASTRA volume or projection geometry.
    """
    try:
        return x.to_astra()
    except AttributeError:
        raise TypeError(
            f"The object of type {type(x)} does not support conversion to ASTRA."
            f"Perhaps you meant to use `ts.from_astra'? "
        )


def create_astra_projector(
    volume_geometry,
    projection_geometry,
    *,
    voxel_supersampling=1,
    detector_supersampling=1,
):
    vg, pg = volume_geometry, projection_geometry
    assert isinstance(vg, ts.geometry.VolumeGeometry)

    return astra.create_projector(
        "cuda3d",
        pg.to_astra(),
        vg.to_astra(),
        options={
            "VoxelSuperSampling": voxel_supersampling,
            "DetectorSuperSampling": detector_supersampling,
        },
    )


###############################################################################
#                       Direct ASTRA projection (modern)                      #
###############################################################################
def direct_project(
    projector,
    vol_link,
    proj_link,
    forward=None,
    additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :param vol_link: TODO
    :param proj_link: TODO
    :param forward: bool
        True for forward project, False for backproject.
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :returns:
    :rtype:

    """
    if forward is None:
        raise ValueError("project must be given a forward argument (True/False).")

    # These constants have become the default. See:
    # https://github.com/astra-toolbox/astra-toolbox/commit/4d673b3cdb6d27d430087758a8081e4a10267595
    MODE_SET = 1
    MODE_ADD = 0

    if not ts.links.are_compatible(vol_link, proj_link):
        raise ValueError(
            "Cannot perform ASTRA projection on volume and projection data, because they are not compatible. "
            "Usually, this indicates that the data are located on different computing devices. "
        )

    # If necessary, the link may adjust the current state of the
    # program temporarily to ensure ASTRA runs correctly. For torch
    # tensors, this may entail changing the currently active GPU.
    with vol_link.context():
        astra.experimental.direct_FPBP3D(
            projector,
            vol_link.linked_data,
            proj_link.linked_data,
            MODE_ADD if additive else MODE_SET,
            "FP" if forward else "BP",
        )


def direct_fp(
    projector,
    vol_data,
    proj_data,
    additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :param vol_data: TODO
    :param proj_data: TODO
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :returns:
    :rtype:

    """
    return direct_project(
        projector,
        vol_data,
        proj_data,
        forward=True,
        additive=additive,
    )


def direct_bp(
    projector,
    vol_data,
    proj_data,
    additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :param vol_data: TODO
    :param proj_data: TODO
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :returns:
    :rtype:

    """
    return direct_project(
        projector,
        vol_data,
        proj_data,
        forward=False,
        additive=additive,
    )


###############################################################################
#                          ASTRA projection (legacy)                          #
###############################################################################
def project(
    *data,
    voxel_supersampling=1,
    detector_supersampling=1,
    forward=None,
    additive=False,
    projector=None,
):
    """Project forward or backward

    Projects all volumes on all projection datasets.

    :param \\*data: `Data` objects
        Data to use for forward and back projection. At least one
        data object relating to a VolumeGeometry and at least one
        data object relating to a projection geometry is required.
    :param detector_supersampling: `int`
        For the forward projection, DetectorSuperSampling^2 rays
        will be used.  This should only be used if your detector
        pixels are larger than the voxels in the reconstruction
        volume.  (default: 1)
    :param voxel_supersampling: `int`
        For the backward projection, VoxelSuperSampling^3 rays
        will be used.  This should only be used if your voxels in
        the reconstruction volume are larger than the detector
        pixels.  (default: 1)
    :param forward: bool
        True for forward project, False for backproject.
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :returns:
    :rtype:

    """
    vol_data = [d for d in data if d.is_volume()]
    proj_data = [d for d in data if d.is_projection()]

    if forward is None:
        raise ValueError("project must be given a forward argument (True/False).")

    if len(vol_data) < 1 or len(proj_data) < 1:
        raise ValueError(
            "Expected at least one projection dataset and one volume dataset"
        )
    if projector is None:
        projector = create_astra_projector(
            vol_data[0].geometry,
            proj_data[0].geometry,
            voxel_supersampling=voxel_supersampling,
            detector_supersampling=detector_supersampling,
        )
    # These constants have become the default. See:
    # https://github.com/astra-toolbox/astra-toolbox/commit/4d673b3cdb6d27d430087758a8081e4a10267595
    MODE_SET = 1
    MODE_ADD = 0

    astra.experimental.do_composite(
        projector,
        [d.to_astra() for d in vol_data],
        [d.to_astra() for d in proj_data],
        MODE_ADD if additive else MODE_SET,
        "FP" if forward else "BP",
    )


def forward(
    *data,
    voxel_supersampling=1,
    detector_supersampling=1,
    additive=False,
    projector=None,
):
    """Forward project

    Projects all volumes on all projection datasets.

    :param \\*data: `Data` objects
        Data to use for forward and back projection. At least one
        data object relating to a VolumeGeometry and at least one
        data object relating to a projection geometry is required.
    :param detector_supersampling: `int`
        For the forward projection, DetectorSuperSampling^2 rays
        will be used.  This should only be used if your detector
        pixels are larger than the voxels in the reconstruction
        volume.  (default: 1)
    :param voxel_supersampling: `int`
        For the backward projection, VoxelSuperSampling^3 rays
        will be used.  This should only be used if your voxels in
        the reconstruction volume are larger than the detector
        pixels.  (default: 1)
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :returns:
    :rtype:
    """

    project(
        *data,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
        forward=True,
        projector=projector,
    )


def backward(
    *data,
    voxel_supersampling=1,
    detector_supersampling=1,
    additive=False,
    projector=None,
):
    """Backproject

    Backprojects all projection datasets on all volumes.

    :param \\*data: `Data` objects
        Data to use for forward and back projection. At least one
        data object relating to a VolumeGeometry and at least one
        data object relating to a projection geometry is required.
    :param detector_supersampling: `int`
        For the forward projection, DetectorSuperSampling^2 rays
        will be used.  This should only be used if your detector
        pixels are larger than the voxels in the reconstruction
        volume.  (default: 1)
    :param voxel_supersampling: `int`
        For the backward projection, VoxelSuperSampling^3 rays
        will be used.  This should only be used if your voxels in
        the reconstruction volume are larger than the detector
        pixels.  (default: 1)
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :returns:
    :rtype:

    """
    project(
        *data,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
        forward=False,
        projector=projector,
    )


def fdk(vol_data, proj_data, *, voxel_supersampling=1, detector_supersampling=1):
    """Do an FDK reconstruction of the given geometry.

    Expects a single volume dataset and a single projection dataset to
    be associated with the geometry.

    :param vol_data: Volume dataset to use.
    :param proj_data: Projection dataset to use.
    :param detector_supersampling: `int`
        For the forward projection, DetectorSuperSampling^2 rays
        will be used.  This should only be used if your detector
        pixels are larger than the voxels in the reconstruction
        volume.  (default: 1)
    :param voxel_supersampling: `int`
        For the backward projection, VoxelSuperSampling^3 rays
        will be used.  This should only be used if your voxels in
        the reconstruction volume are larger than the detector
        pixels.  (default: 1)
    :returns:
    :rtype:

    """

    projector = create_astra_projector(
        vol_data.geometry,
        proj_data.geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
    )

    astra.experimental.accumulate_FDK(
        projector, vol_data.to_astra(), proj_data.to_astra()
    )
