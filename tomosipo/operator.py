import astra


def astra_projector(
    volume_geometry,
    projection_geometry,
    *,
    voxel_supersampling=1,
    detector_supersampling=1
):
    return astra.create_projector(
        "cuda3d",
        projection_geometry.to_astra(),
        volume_geometry.to_astra(),
        options={
            "VoxelSuperSampling": voxel_supersampling,
            "DetectorSuperSampling": detector_supersampling,
        },
    )


def project(
    *data, voxel_supersampling=1, detector_supersampling=1, forward=None, additive=False
):
    """Project forward or backward

    Projects all volumes on all projection datasets.

    :param *data: `Data` objects
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
    projector = astra_projector(
        vol_data[0].geometry,
        proj_data[0].geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
    )
    MODE_SET = 0
    MODE_ADD = 1

    mode = MODE_ADD if additive else MODE_SET
    t = "FP" if forward else "BP"
    astra.experimental.do_composite(
        projector,
        [d.to_astra() for d in vol_data],
        [d.to_astra() for d in proj_data],
        mode,
        t,
    )


def forward(*data, voxel_supersampling=1, detector_supersampling=1, additive=False):
    """Forward project

    Projects all volumes on all projection datasets.

    :param *data: `Data` objects
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
    :returns:
    :rtype:
    """

    project(
        *data,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
        forward=True
    )


def backward(*data, voxel_supersampling=1, detector_supersampling=1, additive=False):
    """Backproject

    Backprojects all projection datasets on all volumes.

    :param *data: `Data` objects
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
    :returns:
    :rtype:

    """
    project(
        *data,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
        forward=False
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

    projector = astra_projector(
        vol_data.geometry,
        proj_data.geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
    )

    astra.experimental.accumulate_FDK(
        projector, vol_data.to_astra(), proj_data.to_astra()
    )
