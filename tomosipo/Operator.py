import astra
import tomosipo as ts


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
        *data, voxel_supersampling=1, detector_supersampling=1, forward=None, additive=False, projector=None,
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
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector' to generate an astra projector.
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
        projector = astra_projector(
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


def forward(*data, voxel_supersampling=1, detector_supersampling=1, additive=False, projector=None):
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
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector' to generate an astra projector.
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


def backward(*data, voxel_supersampling=1, detector_supersampling=1, additive=False, projector=None):
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
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector' to generate an astra projector.
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

    projector = astra_projector(
        vol_data.geometry,
        proj_data.geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
    )

    astra.experimental.accumulate_FDK(
        projector, vol_data.to_astra(), proj_data.to_astra()
    )


def operator(
    volume_geometry,
    projection_geometry,
    voxel_supersampling=1,
    detector_supersampling=1,
    additive=False,
):
    """Create a new tomographic operator

    :param volume_geometry: `VolumeGeometry`
        The domain of the operator.
    :param projection_geometry:  `ProjectionGeometry`
        The range of the operator.
    :param voxel_supersampling: `int` (optional)
        Specifies the amount of voxel supersampling, i.e., how
        many (one dimension) subvoxels are generated from a single
        parent voxel. The default is 1.
    :param detector_supersampling: `int` (optional)
        Specifies the amount of detector supersampling, i.e., how
        many rays are cast per detector. The default is 1.
    :param additive: `bool` (optional)
        Specifies whether the operator should overwrite its range
        (forward) and domain (transpose). When `additive=True`,
        the operator adds instead of overwrites. The default is
        `additive=False`.
    :returns:
    :rtype:

    """
    return Operator(
        volume_geometry,
        projection_geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
    )


def direct_project(
        projector, vol_link, proj_link, forward=None, additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    # TODO: Describe vol_data and proj_data
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector' to generate an astra projector.

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

    with vol_link.context():
        astra.experimental.direct_FPBP3D(
            projector,
            vol_link.linked_data,
            proj_link.linked_data,
            MODE_ADD if additive else MODE_SET,
            "FP" if forward else "BP",
        )


def direct_fp(
        projector, vol_data, proj_data, additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    # TODO: Describe vol_data and proj_data
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector' to generate an astra projector.
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
        projector, vol_data, proj_data, additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    # TODO: Describe vol_data and proj_data
    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector' to generate an astra projector.
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


class Operator(object):
    """Documentation for Operator

    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry,
        voxel_supersampling=1,
        detector_supersampling=1,
        additive=False,
    ):
        """Create a new tomographic operator

        :param volume_geometry: `VolumeGeometry`
            The domain of the operator.
        :param projection_geometry:  `ProjectionGeometry`
            The range of the operator.
        :param voxel_supersampling: `int` (optional)
            Specifies the amount of voxel supersampling, i.e., how
            many (one dimension) subvoxels are generated from a single
            parent voxel. The default is 1.
        :param detector_supersampling: `int` (optional)
            Specifies the amount of detector supersampling, i.e., how
            many rays are cast per detector. The default is 1.
        :param additive: `bool` (optional)
            Specifies whether the operator should overwrite its range
            (forward) and domain (transpose). When `additive=True`,
            the operator adds instead of overwrites. The default is
            `additive=False`.
        :returns:
        :rtype:

        """
        super(Operator, self).__init__()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry

        self.astra_projector = astra_projector(
            volume_geometry,
            projection_geometry,
            voxel_supersampling=voxel_supersampling,
            detector_supersampling=detector_supersampling,
        )
        self.additive = additive
        self._transpose = BackprojectionOperator(self)

    def _fp(self, volume, out=None):
        vlink = ts.link(self.volume_geometry, volume)

        if out is not None:
            plink = ts.link(self.projection_geometry, out)
        else:
            shape = ts.links.base.geometry_shape(self.projection_geometry)
            if self.additive:
                plink = vlink.new_zeros(shape)
            else:
                plink = vlink.new_empty(shape)

        direct_fp(
            self.astra_projector,
            vlink,
            plink,
            additive=self.additive
        )

        return plink.data

    def _bp(self, projection, out=None):
        """Apply backprojection

        *Note*: when `projection` is not an instance of `Data`, then
         this function leaks memory. An intermediate `Data` element is
         created for the projection that is not freed, and it cannot
         be freed by the caller. Therefore, it is recommended to only
         use numpy arrays as input, for small data or one-off scripts.

        :param projection: `np.array` or `Data`
            An input projection dataset. If a numpy array, the shape
            must match the operator geometry. If the projection dataset is
            an instance of `Data`, its geometry must match the
            operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A volume dataset on which the projection dataset has been
            backprojected.
        :rtype: `Data`

        """
        plink = ts.link(self.projection_geometry, projection)

        if out is not None:
            plink = ts.link(self.volume_geometry, out)
        else:
            shape = ts.links.base.geometry_shape(self.volume_geometry)
            if self.additive:
                vlink = plink.new_zeros(shape)
            else:
                vlink = plink.new_empty(shape)

        direct_bp(
            self.astra_projector,
            vlink,
            plink,
            additive=self.additive,
        )

        return vlink.data

    def __call__(self, volume, out=None):
        """Apply operator

        :param volume: `np.array` or `Data`
            An input volume. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        return self.transpose()


class BackprojectionOperator(object):
    """Transpose of the Forward operator

    The idea of having a dedicated class for the backprojection
    operator, which just saves a link to the "real" operator has
    been shamelessly ripped from OpTomo.

    We have the following property:

    >>> op = ts.operator(vg, pg)
    >>> op.T == op.T.T.T

    It is nice that we do not allocate a new object every time we use
    `op.T'. If we did, users might save the transpose in a separate
    variable for 'performance reasons', writing

    >>> op = ts.operator(vg, pg)
    >>> op_T = op.T

    This is a waste of time.
    """

    def __init__(
            self,
            parent,
    ):
        """Create a new tomographic operator
        """
        super(BackprojectionOperator, self).__init__()
        self.parent = parent

    def __call__(self, projection, out=None):
        """Apply operator

        :param projection: `np.array` or `Data`
            An input projection. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self.parent._bp(projection, out)

    def transpose(self):
        """Return forward projection operator"""
        return self.parent

    @property
    def T(self):
        return self.transpose()
