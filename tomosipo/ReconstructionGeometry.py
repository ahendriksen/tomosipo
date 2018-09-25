import astra
import numpy as np
import tomosipo as ts
from .utils import up_tuple


def fdk(reconstruction_geometry):
    """Do an FDK reconstruction of the given geometry.

    Expects a single volume dataset and a single projection dataset to
    be associated with the geometry.

    :param reconstruction_geometry: `ReconstructionGeometry`
        A reconstruction geometry with a single projection dataset and
        a single volume dataset.
    :returns: None
    :rtype: NoneType

    """
    r = reconstruction_geometry

    if len(r.projection_data) > 1:
        raise ValueError(
            "ReconstructionGeometry has more than one projection dataset. Expected one."
        )
    if len(r.volume_data) > 1:
        raise ValueError(
            "ReconstructionGeometry has more than one volume dataset. Expected one."
        )

    projector = r.astra_projector
    v = r.astra_vol_data[0]
    p = r.astra_proj_data[0]

    astra.experimental.accumulate_FDK(projector, v, p)


class ReconstructionGeometry(object):
    """ReconstructionGeometry handles reconstruction parameters and object lifetimes

    """

    def __init__(self, *data, detector_supersampling=1, voxel_supersampling=1):
        """Create a new reconstruction object

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
        self.projection_data = []
        self.astra_proj_data = []
        self.volume_data = []
        self.astra_vol_data = []

        self.voxel_supersampling = voxel_supersampling
        self.detector_supersampling = detector_supersampling

        for d in data:
            try:
                if d.is_projection():
                    self.projection_data.append(d)
                    self.astra_proj_data.append(d.to_astra())
                elif d.is_volume():
                    self.volume_data.append(d)
                    self.astra_vol_data.append(d.to_astra())
            except AttributeError:
                raise ValueError(
                    f"Given object with class {d.__class__}; expected: `VolumeGeometry` or `ProjectionGeometry`"
                )

        if len(self.projection_data) < 1:
            raise ValueError(
                "ReconstructionGeometry requires at least one projection dataset"
            )
        if len(self.volume_data) < 1:
            raise ValueError(
                "ReconstructionGeometry requires at least one volume dataset"
            )

        # Set astra projector
        self.astra_projector = self.__astra_projector()

    def __astra_projector(self):
        assert len(self.projection_data) > 0
        assert len(self.volume_data) > 0

        return astra.create_projector(
            "cuda3d",
            self.projection_data[0].geometry.to_astra(),
            self.volume_data[0].geometry.to_astra(),
            options={
                "VoxelSuperSampling": self.voxel_supersampling,
                "DetectorSuperSampling": self.detector_supersampling,
            },
        )

    def forward(self):
        astra.experimental.do_composite_FP(
            self.astra_projector, self.astra_vol_data, self.astra_proj_data
        )

    def backward(self):
        astra.experimental.do_composite_BP(
            self.astra_projector, self.astra_vol_data, self.astra_proj_data
        )
