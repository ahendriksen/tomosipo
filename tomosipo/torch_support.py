""" This module provides functions to interoperate with torch

"""
import tomosipo as ts
import warnings
from tomosipo.Operator import BackprojectionOperator

try:
    import torch
except ModuleNotFoundError:
    warnings.warn(
        "\n------------------------------------------------------------\n\n"
        "Cannot import torch package. \n"
        "Please make sure to install torch. \n"
        "You can install torch using: \n\n"
        " > conda install pytorch -c pytorch \n"
        "\n------------------------------------------------------------\n\n"
    )
    raise
from torch.autograd import Function
import itertools


class OperatorFunction(Function):
    @staticmethod
    def forward(ctx, input, operator, num_extra_dims=0, is_2d=False):
        extra_dims = input.size()[:num_extra_dims]
        if input.requires_grad:
            ctx.operator = operator
            ctx.extra_dims = extra_dims
            ctx.is_2d = is_2d

        expected_ndim = (2 if is_2d else 3) + num_extra_dims
        assert (input.ndim == expected_ndim), (
            f"Tomosipo autograd operator expected {expected_ndim} dimensions "
            f"but got {input.ndim}.\n"
            "The interface of to_autograd was changed in Tomosipo 0.6.0 to "
            "by default match standard Tomosipo operators and extra arguments are "
            "provided to match Pytorch NN functions.\n"
            "To add batch and channel dimensions set argument num_extra_dims=2\n"
            "To remove the first operator dimension set argument is_2d=True\n"
        )

        output = input.new_empty(extra_dims + operator.range_shape, dtype=torch.float32)

        if is_2d:
            input = torch.unsqueeze(input, dim=-3)

        if len(extra_dims) == 0:
            operator(input, out=output)
        else:
            for subspace in itertools.product(*[range(dim_size) for dim_size in extra_dims]):
                operator(input[subspace], out=output[subspace])

        if is_2d:
            output = torch.squeeze(output, dim=-3)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        operator = ctx.operator
        extra_dims = ctx.extra_dims
        is_2d = ctx.is_2d

        grad_input = grad_output.new_empty(extra_dims + operator.domain_shape, dtype=torch.float32)

        if is_2d:
            grad_output = torch.unsqueeze(grad_output, dim=-3)

        if len(extra_dims) == 0:
            operator.T(grad_output, out=grad_input)
        else:
            for subspace in itertools.product(*[range(dim_size) for dim_size in extra_dims]):
                operator.T(grad_output[subspace], out=grad_input[subspace])

        if is_2d:
            grad_input = torch.squeeze(grad_input, dim=-3)

        # do not return gradient for operator
        return grad_input, None, None, None


def to_autograd(operator, num_extra_dims=0, is_2d=False):
    """Converts an operator to an autograd function

    Example:

        >>> import tomosipo as ts
        >>> vg = ts.volume(shape=10)
        >>> pg = ts.parallel(angles=10, shape=10)
        >>> A = ts.operator(vg, pg)

        >>> f = to_autograd(A)
        >>> vd = torch.randn(*A.domain_shape, requires_grad=True)
        >>> f(vd).sum().backward()
        >>> vd.grad is not None
        True

    Likewise, you may use the transpose:

        >>> g = to_autograd(A.T)
        >>> pd = torch.randn(*A.T.domain_shape, requires_grad=True)
        >>> g(pd).sum().backward()
        >>> vd.grad is not None
        True

    Parameters
    ----------
    operator: `Operator`
        Tomosipo operator whose behavior will be mimicked

    num_extra_dims : `int` (optional)
        Number of extra dimensions to prepend to the input and output of
        this operator. Set this to 2 to add channel and batch
        dimensions when taining neural networks. The default is 0.

    is_2d : `bool` (optional)
        Whether to remove the first dimension of the operator, resulting in
        a 2D operator. The default is False.

    Returns
    -------
    Function
        An autograd enabled function
    """

    def f(x):
        return OperatorFunction.apply(x, operator, num_extra_dims, is_2d)

    return f


class Forward(Function):
    @staticmethod
    def forward(ctx, input, vg, pg, projector):
        # TODO: support supersampling
        if input.requires_grad:
            # Save geometries for backward pass
            ctx.vg = vg
            ctx.pg = pg
            ctx.projector = projector

        output_type = torch.scalar_tensor(0.0, dtype=torch.float32, device=input.device)
        # Use temporary astra data link for both geometries:
        with ts.data(vg, input) as vd, ts.data(pg, output_type) as pd:
            ts.forward(vd, pd, projector=projector)
            return pd.data

    @staticmethod
    def backward(ctx, grad_output):
        vg = ctx.vg
        pg = ctx.pg
        projector = ctx.projector

        input_type = torch.scalar_tensor(
            0.0, dtype=torch.float32, device=grad_output.device
        )

        with ts.data(vg, input_type) as vd, ts.data(pg, grad_output) as pd:
            ts.backward(vd, pd, projector=projector)
            # Do not return gradients for vg, pg, and projector
            return vd.data, None, None, None


def forward(input, vg, pg, projector=None):
    return Forward.apply(input, vg, pg, projector)


class Backward(Function):
    @staticmethod
    def forward(ctx, input, vg, pg, projector):
        # TODO: support supersampling
        if input.requires_grad:
            # Save geometries for backward pass
            ctx.vg = vg
            ctx.pg = pg
            ctx.projector = projector

        output_type = torch.scalar_tensor(0.0, dtype=torch.float32, device=input.device)
        # Use temporary astra data link for both geometries:
        with ts.data(vg, output_type) as vd, ts.data(pg, input) as pd:
            ts.backward(vd, pd, projector=projector)
            return vd.data

    @staticmethod
    def backward(ctx, grad_output):
        vg = ctx.vg
        pg = ctx.pg
        projector = ctx.projector

        input_type = torch.scalar_tensor(
            0.0, dtype=torch.float32, device=grad_output.device
        )

        with ts.data(vg, grad_output) as vd, ts.data(pg, input_type) as pd:
            ts.backward(vd, pd, projector=projector)
            # Do not return gradients for vg, pg, and projector
            return pd.data, None, None, None


def backward(input, vg, pg, projector=None):
    return Backward.apply(input, vg, pg, projector)


class AutogradOperator():
    """A linear tomographic projection operator with torch autograd support

    This class has almost the same interface as a normal tomosipo operator, but
    it supports pytorch autograd, allowing for automatic differentiation. This
    can be used for example on the reconstruction algorithms in ts_algorithms.
    Additive operators, numpy inputs and outputs, and transformations are not
    supported on AutogradOperators.
    """
    def __init__(
        self,
        operator,
        num_extra_dims=0,
        is_2d=False
    ):
        """
        Create an operator with autograd support from an existing tomosipo
        operator

        Parameters
        ----------
        operator: `Operator`
            Tomosipo operator whose behavior will be mimicked

        num_extra_dims : `int` (optional)
            Number of extra dimensions to prepend to the input and output of
            this operator. Set this to 2 to add channel and batch
            dimensions when taining neural networks. The default is 0.

        is_2d : `bool` (optional)
            Whether to remove the first dimension of the operator, resulting in
            a 2D operator. The default is False.
        """

        if operator.additive:
            raise ValueError("Additive operators are not supported")

        self.operator = operator
        self._fp_op = to_autograd(operator, num_extra_dims, is_2d)
        self._bp_op = to_autograd(operator.T, num_extra_dims, is_2d)
        self._transpose = BackprojectionOperator(self)

    def _fp(self, volume, out=None):
        if out is None:
            return self._fp_op(volume)
        else:
            out[...] = self._fp_op(volume)
            return out

    def _bp(self, projection, out=None):
        if out is None:
            return self._bp_op(projection)
        else:
            out[...] = self._bp_op(projection)
            return out

    def __call__(self, volume, out=None):
        """Apply operator
        
        Parameters
        ----------
        volume: `torch.Tensor`
            An input volume. The shape must match the operator geometry.
        out: `torch.Tensor` (optional)
            An optional output value. The shape must match the operator
            geometry.
            
        Returns
        -------
        `torch.Tensor`
            A projection dataset on which the volume has been forward
            projected.

        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        """The transpose operator

        This property returns the transpose (backprojection) operator.
        """
        return self.transpose()

    @property
    def astra_compat_vg(self):
        return self.operator.astra_compat_vg

    @property
    def astra_compat_pg(self):
        return self.operator.astra_compat_pg

    @property
    def domain(self):
        """The domain (volume geometry) of the operator"""
        return self.operator.domain

    @property
    def range(self):
        """The range (projection geometry) of the operator"""
        return self.operator.range

    @property
    def domain_shape(self):
        """The expected shape of the input (volume) data"""
        return self.operator.domain_shape

    @property
    def range_shape(self):
        """The expected shape of the output (projection) data"""
        return self.operator.range_shape


def autograd_operator(volume_geometry, projection_geometry, voxel_supersampling=1, detector_supersampling=1,
                      num_extra_dims=0, is_2d=False):
    """
    Create a tomographic operator with PyTorch autograd support.

    Parameters
    ----------
    volume_geometry: `VolumeGeometry`
        The domain of the operator.

    projection_geometry:  `ProjectionGeometry`
        The range of the operator.

    voxel_supersampling: `int` (optional)
        Specifies the amount of voxel supersampling, i.e., how
        many (one dimension) subvoxels are generated from a single
        parent voxel. The default is 1.

    detector_supersampling: `int` (optional)
        Specifies the amount of detector supersampling, i.e., how
        many rays are cast per detector. The default is 1.

    num_extra_dims : `int` (optional)
        Number of extra dimensions to prepend to the input and output of this
        operator. This can be useful to add channel and batch dimensions when
        training neural networks. The default is 0.

    is_2d : `bool` (optional)
        Whether to remove the first dimension of the operator, resulting in a
        2D operator. The default is False.

    Returns
    -------
    `AutogradOperator`
        A tomographic operator with PyTorch autograd support.

    """

    op = ts.operator(
        volume_geometry=volume_geometry,
        projection_geometry=projection_geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling
    )
    return AutogradOperator(
        operator=op,
        num_extra_dims=num_extra_dims,
        is_2d=is_2d
    )
