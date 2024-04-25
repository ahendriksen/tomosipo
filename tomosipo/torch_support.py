""" This module provides functions to interoperate with torch

"""
import tomosipo as ts
import warnings

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
    def forward(input, operator, num_extra_dims=0, is_2d=False):
        extra_dims = input.size()[:num_extra_dims]

        expected_ndim = (2 if is_2d else 3) + num_extra_dims
        assert (
            input.ndim == expected_ndim
        ), (f"Tomosipo autograd operator expected {expected_ndim} dimensions "
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
    def setup_context(ctx, inputs, output):
        _, operator, num_extra_dims, is_2d = inputs
        ctx.operator = operator
        ctx.num_extra_dims = num_extra_dims
        ctx.is_2d = is_2d

    @staticmethod
    def backward(ctx, grad_output):
        operator = ctx.operator
        num_extra_dims = ctx.num_extra_dims
        is_2d = ctx.is_2d

        grad_input = OperatorFunction.apply(grad_output, operator.T, num_extra_dims, is_2d)

        # do not return gradient for operator
        return grad_input, None, None, None

    @staticmethod
    def jvp(ctx, grad_input, *args):
        operator = ctx.operator
        num_extra_dims = ctx.num_extra_dims
        is_2d = ctx.is_2d

        return OperatorFunction.apply(grad_input, operator, num_extra_dims, is_2d)


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

    :param operator: a `ts.Operator`
    :param num_extra_dims: Number of extra dimensions to be prepended. For use
    with Pytorch neural networks, set this to 2 to add batch and channel
    dimensions.
    :param is_2d: if True, the first tomosipo dimension is not used, resulting
    in a 2D operator.
    :returns: an autograd function
    :rtype:

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
