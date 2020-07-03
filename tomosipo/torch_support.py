""" This module provides a short-hand to add torch support to tomosipo

To enable support for torch tensors in tomosipo, use:

>>> import tomosipo.torch_support

"""
import tomosipo as ts
import torch
from torch.autograd import Function

# This import is needed to enable to pytorch linking backend.
import tomosipo.links.torch


class OperatorFunction(Function):
    @staticmethod
    def forward(ctx, input, operator):
        if input.requires_grad:
            ctx.operator = operator

        return operator(input)

    @staticmethod
    def backward(ctx, grad_output):
        operator = ctx.operator

        # do not return gradient for operator
        return operator.T(grad_output), None


def to_autograd(operator):
    """Converts an operator to an autograd function

    Example:

    >>> A = ts.operator(vg, pg)
    >>> f = to_autograd(A)
    >>> vd = torch.randn(ts.links.geometry_shape(vg), requires_grad=True)
    >>> f(vd).sum().backward()
    >>> print(vd.grad)

    Likewise, you may use the transpose:
    >>> g = to_autograd(A.T)
    >>> pd = torch.randn(ts.links.geometry_shape(pg), requires_grad=True)
    >>> g(pd).sum().backward()
    >>> print(vd.grad)

    :param operator: a `ts.Operator'
    :returns: an autograd function
    :rtype:

    """
    def f(x):
        return OperatorFunction.apply(x, operator)

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
            0.0,
            dtype=torch.float32,
            device=grad_output.device
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
            0.0,
            dtype=torch.float32,
            device=grad_output.device
        )

        with ts.data(vg, grad_output) as vd, ts.data(pg, input_type) as pd:
            ts.backward(vd, pd, projector=projector)
            # Do not return gradients for vg, pg, and projector
            return pd.data, None, None, None


def backward(input, vg, pg, projector=None):
    return Backward.apply(input, vg, pg, projector)
