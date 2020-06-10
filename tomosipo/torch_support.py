""" This module provides a short-hand to add torch support to tomosipo

To enable support for torch tensors in tomosipo, use:

>>> import tomosipo.torch_support

"""
import tomosipo.data_backends.torch
import tomosipo as ts
import torch
from torch.autograd import Function


class Forward(Function):
    @staticmethod
    def forward(ctx, input, vg, pg):
        # TODO: support supersampling
        if input.requires_grad:
            # Save geometries for backward pass
            ctx.vg = vg
            ctx.pg = pg

        output_type = torch.scalar_tensor(0.0, dtype=torch.float32, device=input.device)
        # Use temporary astra data link for both geometries:
        with ts.data(vg, input) as vd, ts.data(pg, output_type) as pd:
            ts.forward(vd, pd)
            return pd.data

    @staticmethod
    def backward(ctx, grad_output):
        vg = ctx.vg
        pg = ctx.pg

        input_type = torch.scalar_tensor(
            0.0,
            dtype=torch.float32,
            device=grad_output.device
        )

        with ts.data(vg, input_type) as vd, ts.data(pg, grad_output) as pd:
            ts.backward(vd, pd)
            # Do not return gradients for vg and pg
            return vd.data, None, None


forward = Forward.apply


class Backward(Function):
    @staticmethod
    def forward(ctx, input, vg, pg):
        # TODO: support supersampling
        if input.requires_grad:
            # Save geometries for backward pass
            ctx.vg = vg
            ctx.pg = pg

        output_type = torch.scalar_tensor(0.0, dtype=torch.float32, device=input.device)
        # Use temporary astra data link for both geometries:
        with ts.data(vg, output_type) as vd, ts.data(pg, input) as pd:
            ts.backward(vd, pd)
            return vd.data

    @staticmethod
    def backward(ctx, grad_output):
        vg = ctx.vg
        pg = ctx.pg

        input_type = torch.scalar_tensor(
            0.0,
            dtype=torch.float32,
            device=grad_output.device
        )

        with ts.data(vg, grad_output) as vd, ts.data(pg, input_type) as pd:
            ts.backward(vd, pd)
            # Do not return gradients for vg and pg
            return pd.data, None, None


backward = Backward.apply
