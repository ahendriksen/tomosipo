import pytest
import tomosipo as ts
from . import skip_if_no_cuda
from tomosipo.torch_support import (
    to_autograd,
)

try:
    import torch

    torch_present = True


except ModuleNotFoundError:
    torch_present = False

skip_if_no_torch = pytest.mark.skipif(not torch_present, reason="Pytorch not installed")

if torch_present:
    devices = [torch.device("cpu"), torch.device("cuda")]
else:
    devices = []

N = 64
N_angles = N * 3 // 2
M = N * 3 // 2


@skip_if_no_torch
def test_link_cpu():
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=(N, M))
    A = ts.operator(vg, pg)

    vol = torch.zeros(A.domain_shape)
    link = ts.link(vg, vol)

    assert torch.allclose(vol, link.new_zeros(A.domain_shape).data)
    assert torch.allclose(vol + 1.0, link.new_full(A.domain_shape, 1.0).data)
    assert link.shape == link.new_empty(A.domain_shape).shape


@skip_if_no_torch
@skip_if_no_cuda
@pytest.mark.parametrize("device", devices)
def test_link_gpu(device):
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=(N, M))
    A = ts.operator(vg, pg)

    vol = torch.zeros(A.domain_shape).to(device)
    link = ts.link(vg, vol)

    # Check that new data is created on the same device
    assert vol.device == link.new_zeros(A.domain_shape).data.device
    assert vol.device == link.new_full(A.domain_shape, 1.0).data.device
    assert link.data.device == link.new_empty(A.domain_shape).data.device


@skip_if_no_torch
@skip_if_no_cuda
@pytest.mark.parametrize("device", devices)
def test_fp_bp(device):
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=(N, M))
    A = ts.operator(vg, pg)
    vol = torch.ones(A.domain_shape).to(device)
    sino = A(vol)
    bp = A.T(sino)

    assert 1.0 < sino.sum()
    assert 1.0 < bp.sum()
    
@skip_if_no_torch
def test_float64():
    vg = ts.volume(shape=(1, N, N))
    pg = ts.parallel(angles=N_angles, shape=(1, M))
    A = ts.operator(vg, pg)
    A_ag = to_autograd(A)
    
    x = torch.ones((1, 1, N, N), dtype=torch.float64)
    y = A(x[0, ...])
    y_ag = A_ag(x)

    assert torch.equal(y, y_ag[0, ...])
