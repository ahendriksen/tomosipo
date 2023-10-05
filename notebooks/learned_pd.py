import torch 
import tomosipo as ts
from tomosipo.torch_support import (
    to_autograd,
)
import torch.nn as nn


# The network defined below should be almost the same as the primal-dual network described in:

# Adler, J., & Öktem, Ozan, Learned Primal-Dual Reconstruction, IEEE Transactions on Medical Imaging, (), 1–1 (2018)
# http://dx.doi.org/10.1109/tmi.2018.2799231

# Intended differences are the use of ReLU instead of PReLU, and the lack of biases. 

# All other differences are unintentional.. 

class PrimalModule(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=3, padding=1, bias=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DualModule(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(7, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=3, padding=1, bias=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class LearnedPD(nn.Module):
    def __init__(self, vg, pg, n_iters, do_pingpong=False):
        super().__init__()
        self.n_iters = n_iters
        self.vg = vg
        self.pg = pg
        self.do_pingpong = do_pingpong
        self.ts_op = ts.operator(self.vg[:1], self.pg.to_vec()[:, :1, :])
        self.op = to_autograd(self.ts_op, is_2d=True, num_extra_dims=2)
        self.opT = to_autograd(self.ts_op.T, is_2d=True, num_extra_dims=2)
        
        for i in range(n_iters):
            # To ensure that the parameters of the primal and dual modules
            # are correctly distributed during parallel training, we register
            # them as modules.
            self.add_module(f"{i}_primal", PrimalModule())
            self.add_module(f"{i}_dual", DualModule())

    def forward(self, g):
        B, C, H, W = g.shape
        assert C == 1, "single channel support only for now"
        h = g.new_zeros(B, 5, H, W)
        f_primal = g.new_zeros(B, 5, *self.vg.shape[1:])

        def dual_step(g, h, f, module):
            x = torch.cat((h, f, g), dim=1)
            out = module(x)
            return h + out
        def primal_step(f, update, module):
            x = torch.cat((f, update), dim=1)
            out = module(x)
            return f + out

        def fp(x):
            if self.do_pingpong:
                x = x.cpu()
            return op(x).to(x.device)
        def bp(x):
            if self.do_pingpong:
                x = x.cpu()
            return opT(x).to(x.device)

        for i in range(self.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")

            f_dual = fp(f_primal[:, :1])
            h = dual_step(g, h, f_dual, dual_module)
            update = bp(h[:, :1])
            f_primal = primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
