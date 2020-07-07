"""Distributed training with pytorch-lightning

For this demo, you need pytorch-lightning, which is a framework that
makes it easire to use complex training techniques like distributed
training and reduced precision.

To install, use:
```
conda install python=3.6 cudatoolkit=10.1 pytorch astra-toolbox tqdm matplotlib pytorch-lightning \
                -c pytorch -c defaults -c astra-toolbox/label/dev -c conda-forge
pip install git+https://github.com/ahendriksen/tomosipo.git@WIP-multi-gpu

```

To run, use:

> python learned_pd_lightning.py --gpus 4 --batch_size=2 --N 512

Note that the batch size is defined per gpu per node. So the above
code would have a batch size of 8. The code uses Torch distributed
data-parallel training, which starts a new process per GPU and can be
run on multiple nodes. For more information on distributed training,
see:

https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html
https://pytorch.org/docs/stable/distributed.html#distributed-launch

In an off-the-cuff benchmark, I have compared this lightning
implementation to the implementation in learned_pd_benchmark.py.

I got the following results:

| Backend |   N | B | # GPUs | iterations / sec | iterations / epoch | time (epoch) |
|---------+-----+---+--------+------------------+--------------------+--------------|
| DDP*    | 512 | 1 |      4 |              2.4 |                128 |    53.333333 |
| DDP*    | 512 | 2 |      4 |             1.19 |                 64 |    53.781513 |
| DDP*    | 512 | 1 |      1 |             2.89 |                512 |    177.16263 |
|---------+-----+---+--------+------------------+--------------------+--------------|
| DP*     | 512 | 1 |      1 |                3 |                512 |    170.66667 |
| DP*     | 512 | 2 |      1 |             1.51 |                256 |    169.53642 |
| DP*     | 512 | 4 |      4 |       0.69444444 |                128 |    184.32000 |
|---------+-----+---+--------+------------------+--------------------+--------------|

* DDP = Distributed data-parallel using pytorch-lightning (this file)
* DP  = Data-parallel using pytorch and tomosipo (learned_pd_benchmark)

As you can see, for single-gpu training, pytorch-lightning is slightly
slower. For multi-gpu training, it obtains almost a 3.3-fold speedup
on 4 GPUs.

I should note that I have only benchmarked this code. I have not
tested if the trained networks are any good.

"""

import torch
import pytorch_lightning as pl
import tomosipo as ts
from tomosipo.torch_support import (
    to_autograd,
)
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

pl.seed_everything(123)

# Copy definition of learned primal-dual from previous notebook
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

        ts_op = ts.operator(self.vg[:1], self.pg.to_vec()[:, :1, :])
        op = to_autograd(ts_op)
        opT = to_autograd(ts_op.T)
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


class LightningPD(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        N = hparams.N
        self.vg = ts.volume(size=1, center=0, shape=N)
        self.pg = ts.parallel(angles=3 * N // 2, shape=(N, 3 * N // 2), size=(1, 1.5))

        self.learned_pd = LearnedPD(self.vg, self.pg, hparams.n_iters, do_pingpong=hparams.pingpong)

    def forward(self, x):
        return self.learned_pd(x)

    def training_step(self, batch, batch_idx):

        inp, tgt = batch
        # Add channel dimension
        # TODO: remove
        inp, tgt = inp[:, None, ...], tgt[:, None, ...]
        out = self.learned_pd(inp)
        loss = F.mse_loss(out, tgt)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--N', default=128, type=int)
    parser.add_argument('--n_iters', default=10, type=int)
    parser.add_argument('--pingpong', default=False, type=bool)
    parser.add_argument('--batch_size', default=4, type=int)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # parse params
    args = parser.parse_args()

    # init module
    model = LightningPD(hparams=args)

    # Generate dataset
    phantom = ts.phantom.hollow_box(ts.data(model.vg)).data
    phantom = torch.from_numpy(phantom)
    op = ts.operator(model.vg, model.pg)
    sino = op(phantom)
    noisy_sino = sino + sino.max() / 20 * torch.randn_like(sino)

    train_ds = torch.utils.data.TensorDataset(noisy_sino, phantom)
    # For distributed data-parallel training, the effective batch size
    # will be args.batch_size * gpus * num_nodes.
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        num_workers=0,
        shuffle=True,
        batch_size=args.batch_size
    )

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args, distributed_backend="ddp")
    trainer.fit(model, train_dl)
