"""Distributed training with pytorch-lightning

For this demo, you need pytorch-lightning, which is a framework that
makes it easire to use complex training techniques like distributed
training and reduced precision.

To install, use:
```
conda install python=3.6 cudatoolkit=10.1 pytorch astra-toolbox tqdm matplotlib pytorch-lightning \
                -c pytorch -c defaults -c astra-toolbox/label/dev -c conda-forge
pip install git+https://github.com/ahendriksen/tomosipo.git

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
from learned_pd import LearnedPD

pl.seed_everything(123)

class LightningPD(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        N = hparams.N
        self.vg = ts.volume(size=1, shape=N)
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
