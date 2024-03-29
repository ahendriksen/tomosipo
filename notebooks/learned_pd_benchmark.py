"""Benchmark the speed of the learned primal-dual with various parameters

This demo requires the following packages:

```
conda install pytorch cudatoolkit=X.X  astra-toolbox matplotlib tqdm -c astra-toolbox/label/dev -c pytorch
pip install git+https://github.com/ahendriksen/tomosipo.git
```

To test the effect of moving data to CPU before applying ASTRA
operations and moving the resulting data to GPU again, execute:

> python learned_pd_benchmark.py --N 128 --pingpong
> python learned_pd_benchmark.py --N 128 --no-pingpong

On a dual-socket system with a Titan RTX 2080 Ti GPU, the following
timings (in seconds) were obtained:

|   N | Time (pingpong) | Time (no pingpong) |   Speedup |
|-----+-----------------+--------------------+-----------|
| 128 |           1.476 |              0.910 | 1.6219780 |
| 256 |           2.353 |              1.124 | 2.0934164 |
| 512 |           5.236 |              3.202 | 1.6352280 |


Additionally, the size of the volume and projection data may be
changed by altering the `--N' parameter, and the number of iterations
may be changed by setting `--n_iters' (default 10).

By default, training is benchmarked. The speed of inference can be
tested as follows:

> python learned_pd_benchmark.py --N 128 --train
> python learned_pd_benchmark.py --N 128 --inference

By default, 10 training steps or 10 inference steps are taken per
trial. The reported statistics are the mean over 5 trials.

We find that using torch.nn.DataParallel does not improve performance,
as this benchmark bears out:

> python learned_pd_benchmark.py --N 128 --batch_size 8 --num_trials 5 --n_iter=10
: Time (seconds): 4.269+-0.024 in range (4.243 -- 4.299)

> python learned_pd_benchmark.py --N 128 --batch_size 8 --num_trials 5 --n_iter=10 --parallel_gpu
: Time (seconds): 10.603+-0.165 in range (10.276 -- 10.719)


To use multiple GPUs, check out learned_pd_lightning.py!

"""

import torch
import tomosipo as ts
from tomosipo.torch_support import (
    to_autograd,
)
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from learned_pd import LearnedPD


# Timing functions
def time_function(f):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def benchmark(f, num_trials):
    timings = np.zeros(num_trials)

    for i in tqdm(range(num_trials)):
        timings[i] = time_function(f)

    return (
        timings.mean(),
        timings.std(),
        timings.min(),
        timings.max()
    )


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--N', default=128, type=int)
    parser.add_argument('--n_iters', default=10, type=int)
    parser.add_argument('--pingpong', dest='pingpong', action='store_true', default=False)
    parser.add_argument('--no-pingpong', dest='pingpong', action='store_false')

    parser.add_argument('--parallel_gpu', dest="parallel_gpu", action='store_true', default=False)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_burnin', default=3, type=int)
    parser.add_argument('--num_trials', default=5, type=int)
    parser.add_argument('--num_iterations_per_trial', default=10, type=int)

    parser.add_argument('--train', dest='inference_only', action='store_false', default=False)
    parser.add_argument('--inference', dest='inference_only', action='store_true')

    # parse params
    args = parser.parse_args()

    print("Parameters")
    for k, v in args._get_kwargs():
        print(f"{k:<30} {v}")
    print()

    # Generate data:
    N = args.N
    full_vg = ts.volume(shape=N, size=1)
    full_pg = ts.parallel(angles=3 * N // 2, shape=(N, 3 * N // 2), size=(1, 1.5))
    full_A = ts.operator(full_vg, full_pg)

    phantom = ts.phantom.hollow_box(ts.data(full_vg)).data
    phantom = torch.from_numpy(phantom)
    sino = full_A(phantom)
    noisy_sino = sino + sino.max() / 20 * torch.randn_like(sino)

    # Datasets
    train_ds = torch.utils.data.TensorDataset(noisy_sino, phantom)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    # In practice, parallel_gpu appears to be slower..
    net = LearnedPD(full_vg, full_pg, args.n_iters, do_pingpong=args.pingpong).cuda()
    if args.parallel_gpu:
        net = nn.DataParallel(net).cuda()

    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MSELoss()

    def inference():
        for i, batch in zip(range(args.num_iterations_per_trial), train_dl):
            inp, tgt = batch
            # Move to gpu and add channel dimension
            inp, tgt = inp.cuda()[:, None, ...], tgt.cuda()[:, None, ...]
            # Include time to move data back to cpu
            out = net(inp).cpu()

    def train():
        for i, batch in zip(range(args.num_iterations_per_trial), train_dl):
            inp, tgt = batch
            # Move to gpu and add channel dimension
            inp, tgt = inp.cuda()[:, None, ...], tgt.cuda()[:, None, ...]
            out = net(inp)
            loss = criterion(out, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    f = inference if args.inference_only else train

    print("Burning in.. ")
    for _ in tqdm(range(args.num_burnin)):
        f()

    print("Benchmark.. ")
    mean, std, min, max = benchmark(f, args.num_trials)

    print(f"Time (seconds): {mean:0.3f}+-{std:0.3f} in range ({min:0.3f} -- {max:0.3f})")
