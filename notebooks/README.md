# ASTRA + Pytorch integration

This directory contains notebooks and scripts to illustrate how to use
tomosipo to integrate ASTRA and Pytorch.  It contains an
implementation of the simultaneous iterative reconstruction technique
(SIRT) and of learned primal-dual reconstruction (Adler & Öktem,
Learned Primal-Dual Reconstruction, IEEE TMI, (2018)).

## Requirements

The preferred way to obtain dependencies is using Conda. Run:

``` bash
# Create environment 'tomosipo-demo'
conda create -n tomosipo-demo \
	python=3.6 cudatoolkit=10.1 pytorch astra-toolbox tqdm matplotlib pytorch-lightning \
	-c pytorch -c defaults -c astra-toolbox/label/dev -c conda-forge
# Activate environment
conda activate tomosipo-demo
# Install latest tomosipo dev-branch:
pip install git+https://github.com/ahendriksen/tomosipo.git@WIP-multi-gpu

```

## SIRT

We describe how to implement SIRT in `sirt.ipynb`. This describes how
to set up a simple geometry in tomosipo and how to call ASTRA's
forward and backward projection operator. Specifically, we compare to
ASTRA's internal `SIRT3D_CUDA` algorithm and find that we get almost
the same results (up to 1e-8).

The update step of the SIRT algorithm can be implemented in a single
line without impeeding performance:
``` python
for i in range(num_iters):
    x += C * A.T(R * (y - A(x)))
```

To compare speed, we have included a benchmark script `sirt_benchmark.py`.
Here, we find that tomosipo can be faster than the `SIRT3D_CUDA` algorithm:
``` bash
$ python sirt_benchmark.py --N 256 --num_burnin 1 --num_trials 5 --n_iter=200 --tomosipo
[.. snip ..]
Benchmark..
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:29<00:00, 17.92s/it]
Time (seconds): 17.914+-0.076 in range (17.780 -- 17.984)

$ python sirt_benchmark.py --N 256 --num_burnin 1 --num_trials 5 --n_iter=200 --astra
[.. snip ..]
Benchmark..
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:36<00:00, 19.29s/it]
Time (seconds): 19.288+-0.051 in range (19.195 -- 19.339)

```

## Learned primal-dual

A visual introduction to learned-primal dual is given in
`learned_pd.ipynb`. We obtain nice results on a toy problem by testing
on the training set.


### Removing the pingpong

The benchmark script `learned_pd_benchmark.py` shows that keeping the
data on the GPU during training improves performance. Before the
direct integration with ASTRA and pytorch, it was more difficult to
directly operate on GPU arrays using ASTRA. Therefore, a "pingpong"
scheme was used. Before any ASTRA operation,

1. the data was moved from GPU to CPU,
2. ASTRA moved it to GPU,
3. ASTRA performed a forward/back-projection,
4. ASTRA moved the result to CPU
5. this result was moved back to GPU again.

Now we know that removing this pingpong scheme enables a speedup of
1.6x - 2x. We have the following results:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<colgroup>
<col  class="org-right" />
<col  class="org-right" />
<col  class="org-right" />
<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">N</th>
<th scope="col" class="org-right">Time (pingpong)</th>
<th scope="col" class="org-right">Time (no pingpong)</th>
<th scope="col" class="org-right">Speedup</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-right">128</td>
<td class="org-right">1.476</td>
<td class="org-right">0.910</td>
<td class="org-right">1.6219780</td>
</tr>
<tr>
<td class="org-right">256</td>
<td class="org-right">2.353</td>
<td class="org-right">1.124</td>
<td class="org-right">2.0934164</td>
</tr>
<tr>
<td class="org-right">512</td>
<td class="org-right">5.236</td>
<td class="org-right">3.202</td>
<td class="org-right">1.6352280</td>
</tr>
</tbody>
</table>

This shows that the additional integration with GPU arrays was worth the effort!

### Multi-GPU training

In addition, we describe how to obtain a speedup from multi-gpu
training. This is not straightforward. Using the single-process,
multi-gpu approach of `torch.nn.DataParallel`, training actually
becomes slower! Instead, we use the
[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
framework to obtain a distributed data-parallel training strategy for
learned primal-dual.

Comparing the two implementations, we find the following performance figures:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Backend</th>
<th scope="col" class="org-right">N</th>
<th scope="col" class="org-right">B</th>
<th scope="col" class="org-right"># GPUs</th>
<th scope="col" class="org-right">iterations / sec</th>
<th scope="col" class="org-right">iterations / epoch</th>
<th scope="col" class="org-right">time (epoch)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">DDP*</td>
<td class="org-right">512</td>
<td class="org-right">1</td>
<td class="org-right">4</td>
<td class="org-right">2.4</td>
<td class="org-right">128</td>
<td class="org-right">53.333333</td>
</tr>


<tr>
<td class="org-left">DDP*</td>
<td class="org-right">512</td>
<td class="org-right">2</td>
<td class="org-right">4</td>
<td class="org-right">1.19</td>
<td class="org-right">64</td>
<td class="org-right">53.781513</td>
</tr>


<tr>
<td class="org-left">DDP*</td>
<td class="org-right">512</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">2.89</td>
<td class="org-right">512</td>
<td class="org-right">177.16263</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">DP*</td>
<td class="org-right">512</td>
<td class="org-right">1</td>
<td class="org-right">1</td>
<td class="org-right">3</td>
<td class="org-right">512</td>
<td class="org-right">170.66667</td>
</tr>


<tr>
<td class="org-left">DP*</td>
<td class="org-right">512</td>
<td class="org-right">2</td>
<td class="org-right">1</td>
<td class="org-right">1.51</td>
<td class="org-right">256</td>
<td class="org-right">169.53642</td>
</tr>


<tr>
<td class="org-left">DP*</td>
<td class="org-right">512</td>
<td class="org-right">4</td>
<td class="org-right">4</td>
<td class="org-right">0.69444444</td>
<td class="org-right">128</td>
<td class="org-right">184.32000</td>
</tr>
</tbody>
</table>


* DDP = Distributed data-parallel using pytorch-lightning (`learned_pd_lightning.py`)
* DP  = Data-parallel using vanilla pytorch (`learned_pd_benchmark.py`)

In this benchmark, for single-gpu training, pytorch-lightning is
slightly slower than a vanilla pytorch implementation (177 versus 170
seconds per epoch). For multi-gpu training, pytorch-lightning obtains
almost a 3.3-fold speedup on 4 GPUs (53 versus 177 seconds per epoch).
