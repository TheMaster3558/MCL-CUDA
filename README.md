# Monte Carlo Localization (MCL) with CUDA

## Overview

This project implements a CUDA particle filter for estimating a moving target's position from observations. It propagates candidate positions, scores them against each observation, computes a weighted estimate, and resamples the population for the next update.

The benchmark measures particle counts from 1,024 to 268,435,456. The sections below describe the filter's CUDA execution and the measured performance.

## GPU benchmark

The recorded run reaches **6,668.69 million particle updates per second** at 4,194,304 particles on an NVIDIA GeForce RTX 5090. Throughput stays near 6,350–6,670 million updates per second from 1,048,576 particles onward.

<p align="center">
  <img src="assets/readme/throughput.svg" width="100%" alt="Bar chart of measured GPU particle-update throughput for 13 particle counts up to 268,435,456, peaking at 6,668.69 million updates per second at 4,194,304 particles">
</p>

| Particles | GPU time (ms/update) | Throughput (M updates/s) | RMSE |
|---:|---:|---:|---:|
| 1,024 | 0.0351 | 29.19 | 0.0644 |
| 4,096 | 0.0348 | 117.73 | 0.0648 |
| 16,384 | 0.0339 | 482.88 | 0.0641 |
| 65,536 | 0.0412 | 1589.51 | 0.0638 |
| 262,144 | 0.0720 | 3639.40 | 0.0638 |
| 1,048,576 | 0.1603 | 6539.97 | 0.0638 |
| 4,194,304 | 0.6290 | 6668.69 | 0.0738 |
| 8,388,608 | 1.2609 | 6652.94 | 0.0738 |
| 16,777,216 | 2.6105 | 6426.89 | 0.0739 |
| 33,554,432 | 5.2019 | 6450.39 | 0.0943 |
| 67,108,864 | 10.4365 | 6430.22 | 0.0944 |
| 134,217,728 | 20.9798 | 6397.49 | 0.0944 |
| 268,435,456 | 42.2880 | 6347.79 | 0.0944 |

The GPU results were recorded on an RTX 5090 with 170 SMs, 33.8 GB of memory, and compute capability 12.0. RMSE is the root mean squared Euclidean position error over timed updates. Full output, including CPU timing reference results, is in [`resultsdesktop5090.txt`](resultsdesktop5090.txt); the CPU implementation is used only for benchmarking.

## Build status

The source requires the CUDA runtime, cuRAND headers, a CUDA-capable C++ compiler, and an NVIDIA GPU. The repository does not contain a checked-in build configuration, so it currently has no reproducible build command.

## Source files

- [`main.cu`](main.cu) — program entry point, simulation setup, and result reporting.
- [`particle_filter.cuh`](particle_filter.cuh) — CUDA kernels and particle-filter orchestration.
- [`resultsdesktop5090.txt`](resultsdesktop5090.txt) — recorded benchmark output.

## Technical details

### Update algorithm

<p align="center">
  <img src="assets/readme/filter-cycle.svg" width="100%" alt="Particle update cycle: motion and range observations feed particle propagation and scoring, then branch to a weighted estimate and to systematic resampling for the next particle set">
</p>

The scan and weighted-mean reduction both consume the freshly computed weights. The mean uses the moved particle positions. The scan produces an inclusive cumulative distribution used to select the next population. Resampled particles go into a separate buffer, so reads from the current population do not overlap writes to the next one.

### CUDA execution and data flow

Each launch uses 256 threads per block. The grid has `ceil(N / 256)` blocks, and the final block masks particle indices beyond `N`. The filter keeps these device-side arrays:

- Two `float2` particle buffers: current positions and resampling output.
- One float weight per particle and one double cumulative weight per particle.
- One cuRAND state per particle.
- A `double2` weighted-position accumulator and recursively sized scan scratch storage.

`moveAndWeighKernel` assigns one thread to each particle. It loads that particle's position and RNG state, applies the target displacement and process noise, stores the updated position and RNG state, evaluates the sensor likelihoods, and writes a fresh weight.

#### Weighted-position reduction

`weightedSumReductionKernel` forms `(x_i' * weight_i, y_i' * weight_i)` for each particle. Within each block, threads reduce those pairs through shared memory using a binary tree: strides decrease from 128 to 1, with a block barrier after each level. Out-of-range threads contribute zero. Thread zero atomically adds each block's partial result to a cleared global `double2` accumulator; the shared-memory block partials themselves are `float2`.

#### Hierarchical inclusive scan

`prefixSum` scans the float weights into double cumulative weights. For each group of up to 256 values, `prefixSumKernel` uses double shared memory and doubling offsets (`1, 2, 4, ... 128`) to produce an inclusive block scan. The last lane writes the block total. If multiple blocks were used, the routine recursively scans those totals, then `addBlockOffsetsKernel` adds each block's preceding scanned total to its local outputs. A one-block input skips the recursive and offset-add stages. `prefixSumScratchSize` reserves the scratch arrays for the full recursion depth.

#### Systematic resampling

After the scan, the host reads the final cumulative weight, `total_weight = cumulative[N - 1]`. It draws one uniform offset `u0`; for output index `k`, the resampling target is:

```math
t_k = \frac{u_0 + k}{N}\,\text{total\_weight}
```

`resampleKernel` performs a lower-bound binary search for the first cumulative weight greater than or equal to `target_k`. The output thread copies that source particle to the alternate buffer and sets its output weight to `1/N`. Once the kernel is queued, the two particle-buffer pointers are swapped for the next call to `iterate`.

#### Host and device boundary

The weighted-position reduction runs first. A device-to-host copy returns its `double2` sum, then the inclusive scan runs and a second copy returns its final double total. Those values are divided on the host to update `prediction`. The host also generates the resampling offset; the selection search and particle copies remain on the GPU. The benchmark synchronizes around each timed update, so its reported GPU time includes kernel work and these host-device transfers.

The execution path is summarized below. The reduction and scan process the same per-particle weights for different purposes: one produces the estimate, the other supplies the cumulative distribution for resampling.

<p align="center">
  <img src="assets/readme/pipeline.svg" width="100%" alt="CUDA execution flow showing particle propagation and scoring, weighted reduction, prefix scan, host aggregation, systematic resampling, and particle-buffer swap">
</p>
