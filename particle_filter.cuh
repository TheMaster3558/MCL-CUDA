#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);         \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

constexpr int THREADS = 256;

__global__ void init(float2* particles, curandState* states, int seed, int n,
                     float2 mean, float stddev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);

        particles[i].x = mean.x + stddev * curand_normal(&state);
        particles[i].y = mean.y + stddev * curand_normal(&state);
        states[i] = state;
    }
}

template <class... Sensors>
__global__ void moveAndWeighKernel(float2* particles, float* weights,
                                   curandState* states, std::size_t n,
                                   float2 delta, float stddev,
                                   Sensors... sensors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    curandState state = states[i];
    float2 particle = particles[i];

    particle.x += delta.x + stddev * curand_normal(&state);
    particle.y += delta.y + stddev * curand_normal(&state);

    particles[i] = particle;
    states[i] = state;

    float weight = 1.0f;
    ((weight *= sensors.likelihood(particle)), ...);
    weights[i] = weight;
}

__global__ void weightedSumReductionKernel(const float2* particles,
                                           const float* weights,
                                           std::size_t n,
                                           double2* weightedSum) {
    __shared__ float2 blockParticleData[THREADS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < n) {
        float weight = static_cast<float>(weights[i]);
        blockParticleData[tid] = {
            particles[i].x * weight,
            particles[i].y * weight
        };
    }
    else {
        blockParticleData[tid] = {0.0, 0.0};
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            blockParticleData[tid].x += blockParticleData[tid + stride].x;
            blockParticleData[tid].y += blockParticleData[tid + stride].y;
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&weightedSum->x, blockParticleData[0].x);
        atomicAdd(&weightedSum->y, blockParticleData[0].y);
    }
}

template <class Input>
__global__ void prefixSumKernel(const Input* input, std::size_t n,
                                double* output, double* blockSums) {
    __shared__ double temp[THREADS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    temp[tid] = (i < n) ? static_cast<double>(input[i]) : 0.0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        double value = 0.0;
        if (tid >= offset) {
            value = temp[tid - offset];
        }

        __syncthreads();
        temp[tid] += value;
        __syncthreads();
    }

    if (i < n) {
        output[i] = temp[tid];
    }
    if (blockSums != nullptr && tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = temp[tid];
    }
}

__global__ void addBlockOffsetsKernel(std::size_t n, double* output,
                                      const double* blockPrefixSums) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }
    if (blockIdx.x > 0) {
        output[i] += blockPrefixSums[blockIdx.x - 1];
    }
}

inline std::size_t prefixSumScratchSize(std::size_t n) {
    if (n == 0) {
        return 0;
    }

    std::size_t total = 0;
    std::size_t blocks = (n - 1) / THREADS + 1;

    while (blocks > 1) {
        total += 2 * blocks;
        blocks = (blocks - 1) / THREADS + 1;
    }
    return total;
}

template <class Input>
void prefixSum(const Input* input, std::size_t n, double* output,
               double* scratch) {
    if (n == 0) {
        return;
    }

    std::size_t blocks = (n - 1) / THREADS + 1;
    if (blocks == 1) {
        prefixSumKernel<Input><<<1, THREADS>>>(input, n, output, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    double* blockSums = scratch;
    double* blockPrefixSums = scratch + blocks;

    prefixSumKernel<Input><<<blocks, THREADS>>>(input, n, output, blockSums);
    CUDA_CHECK(cudaGetLastError());

    prefixSum<double>(blockSums, blocks, blockPrefixSums,
                      scratch + 2 * blocks);

    addBlockOffsetsKernel<<<blocks, THREADS>>>(n, output, blockPrefixSums);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void resampleKernel(const float2* particles, float* weights,
                               const double* cumWeights, double u0,
                               std::size_t n, float2* resampledParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    double totalWeight = cumWeights[n - 1];
    double u = (u0 + static_cast<double>(i)) / static_cast<double>(n) *
        totalWeight;

    int lo = 0;
    int hi = static_cast<int>(n) - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cumWeights[mid] < u) {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }

    resampledParticles[i] = particles[lo];
    weights[i] = 1.0f / static_cast<float>(n);
}

class ParticleFilter {
public:
    explicit ParticleFilter(std::size_t n, float2 mean, float stddev,
                            std::uint64_t seed = 12345)
        : n(n), blocks((n + THREADS - 1) / THREADS) {
        CUDA_CHECK(cudaMalloc(&particles, n * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&resampledParticles, n * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&weightedSum, sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&weights, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cumWeights, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&prefixScratch,
                              prefixSumScratchSize(n) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&states, n * sizeof(curandState)));

        init<<<blocks, THREADS>>>(particles, states, static_cast<int>(seed),
                                  static_cast<int>(n), mean, stddev);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~ParticleFilter() {
        release();
    }

    ParticleFilter(const ParticleFilter&) = delete;
    ParticleFilter& operator=(const ParticleFilter&) = delete;

    template <class... Sensors>
    void iterate(float2 delta, const Sensors&... sensors) {
        const float stddev = 0.01f + 0.05f * std::hypot(delta.x, delta.y);

        moveAndWeighKernel<<<blocks, THREADS>>>(
            particles, weights, states, n, delta, stddev, sensors...);
        CUDA_CHECK(cudaGetLastError());
        estimateAndResample();
    }

    float2 getPrediction() const {
        return prediction;
    }

private:
    void estimateAndResample() {
        CUDA_CHECK(cudaMemset(weightedSum, 0, sizeof(double2)));

        weightedSumReductionKernel<<<blocks, THREADS>>>(
            particles, weights, n, weightedSum);
        CUDA_CHECK(cudaGetLastError());

        double2 hostWeightedSum;
        double hostTotalWeight;
        CUDA_CHECK(cudaMemcpy(&hostWeightedSum, weightedSum, sizeof(double2),
            cudaMemcpyDeviceToHost));

        prefixSum(weights, n, cumWeights, prefixScratch);
        CUDA_CHECK(cudaMemcpy(&hostTotalWeight, cumWeights + n - 1,
            sizeof(double), cudaMemcpyDeviceToHost));

        prediction = {
            static_cast<float>(hostWeightedSum.x / hostTotalWeight),
            static_cast<float>(hostWeightedSum.y / hostTotalWeight)
        };

        double u0 = uniform_dist(host_rng);
        resampleKernel<<<blocks, THREADS>>>(particles, weights, cumWeights, u0,
                                            n, resampledParticles);
        CUDA_CHECK(cudaGetLastError());
        std::swap(particles, resampledParticles);
    }

    void release() {
        if (particles) {
            CUDA_CHECK(cudaFree(particles));
            particles = nullptr;
        }
        if (resampledParticles) {
            CUDA_CHECK(cudaFree(resampledParticles));
            resampledParticles = nullptr;
        }
        if (weightedSum) {
            CUDA_CHECK(cudaFree(weightedSum));
            weightedSum = nullptr;
        }
        if (weights) {
            CUDA_CHECK(cudaFree(weights));
            weights = nullptr;
        }
        if (cumWeights) {
            CUDA_CHECK(cudaFree(cumWeights));
            cumWeights = nullptr;
        }
        if (prefixScratch) {
            CUDA_CHECK(cudaFree(prefixScratch));
            prefixScratch = nullptr;
        }
        if (states) {
            CUDA_CHECK(cudaFree(states));
            states = nullptr;
        }
    }

    std::size_t n;
    int blocks;

    float2* particles = nullptr;
    float2* resampledParticles = nullptr;
    double2* weightedSum = nullptr;
    float* weights = nullptr;
    double* cumWeights = nullptr;
    double* prefixScratch = nullptr;
    curandState* states = nullptr;

    float2 prediction{};
    std::mt19937_64 host_rng;
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
};

