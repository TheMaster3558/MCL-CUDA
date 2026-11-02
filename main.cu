#include "particle_filter.cuh"
#include "cpu_particle_filter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

struct RangeSensor {
    float2 beacon;
    float measured;
    float sigma;

    __host__ __device__ float likelihood(float2 particle) const {
        float dx = particle.x - beacon.x;
        float dy = particle.y - beacon.y;
        float distance = sqrtf(dx * dx + dy * dy);
        float z = (distance - measured) / sigma;
        return expf(-0.5f * z * z);
    }
};

static float rangeTo(float2 a, float2 b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

struct BenchResult {
    double msPerIter;
    double rmse;
};

static std::string getCpuName() {
    char brand[49] = {};

#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 0x80000000);
    if (static_cast<unsigned int>(registers[0]) >= 0x80000004) {
        for (int leaf = 0; leaf < 3; ++leaf) {
            __cpuid(registers, 0x80000002 + leaf);
            std::memcpy(brand + leaf * 16, registers, 16);
        }
    }
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000004) {
        for (unsigned int leaf = 0; leaf < 3; ++leaf) {
            __get_cpuid(0x80000002 + leaf, &eax, &ebx, &ecx, &edx);
            unsigned int registers[] = {eax, ebx, ecx, edx};
            std::memcpy(brand + leaf * 16, registers, 16);
        }
    }
#endif

    std::string name(brand);
    const auto first = name.find_first_not_of(' ');
    const auto last = name.find_last_not_of(' ');
    if (first == std::string::npos) {
        return "Unknown CPU";
    }
    return name.substr(first, last - first + 1);
}

static BenchResult runBenchmark(std::size_t nParticles, int warmupIters,
                                int timedIters) {
    const float2 beacons[3] = {{0.f, 0.f}, {10.f, 0.f}, {5.f, 10.f}};
    const float sensorSigma = 0.2f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.f, sensorSigma);

    const float radius = 3.f;
    const float omega = 0.05f;
    const float2 center = {5.f, 4.f};
    auto truthAt = [&](int step) {
        return float2{center.x + radius * std::cos(omega * step),
                      center.y + radius * std::sin(omega * step)};
    };

    ParticleFilter particleFilter(nParticles, truthAt(0), 1.0f);
    float2 previousTruth = truthAt(0);
    double squaredError = 0.0;
    double totalMs = 0.0;
    int errorCount = 0;

    const int totalIters = warmupIters + timedIters;
    for (int step = 1; step <= totalIters; ++step) {
        float2 truth = truthAt(step);
        float2 delta = {truth.x - previousTruth.x,
                        truth.y - previousTruth.y};
        previousTruth = truth;

        RangeSensor s0{beacons[0], rangeTo(truth, beacons[0]) + noise(rng),
                       sensorSigma};
        RangeSensor s1{beacons[1], rangeTo(truth, beacons[1]) + noise(rng),
                       sensorSigma};
        RangeSensor s2{beacons[2], rangeTo(truth, beacons[2]) + noise(rng),
                       sensorSigma};

        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();

        particleFilter.iterate(delta, s0, s1, s2);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();

        if (step > warmupIters) {
            totalMs +=
                std::chrono::duration<double, std::milli>(end - start).count();

            float2 estimate = particleFilter.getPrediction();
            double errorX = estimate.x - truth.x;
            double errorY = estimate.y - truth.y;
            squaredError += errorX * errorX + errorY * errorY;
            ++errorCount;
        }
    }

    return {totalMs / timedIters, std::sqrt(squaredError / errorCount)};
}

static BenchResult runCpuBenchmark(std::size_t nParticles, int warmupIters,
                                   int timedIters) {
    const float2 beacons[3] = {{0.f, 0.f}, {10.f, 0.f}, {5.f, 10.f}};
    const float sensorSigma = 0.2f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.f, sensorSigma);

    const float radius = 3.f;
    const float omega = 0.05f;
    const float2 center = {5.f, 4.f};
    auto truthAt = [&](int step) {
        return float2{center.x + radius * std::cos(omega * step),
                      center.y + radius * std::sin(omega * step)};
    };

    float2 initial = truthAt(0);
    CpuParticleFilter particleFilter(
        nParticles, CpuPoint{initial.x, initial.y}, 1.0f);
    float2 previousTruth = initial;
    double squaredError = 0.0;
    double totalMs = 0.0;
    int errorCount = 0;

    const int totalIters = warmupIters + timedIters;
    for (int step = 1; step <= totalIters; ++step) {
        float2 truth = truthAt(step);
        float2 delta = {truth.x - previousTruth.x,
                        truth.y - previousTruth.y};
        previousTruth = truth;

        CpuRangeSensor sensors[] = {
            {{beacons[0].x, beacons[0].y},
             rangeTo(truth, beacons[0]) + noise(rng), sensorSigma},
            {{beacons[1].x, beacons[1].y},
             rangeTo(truth, beacons[1]) + noise(rng), sensorSigma},
            {{beacons[2].x, beacons[2].y},
             rangeTo(truth, beacons[2]) + noise(rng), sensorSigma}};

        auto start = std::chrono::steady_clock::now();
        particleFilter.iterate(CpuPoint{delta.x, delta.y}, sensors, 3);
        auto end = std::chrono::steady_clock::now();

        if (step > warmupIters) {
            totalMs +=
                std::chrono::duration<double, std::milli>(end - start).count();

            CpuPoint estimate = particleFilter.getPrediction();
            double errorX = estimate.x - truth.x;
            double errorY = estimate.y - truth.y;
            squaredError += errorX * errorX + errorY * errorY;
            ++errorCount;
        }
    }

    return {totalMs / timedIters, std::sqrt(squaredError / errorCount)};
}

int main() {
    const std::string cpuName = getCpuName();
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("CPU: %s\n", cpuName.c_str());
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        printf("GPU %d of %d: %s (SM %d.%d, %d SMs, %.1f GB)\n", device,
               deviceCount, properties.name, properties.major,
               properties.minor, properties.multiProcessorCount,
               properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    printf("%10s %-18s %12s %12s %12s\n", "particles", "device",
           "ms/iter", "M particles/s", "RMSE");

    const std::size_t particleCounts[] = {
        1u << 10, 1u << 12, 1u << 14, 1u << 16, 1u << 18, 1u << 20,
        1u << 22, 1u << 23, 1u << 24, 1u << 25, 1u << 26};
    const std::size_t bytesPerParticle = 80;
    const std::size_t maxCpuParticles = 1u << 20;
    for (std::size_t n : particleCounts) {
        int warmupIters = 20;
        int timedIters = 200;
        if (n >= (1u << 22)) {
            warmupIters = 5;
            timedIters = 50;
        }
        if (n >= (1u << 25)) {
            warmupIters = 2;
            timedIters = 20;
        }

        for (int device = 0; device < deviceCount; ++device) {
            CUDA_CHECK(cudaSetDevice(device));

            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
            if (n * bytesPerParticle > freeBytes) {
                printf("%10zu GPU %d skipped (needs ~%.1f GB, %.1f GB free)\n",
                       n, device, n * bytesPerParticle / 1e9,
                       freeBytes / 1e9);
                continue;
            }

            BenchResult result = runBenchmark(n, warmupIters, timedIters);
            double throughput = (n / 1e6) / (result.msPerIter / 1e3);
            printf("%10zu GPU %d %12.4f %12.2f %12.4f\n", n, device,
                   result.msPerIter, throughput, result.rmse);
        }

        // Run the CPU benchmark after all GPUs, on the same thread.
        if (n <= maxCpuParticles) {
            constexpr int cpuWarmupIters = 3;
            constexpr int cpuTimedIters = 30;
            BenchResult result = runCpuBenchmark(
                n, cpuWarmupIters, cpuTimedIters);
            double throughput = (n / 1e6) / (result.msPerIter / 1e3);
            printf("%10zu %-18s %12.4f %12.2f %12.4f\n", n, "CPU",
                   result.msPerIter, throughput, result.rmse);
        } else {
            printf("%10zu %-18s %12s %12s %12s\n", n, "CPU", "skipped",
                   "skipped", "skipped");
        }
    }

    return 0;
}
