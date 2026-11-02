#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

struct CpuPoint {
    float x;
    float y;
};

struct CpuRangeSensor {
    CpuPoint beacon;
    float measured;
    float sigma;
};

class CpuParticleFilter {
public:
    explicit CpuParticleFilter(std::size_t n, CpuPoint mean, float stddev,
                               std::uint64_t seed = 12345)
        : particles(n), resampledParticles(n), weights(n),
          cumulativeWeights(n), rng(seed) {
        for (CpuPoint& particle : particles) {
            particle.x = mean.x + stddev * normal(rng);
            particle.y = mean.y + stddev * normal(rng);
        }
    }

    void iterate(CpuPoint delta, const CpuRangeSensor* sensors,
                 std::size_t sensorCount) {
        const float motionStddev =
            0.01f + 0.05f * std::hypot(delta.x, delta.y);
        double weightedX = 0.0;
        double weightedY = 0.0;
        double totalWeight = 0.0;

        for (std::size_t i = 0; i < particles.size(); ++i) {
            CpuPoint& particle = particles[i];
            particle.x += delta.x + motionStddev * normal(rng);
            particle.y += delta.y + motionStddev * normal(rng);

            float weight = 1.0f;
            for (std::size_t sensorIndex = 0; sensorIndex < sensorCount;
                 ++sensorIndex) {
                const CpuRangeSensor& sensor = sensors[sensorIndex];
                const float dx = particle.x - sensor.beacon.x;
                const float dy = particle.y - sensor.beacon.y;
                const float distance = std::sqrt(dx * dx + dy * dy);
                const float z = (distance - sensor.measured) / sensor.sigma;
                weight *= std::exp(-0.5f * z * z);
            }

            weights[i] = weight;
            weightedX += static_cast<double>(particle.x) * weight;
            weightedY += static_cast<double>(particle.y) * weight;
            totalWeight += weight;
            cumulativeWeights[i] = totalWeight;
        }

        prediction = {static_cast<float>(weightedX / totalWeight),
                      static_cast<float>(weightedY / totalWeight)};

        const double offset = uniform(rng);
        const double particleCount = static_cast<double>(particles.size());
        std::size_t selected = 0;
        for (std::size_t i = 0; i < particles.size(); ++i) {
            const double target =
                (offset + static_cast<double>(i)) / particleCount * totalWeight;
            while (selected + 1 < particles.size() &&
                   cumulativeWeights[selected] < target) {
                ++selected;
            }
            resampledParticles[i] = particles[selected];
            weights[i] = 1.0f / static_cast<float>(particles.size());
        }

        particles.swap(resampledParticles);
    }

    CpuPoint getPrediction() const {
        return prediction;
    }

private:
    std::vector<CpuPoint> particles;
    std::vector<CpuPoint> resampledParticles;
    std::vector<float> weights;
    std::vector<double> cumulativeWeights;
    std::mt19937_64 rng;
    std::normal_distribution<float> normal{0.0f, 1.0f};
    std::uniform_real_distribution<double> uniform{0.0, 1.0};
    CpuPoint prediction{};
};
