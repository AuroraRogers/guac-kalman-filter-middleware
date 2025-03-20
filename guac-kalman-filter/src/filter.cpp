#include "filter.hpp"
#include "kalman.hpp"

#include <iostream>

namespace guac {

std::unique_ptr<Filter> Filter::create(int width, int height, bool use_cuda) {
    try {
        if (use_cuda) {
            std::cout << "Creating CUDA Kalman filter (" << width << "x" << height << ")" << std::endl;
            return std::make_unique<CudaKalmanFilter>(width, height);
        } else {
            std::cout << "Creating CPU Kalman filter (" << width << "x" << height << ")" << std::endl;
            return std::make_unique<KalmanFilter>(width, height);
        }
    } catch (const std::exception& e) {
        if (use_cuda) {
            std::cerr << "Failed to create CUDA filter: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation..." << std::endl;
            return std::make_unique<KalmanFilter>(width, height);
        }
        throw;
    }
}

} // namespace guac
