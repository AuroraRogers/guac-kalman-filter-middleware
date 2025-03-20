#pragma once

#include "filter.hpp"

namespace guac {

/**
 * 卡尔曼滤波器参数
 */
struct KalmanParams {
    float process_noise = 1e-4f;     ///< 过程噪声（较小值意味着对系统模型更信任）
    float measurement_noise = 1e-1f; ///< 测量噪声（较大值意味着对测量值不太信任）
    float error_cov_post = 1.0f;     ///< 初始后验估计误差协方差
};

/**
 * CPU 版卡尔曼滤波器
 */
class KalmanFilter : public Filter {
public:
    KalmanFilter(int width, int height, const KalmanParams& params = KalmanParams());
    ~KalmanFilter() override;
    
    bool process(ImageData& image) override;
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * CUDA 加速版卡尔曼滤波器
 */
class CudaKalmanFilter : public Filter {
public:
    CudaKalmanFilter(int width, int height, const KalmanParams& params = KalmanParams());
    ~CudaKalmanFilter() override;
    
    bool process(ImageData& image) override;
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace guac
