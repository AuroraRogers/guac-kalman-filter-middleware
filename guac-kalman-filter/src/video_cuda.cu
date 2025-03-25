/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "kalman_cuda.h"
#include "kalman_filter.h"
#include "video_cuda.h"

// 从kalman_cuda.cu中引用外部变量
extern int g_kalman_log_level;
extern bool cuda_initialized;

// 视频帧处理的CUDA内存
static unsigned char* d_frame_in = NULL;      // 输入帧
static unsigned char* d_frame_out = NULL;     // 输出帧
static unsigned char* d_frame_temp = NULL;    // 临时帧缓冲区
static int g_max_frame_size = 0;              // 最大帧大小
static bool video_cuda_initialized = false;   // 视频CUDA资源初始化标志

// 日志宏定义，复用kalman_cuda.h中的定义
#define VIDEO_LOG(level, ...) \
    do { \
        if (level <= g_kalman_log_level) { \
            fprintf(stderr, "[CUDA Video][%s] ", \
                   (level == KALMAN_LOG_ERROR) ? "ERROR" : \
                   (level == KALMAN_LOG_WARNING) ? "WARNING" : \
                   (level == KALMAN_LOG_INFO) ? "INFO" : \
                   (level == KALMAN_LOG_DEBUG) ? "DEBUG" : "TRACE"); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
        } \
    } while(0)

// CUDA错误检查宏，复用kalman_cuda.h中的定义
#define VIDEO_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            VIDEO_LOG(KALMAN_LOG_ERROR, "CUDA operation failed: %s", cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

/**
 * 初始化视频处理的CUDA资源
 * 
 * @param max_width
 *     最大视频宽度
 * 
 * @param max_height
 *     最大视频高度
 * 
 * @return
 *     初始化成功返回true，否则返回false
 */
bool cuda_init_video(int max_width, int max_height) {
    if (video_cuda_initialized) {
        VIDEO_LOG(KALMAN_LOG_INFO, "视频CUDA资源已初始化");
        return true;
    }
    
    // 检查CUDA是否已初始化
    if (!cuda_initialized) {
        VIDEO_LOG(KALMAN_LOG_ERROR, "CUDA资源尚未初始化，请先调用cuda_init_kalman()");
        return false;
    }
    
    // 计算最大帧大小（假设4通道RGBA格式）
    g_max_frame_size = max_width * max_height * 4;
    
    // 分配设备内存
    VIDEO_LOG(KALMAN_LOG_DEBUG, "为视频帧分配设备内存，大小: %d bytes", g_max_frame_size);
    
    VIDEO_CUDA_CHECK(cudaMalloc((void**)&d_frame_in, g_max_frame_size));
    VIDEO_CUDA_CHECK(cudaMalloc((void**)&d_frame_out, g_max_frame_size));
    VIDEO_CUDA_CHECK(cudaMalloc((void**)&d_frame_temp, g_max_frame_size));
    
    video_cuda_initialized = true;
    VIDEO_LOG(KALMAN_LOG_INFO, "视频CUDA资源初始化成功，最大分辨率: %dx%d", max_width, max_height);
    return true;
}

/**
 * 清理视频处理的CUDA资源
 */
void cuda_cleanup_video(void) {
    VIDEO_LOG(KALMAN_LOG_INFO, "清理视频CUDA资源");
    
    if (!video_cuda_initialized) {
        VIDEO_LOG(KALMAN_LOG_INFO, "无需清理（视频CUDA资源未初始化）");
        return;
    }
    
    // 释放设备内存
    cudaError_t err;
    
    err = cudaFree(d_frame_in);
    if (err != cudaSuccess) VIDEO_LOG(KALMAN_LOG_WARNING, "释放d_frame_in时出错: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_frame_out);
    if (err != cudaSuccess) VIDEO_LOG(KALMAN_LOG_WARNING, "释放d_frame_out时出错: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_frame_temp);
    if (err != cudaSuccess) VIDEO_LOG(KALMAN_LOG_WARNING, "释放d_frame_temp时出错: %s", cudaGetErrorString(err));
    
    d_frame_in = NULL;
    d_frame_out = NULL;
    d_frame_temp = NULL;
    g_max_frame_size = 0;
    video_cuda_initialized = false;
    
    VIDEO_LOG(KALMAN_LOG_INFO, "视频CUDA资源清理完成");
}

/**
 * 高斯模糊核函数
 */
__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, 
                                    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            float kernel[9] = {1.0f/16, 2.0f/16, 1.0f/16,
                              2.0f/16, 4.0f/16, 2.0f/16,
                              1.0f/16, 2.0f/16, 1.0f/16};
            int count = 0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int idx = (iy * width + ix) * channels + c;
                        sum += input[idx] * kernel[count];
                    }
                    count++;
                }
            }
            
            int idx = (y * width + x) * channels + c;
            output[idx] = (unsigned char)sum;
        }
    }
}

/**
 * 锐化核函数
 */
__global__ void sharpen_kernel(const unsigned char* input, unsigned char* output, 
                              int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            float kernel[9] = {0, -1, 0,
                              -1, 5, -1,
                               0, -1, 0};
            int count = 0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int idx = (iy * width + ix) * channels + c;
                        sum += input[idx] * kernel[count];
                    }
                    count++;
                }
            }
            
            int idx = (y * width + x) * channels + c;
            // 确保值在0-255范围内
            sum = sum < 0 ? 0 : (sum > 255 ? 255 : sum);
            output[idx] = (unsigned char)sum;
        }
    }
}

/**
 * 卡尔曼滤波核函数 - 对每个像素应用卡尔曼滤波
 */
__global__ void kalman_filter_kernel(const unsigned char* input, unsigned char* output, 
                                    float* state, float* covariance,
                                    float process_noise, float measurement_noise,
                                    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;
            int state_idx = (y * width + x) * channels * 2 + c * 2;
            int cov_idx = (y * width + x) * channels * 4 + c * 4;
            
            // 当前状态和协方差
            float current_state = state[state_idx];
            float current_velocity = state[state_idx + 1];
            float p00 = covariance[cov_idx];
            float p01 = covariance[cov_idx + 1];
            float p10 = covariance[cov_idx + 2];
            float p11 = covariance[cov_idx + 3];
            
            // 预测步骤
            float predicted_state = current_state + current_velocity;
            float predicted_p00 = p00 + p01 + p10 + p11 + process_noise;
            float predicted_p01 = p01 + p11;
            float predicted_p10 = p10 + p11;
            float predicted_p11 = p11 + process_noise;
            
            // 更新步骤
            float measurement = (float)input[idx];
            float innovation = measurement - predicted_state;
            float innovation_covariance = predicted_p00 + measurement_noise;
            float kalman_gain = predicted_p00 / innovation_covariance;
            float kalman_gain_velocity = predicted_p10 / innovation_covariance;
            
            // 更新状态和协方差
            float updated_state = predicted_state + kalman_gain * innovation;
            float updated_velocity = current_velocity + kalman_gain_velocity * innovation;
            float updated_p00 = (1 - kalman_gain) * predicted_p00;
            float updated_p01 = (1 - kalman_gain) * predicted_p01;
            float updated_p10 = predicted_p10 - kalman_gain_velocity * predicted_p00;
            float updated_p11 = predicted_p11 - kalman_gain_velocity * predicted_p01;
            
            // 保存更新后的状态和协方差
            state[state_idx] = updated_state;
            state[state_idx + 1] = updated_velocity;
            covariance[cov_idx] = updated_p00;
            covariance[cov_idx + 1] = updated_p01;
            covariance[cov_idx + 2] = updated_p10;
            covariance[cov_idx + 3] = updated_p11;
            
            // 更新输出图像
            output[idx] = (unsigned char)(updated_state < 0 ? 0 : (updated_state > 255 ? 255 : updated_state));
        }
    }
}

/**
 * 计算PSNR（峰值信噪比）核函数
 */
__global__ void calculate_psnr_kernel(const unsigned char* original, const unsigned char* processed,
                                     float* mse_result, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    extern __shared__ float shared_mse[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shared_mse[tid] = 0.0f;
    
    if (x < width && y < height) {
        float mse = 0.0f;
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;
            float diff = (float)original[idx] - (float)processed[idx];
            mse += diff * diff;
        }
        mse /= channels;
        shared_mse[tid] = mse;
    }
    
    __syncthreads();
    
    // 规约求和
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mse[tid] += shared_mse[tid + s];
        }
        __syncthreads();
    }
    
    // 将结果写回全局内存
    if (tid == 0) {
        atomicAdd(mse_result, shared_mse[0]);
    }
}

/**
 * 计算SSIM（结构相似性）核函数
 */
__global__ void calculate_ssim_kernel(const unsigned char* original, const unsigned char* processed,
                                     float* ssim_result, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    extern __shared__ float shared_ssim[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shared_ssim[tid] = 0.0f;
    
    if (x < width && y < height) {
        // 计算8x8窗口的SSIM
        if (x + 7 < width && y + 7 < height) {
            float mean_orig = 0.0f, mean_proc = 0.0f;
            float var_orig = 0.0f, var_proc = 0.0f, covar = 0.0f;
            
            // 计算均值
            for (int wy = 0; wy < 8; wy++) {
                for (int wx = 0; wx < 8; wx++) {
                    for (int c = 0; c < channels; c++) {
                        int idx = ((y + wy) * width + (x + wx)) * channels + c;
                        mean_orig += original[idx];
                        mean_proc += processed[idx];
                    }
                }
            }
            
            float n = 8.0f * 8.0f * channels;
            mean_orig /= n;
            mean_proc /= n;
            
            // 计算方差和协方差
            for (int wy = 0; wy < 8; wy++) {
                for (int wx = 0; wx < 8; wx++) {
                    for (int c = 0; c < channels; c++) {
                        int idx = ((y + wy) * width + (x + wx)) * channels + c;
                        float diff_orig = original[idx] - mean_orig;
                        float diff_proc = processed[idx] - mean_proc;
                        var_orig += diff_orig * diff_orig;
                        var_proc += diff_proc * diff_proc;
                        covar += diff_orig * diff_proc;
                    }
                }
            }
            
            var_orig /= n - 1.0f;
            var_proc /= n - 1.0f;
            covar /= n - 1.0f;
            
            // SSIM常数
            float C1 = (0.01f * 255.0f) * (0.01f * 255.0f);
            float C2 = (0.03f * 255.0f) * (0.03f * 255.0f);
            
            // 计算SSIM
            float numerator = (2.0f * mean_orig * mean_proc + C1) * (2.0f * covar + C2);
            float denominator = (mean_orig * mean_orig + mean_proc * mean_proc + C1) * (var_orig + var_proc + C2);
            float ssim = numerator / denominator;
            
            shared_ssim[tid] = ssim;
        }
    }
    
    __syncthreads();
    
    // 规约求和
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_ssim[tid] += shared_ssim[tid + s];
        }
        __syncthreads();
    }
    
    // 将结果写回全局内存
    if (tid == 0) {
        atomicAdd(ssim_result, shared_ssim[0]);
    }
}

/**
 * 处理视频帧 - 应用卡尔曼滤波和图像增强
 * 
 * @param frame_data
 *     输入视频帧数据
 * 
 * @param width
 *     帧宽度
 * 
 * @param height
 *     帧高度
 * 
 * @param channels
 *     颜色通道数（通常为4，RGBA）
 * 
 * @param quality
 *     处理质量（0-100）
 * 
 * @param output_data
 *     输出视频帧数据缓冲区
 * 
 * @return
 *     处理成功返回true，否则返回false
 */
bool cuda_process_video_frame(const unsigned char* frame_data, int width, int height, 
                             int channels, int quality, unsigned char* output_data) {
    if (!video_cuda_initialized) {
        VIDEO_LOG(KALMAN_LOG_ERROR, "视频CUDA资源未初始化，请先调用cuda_init_video()");
        return false;
    }
    
    // 检查帧大小
    int frame_size = width * height * channels;
    if (frame_size > g_max_frame_size) {
        VIDEO_LOG(KALMAN_LOG_ERROR, "帧大小超出预分配内存: %d > %d", frame_size, g_max_frame_size);
        return false;
    }
    
    // 复制输入帧到设备内存
    VIDEO_CUDA_CHECK(cudaMemcpy(d_frame_in, frame_data, frame_size, cudaMemcpyHostToDevice));
    
    // 设置CUDA网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // 根据质量参数选择处理方式
    if (quality < 30) {
        // 低质量：应用高斯模糊减少噪点和细节
        gaussian_blur_kernel<<<gridSize, blockSize>>>(d_frame_in, d_frame_temp, width, height, channels);
        
        // 应用卡尔曼滤波进行时间域平滑
        // 为简化实现，这里使用静态内存来存储状态和协方差
        static float* d_state = NULL;
        static float* d_covariance = NULL;
        static bool state_initialized = false;
        
        if (!state_initialized) {
            // 首次运行时分配内存
            cudaMalloc((void**)&d_state, width * height * channels * 2 * sizeof(float));
            cudaMalloc((void**)&d_covariance, width * height * channels * 4 * sizeof(float));
            
            // 初始化为零
            cudaMemset(d_state, 0, width * height * channels * 2 * sizeof(float));
            cudaMemset(d_covariance, 0, width * height * channels * 4 * sizeof(float));
            
            state_initialized = true;
        }
        
        // 应用卡尔曼滤波
        float process_noise = 0.01f;
        float measurement_noise = 0.1f;
        kalman_filter_kernel<<<gridSize, blockSize>>>(d_frame_temp, d_frame_out, 
                                                    d_state, d_covariance,
                                                    process_noise, measurement_noise,
                                                    width, height, channels);
    } 
    else if (quality < 70) {
        // 中等质量：应用轻度锐化
        sharpen_kernel<<<gridSize, blockSize>>>(d_frame_in, d_frame_out, width, height, channels);
    }
    else {
        // 高质量：应用强锐化增强细节
        // 先进行高斯模糊去噪
        gaussian_blur_kernel<<<gridSize, blockSize>>>(d_frame_in, d_frame_temp, width, height, channels);
        
        // 然后应用锐化
        sharpen_kernel<<<gridSize, blockSize>>>(d_frame_temp, d_frame_out, width, height, channels);
    }
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        VIDEO_LOG(KALMAN_LOG_ERROR, "CUDA核函数执行失败: %s", cudaGetErrorString(err));
        return false;
    }
    
    // 复制结果回主机内存
    VIDEO_CUDA_CHECK(cudaMemcpy(output_data, d_frame_out, frame_size, cudaMemcpyDeviceToHost));
    
    VIDEO_LOG(KALMAN_LOG_DEBUG, "视频帧处理完成，分辨率: %dx%d, 质量: %d", width, height, quality);
    return true;
}