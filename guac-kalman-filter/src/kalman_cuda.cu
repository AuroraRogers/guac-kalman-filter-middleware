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

/* Forward declarations */
void matrix_multiply_4x4_4x4(double* A, double* B, double* C);
void matrix_multiply_4x4_4x1(double* A, double* b, double* c);
void matrix_transpose_4x4(double* A, double* A_T);
void matrix_add_4x4(double* A, double* B, double* C);
void matrix_subtract_4x4(double* A, double* B, double* C);
void matrix_inverse_2x2(double* A, double* A_inv);

// CUDA error checking macro
// 日志级别定义
#define KALMAN_LOG_ERROR   0
#define KALMAN_LOG_WARNING 1
#define KALMAN_LOG_INFO    2
#define KALMAN_LOG_DEBUG   3
#define KALMAN_LOG_TRACE   4

// 当前日志级别 (默认值，可以通过cuda_kalman_set_log_level函数修改)
int g_kalman_log_level = KALMAN_LOG_DEBUG;

// 设置CUDA卡尔曼滤波器的日志级别
void cuda_kalman_set_log_level(int level) {
    if (level >= KALMAN_LOG_ERROR && level <= KALMAN_LOG_TRACE) {
        g_kalman_log_level = level;
    }
}

// 日志宏定义
#define KALMAN_LOG(level, ...) \
    do { \
        if (level <= g_kalman_log_level) { \
            fprintf(stderr, "[CUDA Kalman][%s] ", \
                   (level == KALMAN_LOG_ERROR) ? "ERROR" : \
                   (level == KALMAN_LOG_WARNING) ? "WARNING" : \
                   (level == KALMAN_LOG_INFO) ? "INFO" : \
                   (level == KALMAN_LOG_DEBUG) ? "DEBUG" : "TRACE"); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
        } \
    } while(0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            KALMAN_LOG(KALMAN_LOG_ERROR, "CUDA operation failed: %s", cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

// Device memory pointers
static double *d_F = NULL;           // State transition matrix
static double *d_H = NULL;           // Measurement matrix
static double *d_Q = NULL;           // Process noise covariance
static double *d_R = NULL;           // Measurement noise covariance
static double *d_P = NULL;           // Error covariance matrix
static double *d_K = NULL;           // Kalman gain matrix
static double *d_I = NULL;           // Identity matrix
static double *d_state = NULL;       // State vector
static double *d_measurement = NULL; // Measurement vector
static double *d_temp1 = NULL;       // Temporary workspace for calculations
static double *d_temp2 = NULL;       // Temporary workspace for calculations
static double *d_temp3 = NULL;       // Temporary workspace for calculations
static double *d_temp4 = NULL;       // Temporary workspace for calculations

// Flag to track if CUDA resources have been initialized
bool cuda_initialized = false;

// Constants for matrix dimensions
#define STATE_DIM 4
#define MEAS_DIM 2

/**
 * Matrix multiplication kernel (C = A * B)
 */
__global__ void matrix_multiply_kernel(const double* A, const double* B, double* C, 
                                       int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A_rows && col < B_cols) {
        double sum = 0.0;
        for (int i = 0; i < A_cols; i++) {
            sum += A[row * A_cols + i] * B[i * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

/**
 * Matrix addition kernel (C = A + B)
 */
__global__ void matrix_add_kernel(const double* A, const double* B, double* C,
                                  int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Matrix subtraction kernel (C = A - B)
 */
__global__ void matrix_subtract_kernel(const double* A, const double* B, double* C,
                                       int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] - B[idx];
    }
}

/**
 * Matrix transpose kernel (B = A^T)
 */
__global__ void matrix_transpose_kernel(const double* A, double* B,
                                        int A_rows, int A_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A_rows && col < A_cols) {
        B[col * A_rows + row] = A[row * A_cols + col];
    }
}

/**
 * Matrix-vector multiply kernel (y = A * x)
 */
__global__ void matrix_vector_multiply_kernel(const double* A, const double* x, double* y,
                                              int A_rows, int A_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A_rows) {
        double sum = 0.0;
        for (int i = 0; i < A_cols; i++) {
            sum += A[row * A_cols + i] * x[i];
        }
        y[row] = sum;
    }
}

/**
 * Matrix inversion kernel for small matrices (4x4 or smaller)
 * Uses the adjugate method for inversion
 */
__global__ void matrix_invert_kernel(const double* A, double* A_inv, int dim) {
    // This kernel is designed to run with a single thread for simplicity
    // since we're dealing with small matrices
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // For a 2x2 matrix, we can use the direct formula
        if (dim == 2) {
            double det = A[0] * A[3] - A[1] * A[2];
            if (fabs(det) > 1e-10) {
                double inv_det = 1.0 / det;
                A_inv[0] = A[3] * inv_det;
                A_inv[1] = -A[1] * inv_det;
                A_inv[2] = -A[2] * inv_det;
                A_inv[3] = A[0] * inv_det;
            }
        }
        // For 4x4 matrices, we'd typically use a more sophisticated approach,
        // but for simplicity and since our state matrix is small, we'll use 
        // a direct cofactor method
        else if (dim == 4) {
            // Implementation for 4x4 matrix inversion
            // For now, we'll use a simplified approach that works for our specific case
            // In a production environment, you'd want a more general method
            
            // Calculate cofactors, adjugate, and determinant
            // This is a simplified implementation for the specific case of 
            // Kalman filter matrices where we often encounter well-conditioned matrices
            
            // This is a placeholder - a full implementation would calculate the minor 
            // matrices, their determinants, and construct the adjugate matrix
            
            // Note: In practice, you might want to use cuBLAS or another library
            // for larger matrix operations instead of implementing your own kernels
        }
    }
}

/**
 * Initialize CUDA resources for Kalman filter calculations
 */
bool cuda_init_kalman(void) {
    KALMAN_LOG(KALMAN_LOG_INFO, "Initializing CUDA Kalman filter resources");
    
    if (cuda_initialized) {
        KALMAN_LOG(KALMAN_LOG_INFO, "CUDA resources already initialized");
        return true;
    }
    
    // 获取CUDA设备信息
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to get CUDA device count: %s", cudaGetErrorString(error));
        return false;
    }
    
    if (deviceCount == 0) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "No CUDA-capable devices found");
        return false;
    }
    
    // 获取当前设备信息
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    KALMAN_LOG(KALMAN_LOG_INFO, "Using CUDA device %d: %s", device, deviceProp.name);
    KALMAN_LOG(KALMAN_LOG_DEBUG, "  Compute capability: %d.%d", deviceProp.major, deviceProp.minor);
    KALMAN_LOG(KALMAN_LOG_DEBUG, "  Total global memory: %.2f GB", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Allocate device memory for matrices
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Allocating device memory for Kalman filter matrices");
    
    CUDA_CHECK(cudaMalloc((void**)&d_F, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_H, MEAS_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Q, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_R, MEAS_DIM * MEAS_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_P, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_K, STATE_DIM * MEAS_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_I, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_state, STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_measurement, MEAS_DIM * sizeof(double)));
    
    // Allocate device memory for temporary calculations
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Allocating device memory for temporary calculations");
    CUDA_CHECK(cudaMalloc((void**)&d_temp1, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp2, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp3, STATE_DIM * MEAS_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp4, MEAS_DIM * MEAS_DIM * sizeof(double)));
    
    cuda_initialized = true;
    KALMAN_LOG(KALMAN_LOG_INFO, "CUDA Kalman filter resources initialized successfully");
    return true;
}

/**
 * Clean up CUDA resources
 */
void cuda_cleanup_kalman(void) {
    KALMAN_LOG(KALMAN_LOG_INFO, "Cleaning up CUDA Kalman filter resources");
    
    if (!cuda_initialized) {
        KALMAN_LOG(KALMAN_LOG_INFO, "No CUDA resources to clean up (not initialized)");
        return;
    }
    
    // Free device memory
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Freeing device memory for Kalman filter matrices");
    
    cudaError_t err;
    
    err = cudaFree(d_F);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_F: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_H);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_H: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_Q);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_Q: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_R);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_R: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_P);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_P: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_K);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_K: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_I);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_I: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_state);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_state: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_measurement);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_measurement: %s", cudaGetErrorString(err));
    
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Freeing device memory for temporary calculations");
    
    err = cudaFree(d_temp1);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_temp1: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_temp2);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_temp2: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_temp3);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_temp3: %s", cudaGetErrorString(err));
    
    err = cudaFree(d_temp4);
    if (err != cudaSuccess) KALMAN_LOG(KALMAN_LOG_WARNING, "Error freeing d_temp4: %s", cudaGetErrorString(err));
    
    cuda_initialized = false;
    KALMAN_LOG(KALMAN_LOG_INFO, "CUDA Kalman filter resources cleaned up successfully");
}

/**
 * Initialize the Kalman filter matrices on the GPU
 */
bool cuda_kalman_init_matrices(const double* F, const double* H, const double* Q, 
                             const double* R, const double* P, const double* state) {
    KALMAN_LOG(KALMAN_LOG_INFO, "Initializing Kalman filter matrices on GPU");
    
    if (!cuda_initialized && !cuda_init_kalman()) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to initialize CUDA resources");
        return false;
    }
    
    // 记录矩阵内容用于调试
    if (g_kalman_log_level >= KALMAN_LOG_DEBUG) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "State transition matrix F:");
        for (int i = 0; i < STATE_DIM; i++) {
            char row_str[128] = {0};
            int offset = 0;
            for (int j = 0; j < STATE_DIM; j++) {
                offset += snprintf(row_str + offset, sizeof(row_str) - offset, 
                                  "%.4f ", F[i * STATE_DIM + j]);
            }
            KALMAN_LOG(KALMAN_LOG_DEBUG, "  %s", row_str);
        }
        
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Measurement matrix H:");
        for (int i = 0; i < MEAS_DIM; i++) {
            char row_str[128] = {0};
            int offset = 0;
            for (int j = 0; j < STATE_DIM; j++) {
                offset += snprintf(row_str + offset, sizeof(row_str) - offset, 
                                  "%.4f ", H[i * STATE_DIM + j]);
            }
            KALMAN_LOG(KALMAN_LOG_DEBUG, "  %s", row_str);
        }
        
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Initial state vector:");
        char state_str[128] = {0};
        int offset = 0;
        for (int i = 0; i < STATE_DIM; i++) {
            offset += snprintf(state_str + offset, sizeof(state_str) - offset, 
                              "%.4f ", state[i]);
        }
        KALMAN_LOG(KALMAN_LOG_DEBUG, "  %s", state_str);
    }
    
    // 记录开始时间用于性能测量
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Copy matrices to device memory
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Copying matrices to device memory");
    CUDA_CHECK(cudaMemcpy(d_F, F, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, H, MEAS_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R, R, MEAS_DIM * MEAS_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P, P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, state, STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize identity matrix
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Initializing identity matrix");
    double I[STATE_DIM * STATE_DIM] = {0};
    for (int i = 0; i < STATE_DIM; i++) {
        I[i * STATE_DIM + i] = 1.0;
    }
    CUDA_CHECK(cudaMemcpy(d_I, I, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    // 记录结束时间并计算耗时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    KALMAN_LOG(KALMAN_LOG_INFO, "Kalman filter matrices initialized successfully (%.3f ms)", elapsed_time);
    return true;
}

/**
 * Update Kalman filter process noise matrix Q
 */
bool cuda_kalman_update_Q(const double* Q) {
    KALMAN_LOG(KALMAN_LOG_INFO, "Updating Kalman filter process noise matrix Q");
    
    if (!cuda_initialized && !cuda_init_kalman()) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to initialize CUDA resources");
        return false;
    }
    
    // 记录矩阵内容用于调试
    if (g_kalman_log_level >= KALMAN_LOG_DEBUG) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "New process noise matrix Q:");
        for (int i = 0; i < STATE_DIM; i++) {
            char row_str[128] = {0};
            int offset = 0;
            for (int j = 0; j < STATE_DIM; j++) {
                offset += snprintf(row_str + offset, sizeof(row_str) - offset, 
                                  "%.4f ", Q[i * STATE_DIM + j]);
            }
            KALMAN_LOG(KALMAN_LOG_DEBUG, "  %s", row_str);
        }
    }
    
    // 记录开始时间用于性能测量
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    CUDA_CHECK(cudaMemcpy(d_Q, Q, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    // 记录结束时间并计算耗时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    KALMAN_LOG(KALMAN_LOG_INFO, "Process noise matrix Q updated successfully (%.3f ms)", elapsed_time);
    return true;
}

/**
 * Update Kalman filter measurement noise matrix R
 */
bool cuda_kalman_update_R(const double* R) {
    KALMAN_LOG(KALMAN_LOG_INFO, "Updating Kalman filter measurement noise matrix R");
    
    if (!cuda_initialized && !cuda_init_kalman()) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to initialize CUDA resources");
        return false;
    }
    
    // 记录矩阵内容用于调试
    if (g_kalman_log_level >= KALMAN_LOG_DEBUG) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "New measurement noise matrix R:");
        for (int i = 0; i < MEAS_DIM; i++) {
            char row_str[64] = {0};
            int offset = 0;
            for (int j = 0; j < MEAS_DIM; j++) {
                offset += snprintf(row_str + offset, sizeof(row_str) - offset, 
                                  "%.4f ", R[i * MEAS_DIM + j]);
            }
            KALMAN_LOG(KALMAN_LOG_DEBUG, "  %s", row_str);
        }
    }
    
    // 记录开始时间用于性能测量
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    CUDA_CHECK(cudaMemcpy(d_R, R, MEAS_DIM * MEAS_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    // 记录结束时间并计算耗时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    KALMAN_LOG(KALMAN_LOG_INFO, "Measurement noise matrix R updated successfully (%.3f ms)", elapsed_time);
    return true;
}

/**
 * Predict step of the Kalman filter
 * 
 * Performs:
 * x = F * x
 * P = F * P * F^T + Q
 */
bool cuda_kalman_predict(double dt) {
    KALMAN_LOG(KALMAN_LOG_INFO, "Performing Kalman filter prediction step with dt=%.6f", dt);
    
    if (!cuda_initialized && !cuda_init_kalman()) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to initialize CUDA resources");
        return false;
    }
    
    // 获取当前状态用于日志记录
    double current_state[STATE_DIM];
    cudaError_t err = cudaMemcpy(current_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Current state before prediction: [%.4f, %.4f, %.4f, %.4f]", 
                 current_state[0], current_state[1], current_state[2], current_state[3]);
        
        // 添加TRACE级别的详细日志
        if (g_kalman_log_level >= KALMAN_LOG_TRACE) {
            double P[STATE_DIM * STATE_DIM];
            double Q[STATE_DIM * STATE_DIM];
            double F[STATE_DIM * STATE_DIM];
            
            cudaMemcpy(P, d_P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(Q, d_Q, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(F, d_F, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "预测前状态向量 x = [%.6f, %.6f, %.6f, %.6f]", 
                     current_state[0], current_state[1], current_state[2], current_state[3]);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "状态转移矩阵 F:");
            for (int i = 0; i < STATE_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                         F[i*STATE_DIM+0], F[i*STATE_DIM+1], F[i*STATE_DIM+2], F[i*STATE_DIM+3]);
            }
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "过程噪声协方差矩阵 Q:");
            for (int i = 0; i < STATE_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                         Q[i*STATE_DIM+0], Q[i*STATE_DIM+1], Q[i*STATE_DIM+2], Q[i*STATE_DIM+3]);
            }
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "预测前误差协方差矩阵 P:");
            for (int i = 0; i < STATE_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                         P[i*STATE_DIM+0], P[i*STATE_DIM+1], P[i*STATE_DIM+2], P[i*STATE_DIM+3]);
            }
        }
    }
    
    // Update state transition matrix F for the current time step if dt has changed
    // For our model: F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]
    if (dt > 0) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Updating state transition matrix F with dt=%.6f", dt);
        double F[STATE_DIM * STATE_DIM] = {
            1.0, 0.0, dt, 0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        CUDA_CHECK(cudaMemcpy(d_F, F, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    // 记录开始时间用于性能测量
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Calculate x = F * x
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating x = F * x");
    dim3 blockDim(32, 1);
    dim3 gridDim((STATE_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    // We need to save the current state to use it for calculation
    double temp_state[STATE_DIM];
    CUDA_CHECK(cudaMemcpy(temp_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost));
    
    matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_F, d_state, d_state, STATE_DIM, STATE_DIM);
    
    // Calculate F * P
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating F * P");
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_F, d_P, d_temp1, STATE_DIM, STATE_DIM, STATE_DIM);
    
    // Calculate (F * P) * F^T
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating (F * P) * F^T");
    // First transpose F
    matrix_transpose_kernel<<<gridDim, blockDim>>>(d_F, d_temp2, STATE_DIM, STATE_DIM);
    
    // Then multiply
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_temp1, d_temp2, d_temp3, STATE_DIM, STATE_DIM, STATE_DIM);
    
    // Add Q: P = (F * P * F^T) + Q
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating P = (F * P * F^T) + Q");
    matrix_add_kernel<<<gridDim, blockDim>>>(d_temp3, d_Q, d_P, STATE_DIM, STATE_DIM);
    
    // 记录结束时间并计算耗时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 获取更新后的状态用于日志记录
    double updated_state[STATE_DIM];
    err = cudaMemcpy(updated_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼预测] ===== 预测状态: [%.4f, %.4f, %.4f, %.4f] (耗时: %.3f ms) =====",
              updated_state[0], updated_state[1], updated_state[2], updated_state[3], elapsed_time);
    
    // 计算状态变化量
    double state_change[STATE_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        state_change[i] = updated_state[i] - current_state[i];
    }
    
    // 记录状态变化量
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼预测变化] 位置变化: %.4f, 速度变化: %.4f, 加速度变化: %.4f, 抖动变化: %.4f",
              state_change[0], state_change[1], state_change[2], state_change[3]);
    
    // 记录预测置信度评估
    double confidence = 100.0 * (1.0 - fabs(state_change[1]/10.0));
    if (confidence < 0.0) confidence = 0.0;
    if (confidence > 100.0) confidence = 100.0;
    
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼预测评估] 预测置信度: %.2f%%", confidence);
        
        // 添加TRACE级别的详细日志，显示预测后的状态和协方差矩阵
        if (g_kalman_log_level >= KALMAN_LOG_TRACE) {
            double P[STATE_DIM * STATE_DIM];
            
            cudaMemcpy(P, d_P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "预测后状态向量 x = [%.6f, %.6f, %.6f, %.6f]", 
                     updated_state[0], updated_state[1], updated_state[2], updated_state[3]);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "预测后误差协方差矩阵 P:");
            for (int i = 0; i < STATE_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                         P[i*STATE_DIM+0], P[i*STATE_DIM+1], P[i*STATE_DIM+2], P[i*STATE_DIM+3]);
            }
            
            // 计算并显示状态变化
            double state_change[STATE_DIM];
            for (int i = 0; i < STATE_DIM; i++) {
                state_change[i] = updated_state[i] - current_state[i];
            }
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "状态变化量: [%.6f, %.6f, %.6f, %.6f]", 
                     state_change[0], state_change[1], state_change[2], state_change[3]);
        }
    } else {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to copy predicted state from device to host: %s", 
                  cudaGetErrorString(err));
    }
    
    return true;
}

/**
 * Update step of the Kalman filter with a new measurement
 * 
 * Performs:
 * y = z - H * x
 * S = H * P * H^T + R
 * K = P * H^T * S^-1
 * x = x + K * y
 * P = (I - K * H) * P
 */
bool cuda_kalman_update(const double* measurement, double* updated_state) {
    double improvement_percent = 0.0; // 添加变量声明
    double state_change[STATE_DIM]; // 添加state_change变量声明
    
    KALMAN_LOG(KALMAN_LOG_INFO, "Performing Kalman filter update step with measurement [%.4f, %.4f]", 
              measurement[0], measurement[1]);
    
    if (!cuda_initialized && !cuda_init_kalman()) {
        KALMAN_LOG(KALMAN_LOG_ERROR, "Failed to initialize CUDA resources");
        return false;
    }
    
    // 获取当前状态用于日志记录
    double current_state[STATE_DIM];
    cudaError_t err = cudaMemcpy(current_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Current state before update: [%.4f, %.4f, %.4f, %.4f]", 
                 current_state[0], current_state[1], current_state[2], current_state[3]);
        
        // 添加TRACE级别的详细日志
        if (g_kalman_log_level >= KALMAN_LOG_TRACE) {
            double P[STATE_DIM * STATE_DIM];
            double R[MEAS_DIM * MEAS_DIM];
            double H[MEAS_DIM * STATE_DIM];
            
            cudaMemcpy(P, d_P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(R, d_R, MEAS_DIM * MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(H, d_H, MEAS_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "更新前状态向量 x = [%.6f, %.6f, %.6f, %.6f]", 
                     current_state[0], current_state[1], current_state[2], current_state[3]);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "测量值 z = [%.6f, %.6f]", measurement[0], measurement[1]);
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "测量矩阵 H:");
            for (int i = 0; i < MEAS_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                         H[i*STATE_DIM+0], H[i*STATE_DIM+1], H[i*STATE_DIM+2], H[i*STATE_DIM+3]);
            }
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "测量噪声协方差矩阵 R:");
            for (int i = 0; i < MEAS_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f]",
                         R[i*MEAS_DIM+0], R[i*MEAS_DIM+1]);
            }
            
            KALMAN_LOG(KALMAN_LOG_TRACE, "更新前误差协方差矩阵 P:");
            for (int i = 0; i < STATE_DIM; i++) {
                KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                         P[i*STATE_DIM+0], P[i*STATE_DIM+1], P[i*STATE_DIM+2], P[i*STATE_DIM+3]);
            }
        }
    }
    
    // 记录开始时间用于性能测量
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Copy measurement to device
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Copying measurement to device");
    CUDA_CHECK(cudaMemcpy(d_measurement, measurement, MEAS_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blockDim(16, 16);
    dim3 gridDim;
    
    // Calculate predicted measurement: H * x
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating predicted measurement: H * x");
    blockDim = dim3(32, 1);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    double *d_predicted_measurement;
    CUDA_CHECK(cudaMalloc((void**)&d_predicted_measurement, MEAS_DIM * sizeof(double)));
    
    matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_H, d_state, d_predicted_measurement, MEAS_DIM, STATE_DIM);
    
    // 获取预测测量值用于日志记录
    double predicted_measurement[MEAS_DIM];
    err = cudaMemcpy(predicted_measurement, d_predicted_measurement, MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Predicted measurement: [%.4f, %.4f]", 
                 predicted_measurement[0], predicted_measurement[1]);
    }
    
    // Calculate measurement residual: y = z - H * x
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating measurement residual: y = z - H * x");
    double *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_y, MEAS_DIM * sizeof(double)));
    
    blockDim = dim3(32, 1);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    matrix_subtract_kernel<<<gridDim, blockDim>>>(d_measurement, d_predicted_measurement, d_y, MEAS_DIM, 1);
    
    // 获取测量残差用于日志记录
    double residual[MEAS_DIM];
    err = cudaMemcpy(residual, d_y, MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Measurement residual: [%.4f, %.4f]", 
                 residual[0], residual[1]);
    }
    
    // Calculate S = H * P * H^T + R
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating innovation covariance: S = H * P * H^T + R");
    // First calculate H * P
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (MEAS_DIM + blockDim.y - 1) / blockDim.y);
    
    double *d_HP;
    CUDA_CHECK(cudaMalloc((void**)&d_HP, MEAS_DIM * STATE_DIM * sizeof(double)));
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_H, d_P, d_HP, MEAS_DIM, STATE_DIM, STATE_DIM);
    
    // Transpose H
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Transposing measurement matrix H");
    double *d_HT;
    CUDA_CHECK(cudaMalloc((void**)&d_HT, STATE_DIM * MEAS_DIM * sizeof(double)));
    
    matrix_transpose_kernel<<<gridDim, blockDim>>>(d_H, d_HT, MEAS_DIM, STATE_DIM);
    
    // Calculate (H * P) * H^T
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating (H * P) * H^T");
    double *d_S;
    CUDA_CHECK(cudaMalloc((void**)&d_S, MEAS_DIM * MEAS_DIM * sizeof(double)));
    
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, (MEAS_DIM + blockDim.y - 1) / blockDim.y);
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_HP, d_HT, d_S, MEAS_DIM, STATE_DIM, MEAS_DIM);
    
    // Add R to get S = H * P * H^T + R
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Adding R to get S = H * P * H^T + R");
    matrix_add_kernel<<<gridDim, blockDim>>>(d_S, d_R, d_S, MEAS_DIM, MEAS_DIM);
    
    // Calculate S^-1 (inverse of innovation covariance)
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating S^-1 (inverse of innovation covariance)");
    double *d_S_inv;
    CUDA_CHECK(cudaMalloc((void**)&d_S_inv, MEAS_DIM * MEAS_DIM * sizeof(double)));
    
    // For 2x2 matrix, we can use a simple kernel
    matrix_invert_kernel<<<1, 1>>>(d_S, d_S_inv, MEAS_DIM);
    
    // Calculate Kalman gain: K = P * H^T * S^-1
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Calculating Kalman gain: K = P * H^T * S^-1");
    // First calculate P * H^T
    double *d_PHT;
    CUDA_CHECK(cudaMalloc((void**)&d_PHT, STATE_DIM * MEAS_DIM * sizeof(double)));
    
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_P, d_HT, d_PHT, STATE_DIM, STATE_DIM, MEAS_DIM);
    
    // Then calculate (P * H^T) * S^-1
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_PHT, d_S_inv, d_K, STATE_DIM, MEAS_DIM, MEAS_DIM);
    
    // 获取卡尔曼增益用于日志记录
    double K[STATE_DIM * MEAS_DIM];
    err = cudaMemcpy(K, d_K, STATE_DIM * MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && g_kalman_log_level >= KALMAN_LOG_DEBUG) {
        KALMAN_LOG(KALMAN_LOG_DEBUG, "Kalman gain matrix K:");
        for (int i = 0; i < STATE_DIM; i++) {
            char row_str[64] = {0};
            int offset = 0;
            for (int j = 0; j < MEAS_DIM; j++) {
                offset += snprintf(row_str + offset, sizeof(row_str) - offset, 
                                  "%.4f ", K[i * MEAS_DIM + j]);
            }
            KALMAN_LOG(KALMAN_LOG_DEBUG, "  %s", row_str);
        }
    }
    
    // Update state: x = x + K * y
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Updating state: x = x + K * y");
    double *d_Ky;
    CUDA_CHECK(cudaMalloc((void**)&d_Ky, STATE_DIM * sizeof(double)));
    
    blockDim = dim3(32, 1);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, 1);
    matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_K, d_y, d_Ky, STATE_DIM, MEAS_DIM);
    
    // Add to current state
    matrix_add_kernel<<<gridDim, blockDim>>>(d_state, d_Ky, d_state, STATE_DIM, 1);
    
    // Update error covariance: P = (I - K * H) * P
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Updating error covariance: P = (I - K * H) * P");
    // First calculate K * H
    double *d_KH;
    CUDA_CHECK(cudaMalloc((void**)&d_KH, STATE_DIM * STATE_DIM * sizeof(double)));
    
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_K, d_H, d_KH, STATE_DIM, MEAS_DIM, STATE_DIM);
    
    // Calculate I - K * H
    double *d_IKH;
    CUDA_CHECK(cudaMalloc((void**)&d_IKH, STATE_DIM * STATE_DIM * sizeof(double)));
    
    matrix_subtract_kernel<<<gridDim, blockDim>>>(d_I, d_KH, d_IKH, STATE_DIM, STATE_DIM);
    
    // Calculate (I - K * H) * P
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_IKH, d_P, d_P, STATE_DIM, STATE_DIM, STATE_DIM);
    
    // Free temporary memory
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Freeing temporary memory");
    cudaFree(d_predicted_measurement);
    cudaFree(d_y);
    cudaFree(d_HP);
    cudaFree(d_HT);
    cudaFree(d_S);
    cudaFree(d_S_inv);
    cudaFree(d_PHT);
    cudaFree(d_Ky);
    cudaFree(d_KH);
    cudaFree(d_IKH);

    // Copy updated state back to host
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Copying updated state back to host");
    CUDA_CHECK(cudaMemcpy(updated_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost));
    
    // 获取更新后的状态用于日志记录（使用明显的标记）
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼更新] ===== 更新后状态: [%.4f, %.4f, %.4f, %.4f] (耗时: %.3f ms) =====", 
              updated_state[0], updated_state[1], updated_state[2], updated_state[3], elapsed_time);
    
    // 计算状态变化量
    for (int i = 0; i < STATE_DIM; i++) {
        state_change[i] = updated_state[i] - current_state[i];
    }
    
    // 记录状态变化量
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼变化量] 位置: %.4f, 速度: %.4f, 加速度: %.4f, 抖动: %.4f",
              state_change[0], state_change[1], state_change[2], state_change[3]);
    
    // 计算改进百分比
    improvement_percent = 0.0;
    if (fabs(measurement[0]) > 0.001) { // 避免除以零
        improvement_percent = fabs((updated_state[0] - measurement[0]) / measurement[0]) * 100.0;
    }
    
    // 记录滤波效果评估
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼效果] 改进幅度: %.2f%%, 置信度: %.2f%%",
              improvement_percent, (1.0 - fabs(updated_state[1]/10.0)) * 100.0);
    
    // 添加TRACE级别的详细日志，显示更新后的状态和协方差矩阵
    if (g_kalman_log_level >= KALMAN_LOG_TRACE) {
        double P[STATE_DIM * STATE_DIM];
        double K[STATE_DIM * MEAS_DIM];
        
        cudaMemcpy(P, d_P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(K, d_K, STATE_DIM * MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "卡尔曼增益矩阵 K:");
        for (int i = 0; i < STATE_DIM; i++) {
            KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f]",
                     K[i*MEAS_DIM+0], K[i*MEAS_DIM+1]);
        }
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "更新后状态向量 x = [%.6f, %.6f, %.6f, %.6f]", 
                 updated_state[0], updated_state[1], updated_state[2], updated_state[3]);
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "更新后误差协方差矩阵 P:");
        for (int i = 0; i < STATE_DIM; i++) {
            KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                     P[i*STATE_DIM+0], P[i*STATE_DIM+1], P[i*STATE_DIM+2], P[i*STATE_DIM+3]);
        }
        
        // 计算并显示测量前后的差异
        double predicted_measurement[MEAS_DIM];
        double *d_predicted_measurement;
        cudaMalloc((void**)&d_predicted_measurement, MEAS_DIM * sizeof(double));
        
        dim3 blockDim(32, 1);
        dim3 gridDim((MEAS_DIM + blockDim.x - 1) / blockDim.x, 1);
        matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_H, d_state, d_predicted_measurement, MEAS_DIM, STATE_DIM);
        
        cudaMemcpy(predicted_measurement, d_predicted_measurement, MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_predicted_measurement);
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "测量值: [%.6f, %.6f], 预测测量值: [%.6f, %.6f], 差异: [%.6f, %.6f]", 
                 measurement[0], measurement[1],
                 predicted_measurement[0], predicted_measurement[1],
                 measurement[0] - predicted_measurement[0], measurement[1] - predicted_measurement[1]);
    }

    return true;
    
    // 记录结束时间并计算耗时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free temporary memory
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Freeing temporary memory");
    cudaFree(d_predicted_measurement);
    cudaFree(d_y);
    cudaFree(d_HP);
    cudaFree(d_HT);
    cudaFree(d_S);
    cudaFree(d_S_inv);
    cudaFree(d_PHT);
    cudaFree(d_Ky);
    cudaFree(d_KH);
    cudaFree(d_IKH);

    // Copy updated state back to host
    KALMAN_LOG(KALMAN_LOG_DEBUG, "Copying updated state back to host");
    CUDA_CHECK(cudaMemcpy(updated_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost));
    
    // 获取更新后的状态用于日志记录（使用明显的标记）
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼更新] ===== 更新后状态: [%.4f, %.4f, %.4f, %.4f] (耗时: %.3f ms) =====", 
              updated_state[0], updated_state[1], updated_state[2], updated_state[3], elapsed_time);
    
    // 计算状态变化量
    for (int i = 0; i < STATE_DIM; i++) {
        state_change[i] = updated_state[i] - current_state[i];
    }
    
    // 记录状态变化量
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼变化量] 位置: %.4f, 速度: %.4f, 加速度: %.4f, 抖动: %.4f",
              state_change[0], state_change[1], state_change[2], state_change[3]);
    
    // 计算改进百分比
    improvement_percent = 0.0;
    if (fabs(measurement[0]) > 0.001) { // 避免除以零
        improvement_percent = fabs((updated_state[0] - measurement[0]) / measurement[0]) * 100.0;
    }
    
    // 记录滤波效果评估
    KALMAN_LOG(KALMAN_LOG_INFO, "[CUDA卡尔曼效果] 改进幅度: %.2f%%, 置信度: %.2f%%",
              improvement_percent, (1.0 - fabs(updated_state[1]/10.0)) * 100.0);
    
    // 添加TRACE级别的详细日志，显示更新后的状态和协方差矩阵
    if (g_kalman_log_level >= KALMAN_LOG_TRACE) {
        double P[STATE_DIM * STATE_DIM];
        double K[STATE_DIM * MEAS_DIM];
        
        cudaMemcpy(P, d_P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(K, d_K, STATE_DIM * MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "卡尔曼增益矩阵 K:");
        for (int i = 0; i < STATE_DIM; i++) {
            KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f]",
                     K[i*MEAS_DIM+0], K[i*MEAS_DIM+1]);
        }
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "更新后状态向量 x = [%.6f, %.6f, %.6f, %.6f]", 
                 updated_state[0], updated_state[1], updated_state[2], updated_state[3]);
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "更新后误差协方差矩阵 P:");
        for (int i = 0; i < STATE_DIM; i++) {
            KALMAN_LOG(KALMAN_LOG_TRACE, "  [%.6f, %.6f, %.6f, %.6f]",
                     P[i*STATE_DIM+0], P[i*STATE_DIM+1], P[i*STATE_DIM+2], P[i*STATE_DIM+3]);
        }
        
        // 计算并显示测量前后的差异
        double predicted_measurement[MEAS_DIM];
        double *d_predicted_measurement;
        cudaMalloc((void**)&d_predicted_measurement, MEAS_DIM * sizeof(double));
        
        dim3 blockDim(32, 1);
        dim3 gridDim((MEAS_DIM + blockDim.x - 1) / blockDim.x, 1);
        matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_H, d_state, d_predicted_measurement, MEAS_DIM, STATE_DIM);
        
        cudaMemcpy(predicted_measurement, d_predicted_measurement, MEAS_DIM * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_predicted_measurement);
        
        KALMAN_LOG(KALMAN_LOG_TRACE, "测量值: [%.6f, %.6f], 预测测量值: [%.6f, %.6f], 差异: [%.6f, %.6f]", 
                 measurement[0], measurement[1],
                 predicted_measurement[0], predicted_measurement[1],
                 measurement[0] - predicted_measurement[0], measurement[1] - predicted_measurement[1]);
    }

    return true;
}
