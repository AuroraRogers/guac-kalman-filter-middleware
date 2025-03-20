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
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
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
static bool cuda_initialized = false;

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
    if (cuda_initialized)
        return true;
    
    // Allocate device memory for matrices
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
    CUDA_CHECK(cudaMalloc((void**)&d_temp1, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp2, STATE_DIM * STATE_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp3, STATE_DIM * MEAS_DIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp4, MEAS_DIM * MEAS_DIM * sizeof(double)));
    
    cuda_initialized = true;
    return true;
}

/**
 * Clean up CUDA resources
 */
void cuda_cleanup_kalman(void) {
    if (!cuda_initialized)
        return;
    
    // Free device memory
    cudaFree(d_F);
    cudaFree(d_H);
    cudaFree(d_Q);
    cudaFree(d_R);
    cudaFree(d_P);
    cudaFree(d_K);
    cudaFree(d_I);
    cudaFree(d_state);
    cudaFree(d_measurement);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaFree(d_temp3);
    cudaFree(d_temp4);
    
    cuda_initialized = false;
}

/**
 * Initialize the Kalman filter matrices on the GPU
 */
bool cuda_kalman_init_matrices(const double* F, const double* H, const double* Q, 
                             const double* R, const double* P, const double* state) {
    if (!cuda_initialized && !cuda_init_kalman())
        return false;
    
    // Copy matrices to device memory
    CUDA_CHECK(cudaMemcpy(d_F, F, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, H, MEAS_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R, R, MEAS_DIM * MEAS_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P, P, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, state, STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize identity matrix
    double I[STATE_DIM * STATE_DIM] = {0};
    for (int i = 0; i < STATE_DIM; i++) {
        I[i * STATE_DIM + i] = 1.0;
    }
    CUDA_CHECK(cudaMemcpy(d_I, I, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    return true;
}

/**
 * Update Kalman filter process noise matrix Q
 */
bool cuda_kalman_update_Q(const double* Q) {
    if (!cuda_initialized && !cuda_init_kalman())
        return false;
    
    CUDA_CHECK(cudaMemcpy(d_Q, Q, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    return true;
}

/**
 * Update Kalman filter measurement noise matrix R
 */
bool cuda_kalman_update_R(const double* R) {
    if (!cuda_initialized && !cuda_init_kalman())
        return false;
    
    CUDA_CHECK(cudaMemcpy(d_R, R, MEAS_DIM * MEAS_DIM * sizeof(double), cudaMemcpyHostToDevice));
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
    if (!cuda_initialized && !cuda_init_kalman())
        return false;
    
    // Update state transition matrix F for the current time step if dt has changed
    // For our model: F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]
    if (dt > 0) {
        double F[STATE_DIM * STATE_DIM] = {
            1.0, 0.0, dt, 0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        CUDA_CHECK(cudaMemcpy(d_F, F, STATE_DIM * STATE_DIM * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    // Calculate x = F * x
    dim3 blockDim(32, 1);
    dim3 gridDim((STATE_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    // We need to save the current state to use it for calculation
    double temp_state[STATE_DIM];
    CUDA_CHECK(cudaMemcpy(temp_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost));
    
    matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_F, d_state, d_state, STATE_DIM, STATE_DIM);
    
    // Calculate F * P
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_F, d_P, d_temp1, STATE_DIM, STATE_DIM, STATE_DIM);
    
    // Calculate (F * P) * F^T
    // First transpose F
    matrix_transpose_kernel<<<gridDim, blockDim>>>(d_F, d_temp2, STATE_DIM, STATE_DIM);
    
    // Then multiply
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_temp1, d_temp2, d_temp3, STATE_DIM, STATE_DIM, STATE_DIM);
    
    // Add Q: P = (F * P * F^T) + Q
    matrix_add_kernel<<<gridDim, blockDim>>>(d_temp3, d_Q, d_P, STATE_DIM, STATE_DIM);
    
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
    if (!cuda_initialized && !cuda_init_kalman())
        return false;
    
    // Copy measurement to device
    CUDA_CHECK(cudaMemcpy(d_measurement, measurement, MEAS_DIM * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blockDim(16, 16);
    dim3 gridDim;
    
    // Calculate predicted measurement: H * x
    blockDim = dim3(32, 1);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    double predicted_measurement[MEAS_DIM];
    double *d_predicted_measurement;
    CUDA_CHECK(cudaMalloc((void**)&d_predicted_measurement, MEAS_DIM * sizeof(double)));
    
    matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_H, d_state, d_predicted_measurement, MEAS_DIM, STATE_DIM);
    
    // Calculate measurement residual: y = z - H * x
    double *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_y, MEAS_DIM * sizeof(double)));
    
    blockDim = dim3(32, 1);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    matrix_subtract_kernel<<<gridDim, blockDim>>>(d_measurement, d_predicted_measurement, d_y, MEAS_DIM, 1);
    
    // Calculate S = H * P * H^T + R
    // First calculate H * P
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (MEAS_DIM + blockDim.y - 1) / blockDim.y);
    
    double *d_HP;
    CUDA_CHECK(cudaMalloc((void**)&d_HP, MEAS_DIM * STATE_DIM * sizeof(double)));
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_H, d_P, d_HP, MEAS_DIM, STATE_DIM, STATE_DIM);
    
    // Transpose H
    double *d_HT;
    CUDA_CHECK(cudaMalloc((void**)&d_HT, STATE_DIM * MEAS_DIM * sizeof(double)));
    
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (MEAS_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_transpose_kernel<<<gridDim, blockDim>>>(d_H, d_HT, MEAS_DIM, STATE_DIM);
    
    // Calculate (H * P) * H^T
    double *d_S;
    CUDA_CHECK(cudaMalloc((void**)&d_S, MEAS_DIM * MEAS_DIM * sizeof(double)));
    
    blockDim = dim3(16, 16);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, (MEAS_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_HP, d_HT, d_S, MEAS_DIM, STATE_DIM, MEAS_DIM);
    
    // Add R: S = (H * P * H^T) + R
    matrix_add_kernel<<<gridDim, blockDim>>>(d_S, d_R, d_S, MEAS_DIM, MEAS_DIM);
    
    // Calculate S^-1 (inverse of S)
    double *d_S_inv;
    CUDA_CHECK(cudaMalloc((void**)&d_S_inv, MEAS_DIM * MEAS_DIM * sizeof(double)));
    
    blockDim = dim3(1, 1);
    gridDim = dim3(1, 1);
    
    matrix_invert_kernel<<<gridDim, blockDim>>>(d_S, d_S_inv, MEAS_DIM);
    
    // Calculate P * H^T
    double *d_PHT;
    CUDA_CHECK(cudaMalloc((void**)&d_PHT, STATE_DIM * MEAS_DIM * sizeof(double)));
    
    blockDim = dim3(16, 16);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_P, d_HT, d_PHT, STATE_DIM, STATE_DIM, MEAS_DIM);
    
    // Calculate K = P * H^T * S^-1
    blockDim = dim3(16, 16);
    gridDim = dim3((MEAS_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_PHT, d_S_inv, d_K, STATE_DIM, MEAS_DIM, MEAS_DIM);
    
    // Calculate K * y
    double *d_Ky;
    CUDA_CHECK(cudaMalloc((void**)&d_Ky, STATE_DIM * sizeof(double)));
    
    blockDim = dim3(32, 1);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    matrix_vector_multiply_kernel<<<gridDim, blockDim>>>(d_K, d_y, d_Ky, STATE_DIM, MEAS_DIM);
    
    // Update state: x = x + K * y
    blockDim = dim3(32, 1);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, 1);
    
    matrix_add_kernel<<<gridDim, blockDim>>>(d_state, d_Ky, d_state, STATE_DIM, 1);
    
    // Calculate K * H
    double *d_KH;
    CUDA_CHECK(cudaMalloc((void**)&d_KH, STATE_DIM * STATE_DIM * sizeof(double)));
    
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_K, d_H, d_KH, STATE_DIM, MEAS_DIM, STATE_DIM);
    
    // Calculate I - K * H
    double *d_IKH;
    CUDA_CHECK(cudaMalloc((void**)&d_IKH, STATE_DIM * STATE_DIM * sizeof(double)));
    
    blockDim = dim3(16, 16);
    gridDim = dim3((STATE_DIM + blockDim.x - 1) / blockDim.x, (STATE_DIM + blockDim.y - 1) / blockDim.y);
    
    matrix_subtract_kernel<<<gridDim, blockDim>>>(d_I, d_KH, d_IKH, STATE_DIM, STATE_DIM);

    // Calculate P = (I - K * H) * P
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_IKH, d_P, d_P, STATE_DIM, STATE_DIM, STATE_DIM);

    // Free temporary memory
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
    CUDA_CHECK(cudaMemcpy(updated_state, d_state, STATE_DIM * sizeof(double), cudaMemcpyDeviceToHost));

    return true;
}
