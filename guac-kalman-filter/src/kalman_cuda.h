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

#ifndef GUAC_KALMAN_CUDA_H
#define GUAC_KALMAN_CUDA_H

#include <stdbool.h>
#include <stdint.h>
#include "kalman_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize CUDA resources for Kalman filter calculations
 * 
 * @return
 *     true if initialization was successful, false otherwise
 */
bool cuda_init_kalman(void);

/**
 * Clean up CUDA resources
 */
void cuda_cleanup_kalman(void);

/**
 * Initialize the Kalman filter matrices on the GPU
 * 
 * @param F
 *     The 4x4 state transition matrix
 * 
 * @param H
 *     The 2x4 measurement matrix
 * 
 * @param Q
 *     The 4x4 process noise covariance matrix
 * 
 * @param R
 *     The 2x2 measurement noise covariance matrix
 * 
 * @param P
 *     The 4x4 state estimate covariance matrix
 * 
 * @param state
 *     The 4x1 initial state vector
 * 
 * @return
 *     true if matrix initialization was successful, false otherwise
 */
bool cuda_kalman_init_matrices(const double* F, const double* H, const double* Q, 
                               const double* R, const double* P, const double* state);

/**
 * Update Kalman filter process noise matrix Q
 * 
 * @param Q
 *     The new 4x4 process noise covariance matrix
 * 
 * @return
 *     true if update was successful, false otherwise
 */
bool cuda_kalman_update_Q(const double* Q);

/**
 * Update Kalman filter measurement noise matrix R
 * 
 * @param R
 *     The new 2x2 measurement noise covariance matrix
 * 
 * @return
 *     true if update was successful, false otherwise
 */
bool cuda_kalman_update_R(const double* R);

/**
 * Predict step of the Kalman filter
 * 
 * @param dt
 *     Time delta since last update, used to update state transition matrix
 * 
 * @return
 *     true if prediction was successful, false otherwise
 */
bool cuda_kalman_predict(double dt);

/**
 * Update step of the Kalman filter with a new measurement
 * 
 * @param measurement
 *     The 2x1 measurement vector [x, y]
 * 
 * @param updated_state
 *     Pointer to 4x1 array to store the updated state vector
 * 
 * @return
 *     true if update was successful, false otherwise
 */
bool cuda_kalman_update(const double* measurement, double* updated_state);

/**
 * 根据图层优先级调整卡尔曼滤波器参数
 * 
 * @param filter
 *     卡尔曼滤波器结构体指针
 * 
 * @param priority
 *     图层优先级
 */
void adjust_kalman_params_cuda(guac_kalman_filter* filter, layer_priority_t priority);

#ifdef __cplusplus
}
#endif

#endif /* GUAC_KALMAN_CUDA_H */
