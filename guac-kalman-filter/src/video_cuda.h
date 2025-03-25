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

#ifndef GUAC_VIDEO_CUDA_H
#define GUAC_VIDEO_CUDA_H

#include <stdbool.h>
#include <stdint.h>
#include "kalman_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 视频质量指标结构体
 */
typedef struct {
    double psnr;     // 峰值信噪比
    double ssim;     // 结构相似性
    double vmaf;     // 视频多方法评估融合
    int width;       // 视频宽度
    int height;      // 视频高度
    int channels;    // 颜色通道数
} video_quality_metrics_t;

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
bool cuda_init_video(int max_width, int max_height);

/**
 * 清理视频处理的CUDA资源
 */
void cuda_cleanup_video(void);

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
                             int channels, int quality, unsigned char* output_data);

/**
 * 计算视频质量指标
 * 
 * @param original_data
 *     原始视频帧数据
 * 
 * @param processed_data
 *     处理后的视频帧数据
 * 
 * @param width
 *     帧宽度
 * 
 * @param height
 *     帧高度
 * 
 * @param channels
 *     颜色通道数
 * 
 * @param metrics
 *     输出质量指标结构体
 * 
 * @return
 *     计算成功返回true，否则返回false
 */
bool cuda_calculate_video_metrics(const unsigned char* original_data, 
                                 const unsigned char* processed_data,
                                 int width, int height, int channels,
                                 video_quality_metrics_t* metrics);

/**
 * 使用CUDA卡尔曼滤波器处理视频指令
 * 
 * @param filter
 *     卡尔曼滤波器结构体
 * 
 * @param stream_id
 *     视频流ID
 * 
 * @param layer_id
 *     图层ID
 * 
 * @param mimetype
 *     视频MIME类型
 * 
 * @return
 *     处理成功返回true，否则返回false
 */
bool cuda_process_video_instruction(guac_kalman_filter* filter, 
                                   int stream_id, int layer_id, 
                                   const char* mimetype);

#ifdef __cplusplus
}
#endif

#endif /* GUAC_VIDEO_CUDA_H */