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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// 获取当前时间戳（微秒）
static uint64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t) tv.tv_sec * 1000000 + tv.tv_usec;
}

#include "kalman_filter.h"
#include "kalman_cuda.h"
#include "video_cuda.h"

// 外部声明
extern bool cuda_init_video(int max_width, int max_height);
extern void cuda_cleanup_video(void);
extern bool cuda_process_video_frame(const unsigned char* frame_data, int width, int height, 
                                   int channels, int quality, unsigned char* output_data);

// 视频帧缓存
static unsigned char* video_frame_buffer = NULL;
static int video_frame_buffer_size = 0;
static int video_max_width = 1920;  // 默认最大宽度
static int video_max_height = 1080; // 默认最大高度

// 视频流信息映射表
typedef struct {
    int stream_id;
    int layer_id;
    char mimetype[32];
    int width;
    int height;
    int quality;
    uint64_t last_frame_time;
    bool active;
} video_stream_info_t;

#define MAX_VIDEO_STREAMS 16
static video_stream_info_t video_streams[MAX_VIDEO_STREAMS];
static bool video_processing_initialized = false;

/**
 * 初始化视频处理
 */
static bool init_video_processing(void) {
    if (video_processing_initialized) {
        return true;
    }
    
    // 初始化视频流信息
    memset(video_streams, 0, sizeof(video_streams));
    for (int i = 0; i < MAX_VIDEO_STREAMS; i++) {
        video_streams[i].active = false;
    }
    
    // 初始化CUDA视频处理资源
    if (!cuda_init_video(video_max_width, video_max_height)) {
        fprintf(stderr, "Failed to initialize CUDA video resources\n");
        return false;
    }
    
    // 分配视频帧缓存
    video_frame_buffer_size = video_max_width * video_max_height * 4; // RGBA格式
    video_frame_buffer = (unsigned char*)malloc(video_frame_buffer_size);
    if (!video_frame_buffer) {
        fprintf(stderr, "Failed to allocate video frame buffer\n");
        cuda_cleanup_video();
        return false;
    }
    
    video_processing_initialized = true;
    return true;
}

/**
 * 清理视频处理资源
 */
static void cleanup_video_processing(void) {
    if (!video_processing_initialized) {
        return;
    }
    
    // 释放视频帧缓存
    if (video_frame_buffer) {
        free(video_frame_buffer);
        video_frame_buffer = NULL;
    }
    
    // 清理CUDA视频处理资源
    cuda_cleanup_video();
    
    video_processing_initialized = false;
}

/**
 * 计算视频质量指标
 */
bool cuda_calculate_video_metrics(const unsigned char* original_data, 
                                 const unsigned char* processed_data,
                                 int width, int height, int channels,
                                 video_quality_metrics_t* metrics) {
    if (!video_processing_initialized) {
        if (!init_video_processing()) {
            return false;
        }
    }
    
    if (!original_data || !processed_data || !metrics) {
        return false;
    }
    
    // 设置基本信息
    metrics->width = width;
    metrics->height = height;
    metrics->channels = channels;
    
    // 计算PSNR (Peak Signal-to-Noise Ratio)
    double mse = 0.0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                double diff = (double)original_data[idx] - (double)processed_data[idx];
                mse += diff * diff;
            }
        }
    }
    
    mse /= (width * height * channels);
    if (mse > 0.0) {
        metrics->psnr = 10.0 * log10(255.0 * 255.0 / mse);
    } else {
        metrics->psnr = 100.0; // 完美匹配
    }
    
    // 简化的SSIM计算 (Structural Similarity Index)
    // 注意：完整的SSIM计算需要考虑局部窗口，这里使用简化版本
    double mean_orig = 0.0, mean_proc = 0.0;
    double var_orig = 0.0, var_proc = 0.0, covar = 0.0;
    
    // 计算均值
    for (int i = 0; i < width * height * channels; i++) {
        mean_orig += original_data[i];
        mean_proc += processed_data[i];
    }
    
    mean_orig /= (width * height * channels);
    mean_proc /= (width * height * channels);
    
    // 计算方差和协方差
    for (int i = 0; i < width * height * channels; i++) {
        double diff_orig = original_data[i] - mean_orig;
        double diff_proc = processed_data[i] - mean_proc;
        var_orig += diff_orig * diff_orig;
        var_proc += diff_proc * diff_proc;
        covar += diff_orig * diff_proc;
    }
    
    var_orig /= (width * height * channels - 1);
    var_proc /= (width * height * channels - 1);
    covar /= (width * height * channels - 1);
    
    // SSIM常数
    double C1 = (0.01 * 255.0) * (0.01 * 255.0);
    double C2 = (0.03 * 255.0) * (0.03 * 255.0);
    
    // 计算SSIM
    double numerator = (2.0 * mean_orig * mean_proc + C1) * (2.0 * covar + C2);
    double denominator = (mean_orig * mean_orig + mean_proc * mean_proc + C1) * (var_orig + var_proc + C2);
    metrics->ssim = numerator / denominator;
    
    // VMAF估计 (Video Multi-method Assessment Fusion)
    // 注意：真正的VMAF需要更复杂的计算，这里使用简化的估计
    metrics->vmaf = 0.8 * metrics->psnr / 60.0 + 0.2 * metrics->ssim;
    metrics->vmaf = metrics->vmaf > 1.0 ? 1.0 : metrics->vmaf;
    metrics->vmaf *= 100.0; // 转换为0-100范围
    
    return true;
}

/**
 * 使用CUDA卡尔曼滤波器处理视频指令
 */
bool cuda_process_video_instruction(guac_kalman_filter* filter, 
                                   int stream_id, int layer_id, 
                                   const char* mimetype) {
    if (!filter) {
        return false;
    }
    
    // 初始化视频处理（如果尚未初始化）
    if (!video_processing_initialized) {
        if (!init_video_processing()) {
            return false;
        }
    }
    
    // 查找或创建视频流信息
    int stream_idx = -1;
    for (int i = 0; i < MAX_VIDEO_STREAMS; i++) {
        if (video_streams[i].active && video_streams[i].stream_id == stream_id) {
            stream_idx = i;
            break;
        }
    }
    
    if (stream_idx == -1) {
        // 创建新的视频流信息
        for (int i = 0; i < MAX_VIDEO_STREAMS; i++) {
            if (!video_streams[i].active) {
                stream_idx = i;
                video_streams[i].active = true;
                video_streams[i].stream_id = stream_id;
                video_streams[i].layer_id = layer_id;
                strncpy(video_streams[i].mimetype, mimetype, sizeof(video_streams[i].mimetype) - 1);
                video_streams[i].width = 640;  // 默认值，将在实际帧到达时更新
                video_streams[i].height = 480; // 默认值，将在实际帧到达时更新
                video_streams[i].quality = filter->target_quality;
                video_streams[i].last_frame_time = 0;
                break;
            }
        }
        
        if (stream_idx == -1) {
            fprintf(stderr, "Error: Maximum number of video streams reached\n");
            return false;
        }
    } else {
        // 更新现有流信息
        video_streams[stream_idx].layer_id = layer_id;
        strncpy(video_streams[stream_idx].mimetype, mimetype, sizeof(video_streams[stream_idx].mimetype) - 1);
    }
    
    // 设置图层优先级为视频优先级
    if (layer_id < filter->max_layers) {
        filter->layer_priorities[layer_id] = LAYER_PRIORITY_VIDEO;
    }
    
    // 根据带宽预测调整视频质量
    if (filter->video_optimization_enabled) {
        // 获取当前带宽预测
        double predicted_bandwidth = filter->bandwidth_prediction.predicted_bandwidth;
        int target_quality = filter->target_quality;
        
        // 如果预测带宽低于目标带宽的80%，降低质量
        if (filter->target_bandwidth > 0 && 
            predicted_bandwidth < filter->target_bandwidth * 0.8) {
            target_quality = filter->target_quality - 10;
            if (target_quality < 30) target_quality = 30; // 最低质量限制
            
            fprintf(stderr, "[CUDA Video] 带宽不足，降低视频质量: %d -> %d\n", 
                   filter->target_quality, target_quality);
        }
        // 如果预测带宽高于目标带宽的120%，提高质量
        else if (filter->target_bandwidth > 0 && 
                 predicted_bandwidth > filter->target_bandwidth * 1.2) {
            target_quality = filter->target_quality + 5;
            if (target_quality > 95) target_quality = 95; // 最高质量限制
            
            fprintf(stderr, "[CUDA Video] 带宽充足，提高视频质量: %d -> %d\n", 
                   filter->target_quality, target_quality);
        }
        
        // 应用新的质量设置
        if (target_quality != filter->target_quality) {
            filter->target_quality = target_quality;
            video_streams[stream_idx].quality = target_quality;
            fprintf(stderr, "[CUDA Video] 更新视频质量目标: %d\n", filter->target_quality);
        }
    }
    
    return true;
}

/**
 * 处理视频帧数据
 */
bool cuda_process_video_frame_data(guac_kalman_filter* filter, int stream_id, 
                                  const unsigned char* frame_data, int width, int height, 
                                  int channels) {
    if (!filter || !frame_data) {
        return false;
    }
    
    // 初始化视频处理（如果尚未初始化）
    if (!video_processing_initialized) {
        if (!init_video_processing()) {
            return false;
        }
    }
    
    // 查找视频流信息
    int stream_idx = -1;
    for (int i = 0; i < MAX_VIDEO_STREAMS; i++) {
        if (video_streams[i].active && video_streams[i].stream_id == stream_id) {
            stream_idx = i;
            break;
        }
    }
    
    if (stream_idx == -1) {
        fprintf(stderr, "Error: Video stream %d not found\n", stream_id);
        return false;
    }
    
    // 更新视频流信息
    video_streams[stream_idx].width = width;
    video_streams[stream_idx].height = height;
    video_streams[stream_idx].last_frame_time = get_timestamp_us();
    
    // 检查帧大小
    int frame_size = width * height * channels;
    if (frame_size > video_frame_buffer_size) {
        fprintf(stderr, "Error: Frame size exceeds buffer size: %d > %d\n", 
               frame_size, video_frame_buffer_size);
        return false;
    }
    
    // 处理视频帧
    if (!cuda_process_video_frame(frame_data, width, height, channels, 
                                video_streams[stream_idx].quality, video_frame_buffer)) {
        fprintf(stderr, "Error: Failed to process video frame\n");
        return false;
    }
    
    // 计算视频质量指标
    video_quality_metrics_t metrics;
    if (cuda_calculate_video_metrics(frame_data, video_frame_buffer, 
                                   width, height, channels, &metrics)) {
        // 记录质量指标
        fprintf(stderr, "[CUDA Video] 视频质量指标 - PSNR: %.2f dB, SSIM: %.4f, VMAF: %.2f\n", 
               metrics.psnr, metrics.ssim, metrics.vmaf);
        
        // 更新滤波器的质量指标历史
        if (filter->metrics_history_position < GUAC_KALMAN_MAX_FRAME_COUNT) {
            filter->metrics_history[filter->metrics_history_position].psnr = metrics.psnr;
            filter->metrics_history[filter->metrics_history_position].ssim = metrics.ssim;
            filter->metrics_history[filter->metrics_history_position].vmaf = metrics.vmaf;
            filter->metrics_history[filter->metrics_history_position].frame_size = frame_size;
            filter->metrics_history[filter->metrics_history_position].width = width;
            filter->metrics_history[filter->metrics_history_position].height = height;
            filter->metrics_history[filter->metrics_history_position].timestamp = video_streams[stream_idx].last_frame_time;
            
            // 计算帧率
            if (filter->metrics_history_position > 0) {
                uint64_t prev_time = filter->metrics_history[filter->metrics_history_position - 1].timestamp;
                uint64_t curr_time = video_streams[stream_idx].last_frame_time;
                double time_diff = (curr_time - prev_time) / 1000000.0; // 转换为秒
                
                if (time_diff > 0) {
                    filter->metrics_history[filter->metrics_history_position].fps = 1.0 / time_diff;
                }
            }
            
            filter->metrics_history_position++;
        } else {
            // 循环使用历史记录
            for (int i = 0; i < GUAC_KALMAN_MAX_FRAME_COUNT - 1; i++) {
                filter->metrics_history[i] = filter->metrics_history[i + 1];
            }
            
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].psnr = metrics.psnr;
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].ssim = metrics.ssim;
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].vmaf = metrics.vmaf;
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].frame_size = frame_size;
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].width = width;
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].height = height;
            filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].timestamp = video_streams[stream_idx].last_frame_time;
            
            // 计算帧率
            uint64_t prev_time = filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 2].timestamp;
            uint64_t curr_time = video_streams[stream_idx].last_frame_time;
            double time_diff = (curr_time - prev_time) / 1000000.0; // 转换为秒
            
            if (time_diff > 0) {
                filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].fps = 1.0 / time_diff;
            }
        }
        
        filter->frames_processed++;
    }
    
    return true;
}