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
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "kalman_filter.h"

/**
 * 配置连续帧检测参数
 *
 * @param filter 卡尔曼滤波器
 * @param max_frames 判定为视频的最小连续帧数
 * @param min_interval 判定为视频的最小帧间隔(ms)
 * @param max_interval 判定为视频的最大帧间隔(ms)
 * @param variance_threshold 帧间隔方差阈值
 */
void guac_kalman_filter_configure_continuous_detection(guac_kalman_filter* filter,
                                                     int max_frames,
                                                     double min_interval,
                                                     double max_interval,
                                                     double variance_threshold) {
    if (!filter) {
        return;
    }
    
    // 设置连续帧检测参数
    filter->max_continuous_frames = max_frames;
    filter->min_frame_interval = min_interval;
    filter->max_frame_interval = max_interval;
    filter->frame_interval_threshold = variance_threshold;
    
    // 记录配置信息
    fprintf(stderr, "[连续帧检测] 配置参数: 最小帧数=%d, 帧间隔范围=%.2f-%.2f ms, 方差阈值=%.2f\n",
           max_frames, min_interval, max_interval, variance_threshold);
}

/**
 * 重置连续帧检测状态
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 */
void guac_kalman_filter_reset_continuous_detection(guac_kalman_filter* filter, int layer_id) {
    if (!filter || !filter->continuous_frame_detection || layer_id >= filter->max_layers) {
        return;
    }
    
    continuous_frame_detection_t* frame_detection = &filter->continuous_frame_detection[layer_id];
    
    // 重置检测状态
    frame_detection->layer_id = layer_id;
    frame_detection->last_frame_time = 0;
    frame_detection->frame_count = 0;
    frame_detection->avg_frame_interval = 0;
    frame_detection->frame_interval_variance = 0;
    frame_detection->is_video_content = false;
    frame_detection->detection_confidence = 0;
    frame_detection->first_detection_time = 0;
    frame_detection->last_detection_time = 0;
    
    fprintf(stderr, "[连续帧检测] 重置图层 %d 的检测状态\n", layer_id);
}

/**
 * 获取图层的视频内容检测状态
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 * @return 如果图层被检测为视频内容则返回true，否则返回false
 */
bool guac_kalman_filter_is_video_content(guac_kalman_filter* filter, int layer_id) {
    if (!filter || !filter->continuous_frame_detection || layer_id >= filter->max_layers) {
        return false;
    }
    
    return filter->continuous_frame_detection[layer_id].is_video_content;
}

/**
 * 获取图层的视频内容检测置信度
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 * @return 视频内容检测置信度(0-100)
 */
int guac_kalman_filter_get_video_confidence(guac_kalman_filter* filter, int layer_id) {
    if (!filter || !filter->continuous_frame_detection || layer_id >= filter->max_layers) {
        return 0;
    }
    
    return filter->continuous_frame_detection[layer_id].detection_confidence;
}

/**
 * 获取图层的估计帧率
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 * @return 估计帧率(fps)，如果无法计算则返回0
 */
double guac_kalman_filter_get_estimated_fps(guac_kalman_filter* filter, int layer_id) {
    if (!filter || !filter->continuous_frame_detection || layer_id >= filter->max_layers) {
        return 0.0;
    }
    
    continuous_frame_detection_t* frame_detection = &filter->continuous_frame_detection[layer_id];
    
    if (frame_detection->avg_frame_interval > 0) {
        return 1000.0 / frame_detection->avg_frame_interval;
    }
    
    return 0.0;
}

/**
 * 更新连续帧检测状态
 * 当收到新的图像帧时调用此函数，用于检测视频内容
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 * @param timestamp 当前时间戳(微秒)
 * @return 如果检测状态发生变化则返回true，否则返回false
 */
bool guac_kalman_filter_update_continuous_detection(guac_kalman_filter* filter, int layer_id, uint64_t timestamp) {
    if (!filter || !filter->continuous_frame_detection || layer_id >= filter->max_layers) {
        return false;
    }
    
    continuous_frame_detection_t* frame_detection = &filter->continuous_frame_detection[layer_id];
    bool status_changed = false;
    
    // 如果是第一帧，初始化时间戳
    if (frame_detection->last_frame_time == 0) {
        frame_detection->last_frame_time = timestamp;
        frame_detection->frame_count = 1;
        return false;
    }
    
    // 计算帧间隔(毫秒)
    double frame_interval = (timestamp - frame_detection->last_frame_time) / 1000.0;
    
    // 检查帧间隔是否在合理范围内
    if (frame_interval >= filter->min_frame_interval && 
        frame_interval <= filter->max_frame_interval) {
        
        // 更新帧计数和统计信息
        frame_detection->frame_count++;
        
        // 更新平均帧间隔和方差
        double old_avg = frame_detection->avg_frame_interval;
        frame_detection->avg_frame_interval = 
            (old_avg * (frame_detection->frame_count - 1) + frame_interval) / 
            frame_detection->frame_count;
        
        // 更新方差 (使用增量公式计算方差)
        if (frame_detection->frame_count > 1) {
            double delta = frame_interval - old_avg;
            double delta2 = frame_interval - frame_detection->avg_frame_interval;
            frame_detection->frame_interval_variance = 
                ((frame_detection->frame_count - 1) * frame_detection->frame_interval_variance + 
                 delta * delta2) / frame_detection->frame_count;
        }
        
        // 检查是否满足视频内容的条件
        if (frame_detection->frame_count >= filter->max_continuous_frames && 
            frame_detection->frame_interval_variance <= filter->frame_interval_threshold) {
            
            // 如果之前不是视频内容，现在变成了视频内容
            if (!frame_detection->is_video_content) {
                frame_detection->is_video_content = true;
                frame_detection->first_detection_time = timestamp;
                status_changed = true;
                
                // 计算初始置信度 (基于帧数和方差)
                double frame_ratio = (double)frame_detection->frame_count / filter->max_continuous_frames;
                double variance_ratio = 1.0 - (frame_detection->frame_interval_variance / filter->frame_interval_threshold);
                frame_detection->detection_confidence = (int)((frame_ratio * 0.6 + variance_ratio * 0.4) * 100);
                
                if (frame_detection->detection_confidence > 100) {
                    frame_detection->detection_confidence = 100;
                }
                
                fprintf(stderr, "[连续帧检测] 图层 %d 被检测为视频内容! 帧数=%lu, 平均间隔=%.2f ms, 方差=%.2f, 置信度=%d%%\n", 
                       layer_id, frame_detection->frame_count, 
                       frame_detection->avg_frame_interval,
                       frame_detection->frame_interval_variance,
                       frame_detection->detection_confidence);
            } else {
                // 已经是视频内容，更新置信度
                double frame_ratio = (double)frame_detection->frame_count / filter->max_continuous_frames;
                if (frame_ratio > 2.0) frame_ratio = 2.0; // 限制最大值
                
                double variance_ratio = 1.0 - (frame_detection->frame_interval_variance / filter->frame_interval_threshold);
                if (variance_ratio < 0.0) variance_ratio = 0.0;
                
                int new_confidence = (int)((frame_ratio * 0.6 + variance_ratio * 0.4) * 100);
                if (new_confidence > 100) new_confidence = 100;
                
                // 如果置信度变化超过10%，认为状态有变化
                if (abs(new_confidence - frame_detection->detection_confidence) > 10) {
                    status_changed = true;
                    frame_detection->detection_confidence = new_confidence;
                    
                    fprintf(stderr, "[连续帧检测] 图层 %d 视频内容置信度更新: %d%% (帧数=%lu, 平均间隔=%.2f ms, 方差=%.2f)\n", 
                           layer_id, frame_detection->detection_confidence, 
                           frame_detection->frame_count, 
                           frame_detection->avg_frame_interval,
                           frame_detection->frame_interval_variance);
                }
            }
        }
    } else {
        // 帧间隔不在合理范围内，重置计数
        // 但如果已经确认是视频内容，给予一定的容忍度
        if (frame_detection->is_video_content) {
            // 降低置信度
            frame_detection->detection_confidence -= 5;
            if (frame_detection->detection_confidence < 0) {
                frame_detection->detection_confidence = 0;
            }
            
            // 如果置信度降为0，不再认为是视频内容
            if (frame_detection->detection_confidence == 0) {
                frame_detection->is_video_content = false;
                status_changed = true;
                fprintf(stderr, "[连续帧检测] 图层 %d 不再被检测为视频内容 (帧间隔=%.2f ms 超出范围)\n", 
                       layer_id, frame_interval);
            }
        } else {
            // 不是视频内容，直接重置
            frame_detection->frame_count = 1;
            frame_detection->avg_frame_interval = 0;
            frame_detection->frame_interval_variance = 0;
        }
    }
    
    // 更新最后一帧时间
    frame_detection->last_frame_time = timestamp;
    if (frame_detection->is_video_content) {
        frame_detection->last_detection_time = timestamp;
    }
    
    return status_changed;
}

/**
 * 应用视频内容优化
 * 根据视频内容检测结果调整卡尔曼滤波器参数
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 */
void guac_kalman_filter_apply_video_optimization(guac_kalman_filter* filter, int layer_id) {
    if (!filter || !filter->continuous_frame_detection || layer_id >= filter->max_layers) {
        return;
    }
    
    continuous_frame_detection_t* frame_detection = &filter->continuous_frame_detection[layer_id];
    
    if (frame_detection->is_video_content) {
        // 为视频内容优化卡尔曼滤波器参数
        double confidence_factor = frame_detection->detection_confidence / 100.0;
        
        // 根据置信度调整参数
        filter->config_process_noise = 0.01 * (1.0 + confidence_factor);
        filter->config_measurement_noise_x = 0.05 * (1.0 - confidence_factor * 0.5);
        filter->config_measurement_noise_y = 0.05 * (1.0 - confidence_factor * 0.5);
        
        // 设置图层优先级为视频
        filter->layer_priorities[layer_id] = LAYER_PRIORITY_VIDEO;
        
        // 调整带宽预测
        if (confidence_factor > 0.8) {
            // 高置信度视频，预留更多带宽
            filter->target_bandwidth = (int)(filter->base_target_bandwidth * 1.2);
            filter->target_quality = 85;
        } else {
            // 中等置信度视频
            filter->target_bandwidth = filter->base_target_bandwidth;
            filter->target_quality = 75;
        }
        
        fprintf(stderr, "[视频优化] 应用视频优化参数到图层 %d (置信度=%d%%)\n", 
               layer_id, frame_detection->detection_confidence);
    } else {
        // 恢复默认参数
        filter->config_process_noise = 0.01;
        filter->config_measurement_noise_x = 0.1;
        filter->config_measurement_noise_y = 0.1;
        
        // 根据更新频率设置图层优先级
        if (filter->frequency_stats[layer_id].avg_interval < 500000) { // 小于500ms
            filter->layer_priorities[layer_id] = LAYER_PRIORITY_DYNAMIC;
        } else {
            filter->layer_priorities[layer_id] = LAYER_PRIORITY_STATIC;
        }
        
        // 恢复默认带宽设置
        filter->target_bandwidth = filter->base_target_bandwidth;
        filter->target_quality = 80;
    }
}