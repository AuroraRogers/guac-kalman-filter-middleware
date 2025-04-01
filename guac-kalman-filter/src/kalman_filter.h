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

#ifndef GUACAMOLE_KALMAN_FILTER_H
#define GUACAMOLE_KALMAN_FILTER_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

#include "guacamole/socket.h"
#include "guacamole/client.h"
#include "guacamole/protocol.h"

/* Define buffer size constants */
#define GUAC_KALMAN_BUFFER_SIZE 8192
#define GUAC_KALMAN_IMAGE_BUFFER_SIZE (1024*1024) /* 1MB for image data */
#define GUAC_KALMAN_MAX_FRAME_COUNT 30            /* Maximum frames to track */

/**
 * Video stream quality metrics
 */
typedef struct guac_video_metrics {
    double psnr;     /* Peak Signal-to-Noise Ratio */
    double ssim;     /* Structural Similarity Index */
    double ms_ssim;  /* Multi-Scale Structural Similarity Index */
    double vmaf;     /* Video Multi-method Assessment Fusion */
    double vqm;      /* Video Quality Metric */
    int frame_size;  /* Frame size in bytes */
    int width;       /* Frame width */
    int height;      /* Frame height */
    double fps;      /* Frames per second */
    uint64_t timestamp; /* Timestamp of the measurement */
} guac_video_metrics;

/**
 * Socket wrapper data structure for Kalman filter
 */
typedef struct guac_socket_kalman_filter_data {
    guac_socket* socket;
    struct guac_kalman_filter* filter;
} guac_socket_kalman_filter_data;

/* 图层优先级定义 */
typedef enum {
    LAYER_PRIORITY_BACKGROUND = 0,
    LAYER_PRIORITY_STATIC = 1,
    LAYER_PRIORITY_DYNAMIC = 2,
    LAYER_PRIORITY_VIDEO = 3,
    LAYER_PRIORITY_UI = 4
} layer_priority_t;

/* 图层依赖关系 */
typedef struct layer_dependency {
    int layer_id;
    int depends_on;
    float weight;
} layer_dependency_t;

/* 更新频率统计 */
typedef struct update_frequency_stats {
    uint64_t last_update;
    uint64_t update_count;
    double avg_interval;
    double variance;
    int pattern_type;  // 0: 随机, 1: 周期性, 2: 突发性
} update_frequency_stats_t;

/* 带宽预测模型 */
typedef struct bandwidth_prediction {
    double current_bandwidth;
    double predicted_bandwidth;
    double confidence;
    uint64_t last_update;
} bandwidth_prediction_t;

/* 场景切换检测 */
typedef struct scene_change_detection {
    double threshold;
    int consecutive_changes;
    uint64_t last_scene_change;
    bool is_scene_changing;
} scene_change_detection_t;

/* 连续帧检测 */
typedef struct continuous_frame_detection {
    int layer_id;                 /* 图层ID */
    uint64_t last_frame_time;     /* 上一帧时间戳 */
    uint64_t frame_count;         /* 连续帧计数 */
    double avg_frame_interval;    /* 平均帧间隔(ms) */
    double frame_interval_variance; /* 帧间隔方差 */
    bool is_video_content;        /* 是否为视频内容 */
    int detection_confidence;     /* 检测置信度(0-100) */
    uint64_t first_detection_time; /* 首次检测为视频的时间 */
    uint64_t last_detection_time;  /* 最近一次检测为视频的时间 */
} continuous_frame_detection_t;

/**
 * The Kalman filter state for RDP protocol
 */
typedef struct guac_kalman_filter {
    int max_layers;
    int max_regions;
    layer_priority_t* layer_priorities;
    layer_dependency_t* layer_dependencies;
    update_frequency_stats_t* frequency_stats;
    bandwidth_prediction_t bandwidth_prediction;
    scene_change_detection_t scene_detection;
    
    /* 连续帧检测相关 */
    continuous_frame_detection_t* continuous_frame_detection; /* 每个图层的连续帧检测 */
    int max_continuous_frames;       /* 判定为视频的最小连续帧数 */
    double max_frame_interval;       /* 判定为视频的最大帧间隔(ms) */
    double min_frame_interval;       /* 判定为视频的最小帧间隔(ms) */
    double frame_interval_threshold;  /* 帧间隔方差阈值 */
    int base_target_bandwidth;      /* 基础目标带宽(bps) */


    // CUDA device pointers
    double* d_F;           // State transition matrix
    double* d_H;           // Measurement matrix
    double* d_Q;          // Process noise covariance
    double* d_R;          // Measurement noise covariance
    double* d_P;          // Error covariance matrix
    double* d_K;          // Kalman gain matrix
    double* d_I;          // Identity matrix
    double* d_state;      // State vector
    double* d_measurement; // Measurement vector
    double* d_temp1;      // Temporary workspace
    double* d_temp2;      // Temporary workspace
    double* d_temp3;      // Temporary workspace
    double* d_temp4;      // Temporary workspace

    // 基本配置
    double sampling_rate; // 采样率

    // 噪声参数
    double process_noise;        // 过程噪声
    double measurement_noise_x;  // X方向测量噪声
    double measurement_noise_y;  // Y方向测量噪声

    /**
     * Process noise covariance matrix (Q)
     */
    double Q[4][4];

    /**
     * Measurement noise covariance matrix (R)
     */
    double R[2][2];

    /**
     * State transition matrix (F)
     */
    double F[4][4];
    
    /**
     * Measurement matrix (H)
     */
    double H[2][4];

    /**
     * State estimate covariance matrix (P)
     */
    double P[4][4];

    /**
     * State vector (x): [position_x, position_y, velocity_x, velocity_y]
     */
    double state[4];

    /**
     * Kalman gain matrix (K)
     */
    double K[4][2];

    /**
     * Identity matrix for calculations
     */
    double I[4][4];

    /**
     * Timestamp of last update
     */
    uint64_t last_timestamp;

    /**
     * Flag to determine if this is the first measurement
     */
    bool first_measurement;

    /**
     * Flag to enable or disable the filtering
     */
    bool enabled;
    
    /**
     * Configuration: process noise (sigma^2_a for acceleration)
     */
    double config_process_noise;
    
    /**
     * Configuration: measurement noise for position X
     */
    double config_measurement_noise_x;
    
    /**
     * Configuration: measurement noise for position Y
     */
    double config_measurement_noise_y;
    
    /**
     * Buffer for instruction parsing
     */
    char parse_buffer[GUAC_KALMAN_BUFFER_SIZE];
    
    /**
     * Current position in the instruction buffer
     */
    size_t buffer_position;
    
    /**
     * 统计选项
     */
    int stats_enabled;     /* 是否启用统计 */
    int stats_fd;          /* 统计文件描述符 */
    char* stats_file;      /* 统计文件路径 */
    
    /**
     * Video metrics history buffer for Kalman filtering
     */
    guac_video_metrics metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT];
    
    /**
     * Current position in metrics history
     */
    int metrics_history_position;
    
    /**
     * Number of frames processed
     */
    int frames_processed;
    
    /**
     * Buffer for image data
     */
    unsigned char image_buffer[GUAC_KALMAN_IMAGE_BUFFER_SIZE];
    
    /**
     * Length of current image data in buffer
     */
    size_t image_buffer_length;
    
    /**
     * Flag to enable video optimization
     */
    bool video_optimization_enabled;
    
    /**
     * Target video quality (0-100)
     * Higher values prioritize quality, lower values prioritize bandwidth
     */
    int target_quality;
    
    /**
     * Target bandwidth in kbps (0 = no limit)
     */
    int target_bandwidth;

    /**
     * 质量评估指标权重
     */
    double psnr_weight;     /* PSNR权重 */
    double ssim_weight;     /* SSIM权重 */
    double ms_ssim_weight;  /* MS-SSIM权重 */
    double vmaf_weight;     /* VMAF权重 */
    double vqm_weight;      /* VQM权重 */

    // 新增性能优化相关成员


    
    // 缓冲区管理
    void* frame_buffer;
    size_t buffer_size;
    int buffer_count;
    
    // 性能统计
    struct {
        uint64_t total_frames;
        uint64_t processed_frames;
        double avg_processing_time;
        double avg_bandwidth_usage;
    } performance_stats;
} guac_kalman_filter;

/**
 * Creates a new Kalman filter-wrapped socket.
 *
 * @param socket The original guac_socket to wrap with the Kalman filter
 * @return A newly allocated guac_socket with Kalman filtering capabilities
 */
guac_socket* guac_socket_kalman_filter_alloc(guac_socket* socket);

/**
 * Initializes the Kalman filter
 *
 * @param filter The Kalman filter to initialize
 */
void guac_kalman_filter_init(guac_kalman_filter* filter);

/**
 * Applies the Kalman filter to position data
 *
 * @param filter The Kalman filter to use
 * @param measured_x The measured x position
 * @param measured_y The measured y position
 * @param timestamp The timestamp of the measurement
 * @param filtered_x Pointer to store the filtered x position
 * @param filtered_y Pointer to store the filtered y position
 */
void guac_kalman_filter_update(guac_kalman_filter* filter, 
                               double measured_x, double measured_y, 
                               uint64_t timestamp, 
                               double* filtered_x, double* filtered_y);

/**
 * Toggles the Kalman filter on or off
 *
 * @param filter The Kalman filter to toggle
 * @param enabled Whether to enable (true) or disable (false) the filter
 */
void guac_kalman_filter_set_enabled(guac_kalman_filter* filter, bool enabled);

/**
 * Configures the Kalman filter noise parameters
 *
 * @param filter The Kalman filter to configure
 * @param process_noise Process noise parameter (sigma^2_a for acceleration)
 * @param measurement_noise_x Measurement noise for x position
 * @param measurement_noise_y Measurement noise for y position
 */
void guac_kalman_filter_configure(guac_kalman_filter* filter, 
                                  double process_noise,
                                  double measurement_noise_x, 
                                  double measurement_noise_y);

/**
 * Enables statistical logging for the Kalman filter
 *
 * @param filter The Kalman filter for which to enable statistics
 * @param filename The filename where statistics should be written
 * @return true if successful, false if an error occurred
 */
bool guac_kalman_filter_enable_stats(guac_kalman_filter* filter, const char* filename);

/**
 * Enables video optimization using the Kalman filter
 *
 * @param filter The Kalman filter to configure
 * @param enabled Whether to enable video optimization
 * @param target_quality Target video quality (0-100, where 100 is highest quality)
 * @param target_bandwidth Target bandwidth in kbps (0 = unlimited)
 */
void guac_kalman_filter_enable_video_optimization(guac_kalman_filter* filter,
                                                 bool enabled,
                                                 int target_quality,
                                                 int target_bandwidth);

/**
 * Processes and optimizes image data using the Kalman filter
 *
 * @param filter The Kalman filter to use
 * @param mimetype The mimetype of the image data
 * @param data The raw image data
 * @param length Length of the image data
 * @param quality Pointer to the quality parameter (will be updated)
 * @return Optimized data or NULL if no optimization was performed
 */
unsigned char* guac_kalman_filter_process_image(guac_kalman_filter* filter,
                                              const char* mimetype,
                                              const unsigned char* data,
                                              size_t length,
                                              int* quality);

/**
 * Calculates video quality metrics for the given frame
 *
 * @param filter The Kalman filter to use
 * @param original_data Original uncompressed frame data
 * @param processed_data Processed/compressed frame data
 * @param width Frame width
 * @param height Frame height
 * @param timestamp Frame timestamp
 * @param metrics Output metrics structure to be filled
 */
void guac_kalman_filter_calculate_metrics(guac_kalman_filter* filter,
                                        const unsigned char* original_data,
                                        const unsigned char* processed_data,
                                        int width, int height,
                                        uint64_t timestamp,
                                        guac_video_metrics* metrics);

/* 新增函数声明 */
// 连续帧检测相关函数
void guac_kalman_filter_configure_continuous_detection(guac_kalman_filter* filter,
                                                     int max_frames,
                                                     double min_interval,
                                                     double max_interval,
                                                     double variance_threshold);

void guac_kalman_filter_reset_continuous_detection(guac_kalman_filter* filter, int layer_id);

bool guac_kalman_filter_is_video_content(guac_kalman_filter* filter, int layer_id);

int guac_kalman_filter_get_video_confidence(guac_kalman_filter* filter, int layer_id);

double guac_kalman_filter_get_estimated_fps(guac_kalman_filter* filter, int layer_id);

// 图层管理
void guac_kalman_filter_set_layer_priority(guac_kalman_filter* filter, int layer_id, layer_priority_t priority);
void guac_kalman_filter_add_layer_dependency(guac_kalman_filter* filter, int layer_id, int depends_on, float weight);
void guac_kalman_filter_update_layer_priorities(guac_kalman_filter* filter);

// 更新频率管理
void guac_kalman_filter_update_frequency_stats(guac_kalman_filter* filter, int region_id);
void guac_kalman_filter_analyze_update_pattern(guac_kalman_filter* filter, int region_id);
void guac_kalman_filter_adjust_sampling_rate(guac_kalman_filter* filter, int region_id);

// 带宽管理
void guac_kalman_filter_update_bandwidth_prediction(guac_kalman_filter* filter);
void guac_kalman_filter_adjust_quality(guac_kalman_filter* filter);
void guac_kalman_filter_optimize_bandwidth_usage(guac_kalman_filter* filter);

// 场景切换检测
void guac_kalman_filter_detect_scene_change(guac_kalman_filter* filter, const unsigned char* frame_data);
void guac_kalman_filter_handle_scene_change(guac_kalman_filter* filter);

// 缓冲区管理
void guac_kalman_filter_init_buffer(guac_kalman_filter* filter, size_t size, int count);
void guac_kalman_filter_update_buffer(guac_kalman_filter* filter, const void* frame_data);
void guac_kalman_filter_cleanup_buffer(guac_kalman_filter* filter);

// 性能统计
void guac_kalman_filter_update_performance_stats(guac_kalman_filter* filter, uint64_t processing_time);
void guac_kalman_filter_print_performance_report(guac_kalman_filter* filter);

/**
 * 更新连续帧检测状态
 * 当收到新的图像帧时调用此函数，用于检测视频内容
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 * @param timestamp 当前时间戳(微秒)
 * @return 如果检测状态发生变化则返回true，否则返回false
 */
bool guac_kalman_filter_update_continuous_detection(guac_kalman_filter* filter, int layer_id, uint64_t timestamp);

/**
 * 应用视频内容优化
 * 根据视频内容检测结果调整卡尔曼滤波器参数
 *
 * @param filter 卡尔曼滤波器
 * @param layer_id 图层ID
 */
void guac_kalman_filter_apply_video_optimization(guac_kalman_filter* filter, int layer_id);

/**
 * 记录视频质量评估指标到CSV文件
 * 记录PSNR、SSIM、MS-SSIM、VMAF和VQM等指标
 *
 * @param filter 卡尔曼滤波器
 * @param metrics 视频质量指标结构体
 * @param filename CSV文件名，如果为NULL则使用默认文件名"video_kalman_metrics.csv"
 * @return 成功返回true，失败返回false
 */
bool guac_kalman_filter_record_video_metrics(guac_kalman_filter* filter, const guac_video_metrics* metrics, const char* filename);

/**
 * 更新滤波器处理后的视频质量指标
 */
void guac_kalman_filter_update_metrics(guac_kalman_filter* filter, const unsigned char* original, 
                                       const unsigned char* processed, int width, int height, int channels);

/**
 * 计算两帧之间的差异
 * 
 * @param frame1 第一帧数据
 * @param frame2 第二帧数据
 * @return 帧间差异值
 */
double calculate_frame_difference(const void* frame1, const void* frame2);

#endif /* GUACAMOLE_KALMAN_FILTER_H */
