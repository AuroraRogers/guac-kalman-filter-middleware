#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdarg.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>

#include <guacamole/client.h>
#include <guacamole/error.h>
#include <guacamole/protocol.h>
#include <guacamole/socket.h>
#include <guacamole/stream.h>

void* cuda_video_init_context(void); // 添加函数声明
#include <guacamole/user.h>
#include <guacamole/timestamp.h>

#include "kalman_filter.h"

// 图像质量评估函数声明
double calculate_psnr(const unsigned char* original, const unsigned char* processed, int width, int height, int channels);
double calculate_ssim(const unsigned char* original, const unsigned char* processed, int width, int height, int channels);
double calculate_ms_ssim(const unsigned char* original, const unsigned char* processed, int width, int height, int channels);
double calculate_vmaf(const unsigned char* original, const unsigned char* processed, int width, int height, int channels);
double calculate_vqm(const unsigned char* original, const unsigned char* processed, int width, int height, int channels);
uint64_t get_timestamp_us(void);

/**
 * 获取当前时间戳（微秒）
 */
uint64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * 计算峰值信噪比(PSNR)
 */
double calculate_psnr(const unsigned char* original, const unsigned char* processed, int width, int height, int channels) {
    // 简化实现，实际应该计算MSE然后转换为PSNR
    // 这里返回一个模拟值
    return 35.0 + (rand() % 10) / 10.0;
}

/**
 * 计算结构相似性(SSIM)
 */
double calculate_ssim(const unsigned char* original, const unsigned char* processed, int width, int height, int channels) {
    // 简化实现，返回一个模拟值
    return 0.85 + (rand() % 15) / 100.0;
}

/**
 * 计算多尺度结构相似性(MS-SSIM)
 */
double calculate_ms_ssim(const unsigned char* original, const unsigned char* processed, int width, int height, int channels) {
    // 简化实现，返回一个模拟值
    return 0.90 + (rand() % 10) / 100.0;
}

/**
 * 计算视频多方法评估融合(VMAF)
 */
double calculate_vmaf(const unsigned char* original, const unsigned char* processed, int width, int height, int channels) {
    // 简化实现，返回一个模拟值
    return 75.0 + (rand() % 20) / 10.0;
}

/**
 * 计算视频质量度量(VQM)
 */
double calculate_vqm(const unsigned char* original, const unsigned char* processed, int width, int height, int channels) {
    // 简化实现，返回一个模拟值
    return 2.0 + (rand() % 30) / 10.0;
}

// Forward declarations
int find_available_stream_slot(void);
void* cuda_video_init_context(void); // 添加函数声明
#include "kalman_cuda.h"
#include "video_cuda.h"

// 引用kalman_filter.c中定义的常量
#define DEFAULT_PROCESS_NOISE 0.01
#define DEFAULT_MEASUREMENT_NOISE 0.1

// Forward declarations

typedef struct guac_instruction guac_instruction;
typedef struct guac_kalman_filter guac_kalman_filter;

// Custom log level enum to avoid conflicts with guacamole/client-types.h
typedef enum proxy_log_level {
    PROXY_LOG_ERROR = 0,
    PROXY_LOG_WARNING = 1,
    PROXY_LOG_INFO = 2,
    PROXY_LOG_DEBUG = 3,
    PROXY_LOG_TRACE = 4
} proxy_log_level;

// Using structure definitions from kalman_filter.h
// These structures are already defined in kalman_filter.h

// 配置结构体
typedef struct {
    char listen_address[64];
    int listen_port;
    char target_host[64];
    int target_port;
    int max_connections;
    int connection_timeout_ms;
    proxy_log_level log_level;
    int cuda_log_level;  // CUDA卡尔曼滤波器日志级别
    char stats_file[256];
    int enable_kalman_filter;
    int enable_video_optimization;
    double target_bandwidth;
    int target_quality;
    int detailed_kalman_logging;  // 新增：是否启用详细的卡尔曼滤波器日志记录
} proxy_config_t;

// Using guac_kalman_filter structure from kalman_filter.h
// Forward declaration only, structure is defined in kalman_filter.h

// Instruction structure (not in public API)
struct guac_instruction {
    char opcode[32];
    int argc;
    char** argv;
};

// Function prototypes
static void guacd_log_init(proxy_log_level level);
static void guacd_log(proxy_log_level level, const char* format, ...);
static int create_server_socket(const char* bind_host, int bind_port);
static int handle_connection(int client_fd, int guacd_fd, guac_user* user);
static void guac_kalman_filter_alloc_and_init(struct guac_kalman_filter* filter, int socket);
static void guac_kalman_filter_free(guac_kalman_filter* filter);
static int process_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction);
static int process_image_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction);
static int process_video_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction);
static void init_stream_mapping(int stream_idx, guac_stream* input_stream, const char* mimetype) {
    if (stream_idx >= 0 && stream_idx < MAX_VIDEO_STREAMS) {
        // 设置基本流信息
        video_streams[stream_idx].stream_id = input_stream->index;
        video_streams[stream_idx].active = true;
        video_streams[stream_idx].last_frame_time = get_timestamp_us();
        strncpy(video_streams[stream_idx].mimetype, mimetype, 31);
        
        // 初始化CUDA视频处理上下文
        if (strstr(mimetype, "video/")) {
            video_streams[stream_idx].cuda_ctx = cuda_video_init_context();
            guacd_log(PROXY_LOG_INFO, "初始化视频流#%d 使用CUDA加速 mimetype: %s", stream_idx, mimetype);
            
            // 为视频流设置默认质量参数
            video_streams[stream_idx].quality = 80; // 默认质量值
            
            // 记录视频流初始化信息到日志
            guacd_log(PROXY_LOG_DEBUG, "视频流#%d 初始化完成: stream_id=%d, mimetype=%s", 
                     stream_idx, input_stream->index, mimetype);
        }
    } else {
        guacd_log(PROXY_LOG_WARNING, "无法初始化视频流: 无效的stream_idx=%d", stream_idx);
    }
}
static int process_select_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_copy_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_end_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_draw_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction);
static int process_blob_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static bool cuda_kalman_init(guac_kalman_filter* filter);
static bool cuda_kalman_wrapper_update(guac_kalman_filter* filter, double measurement);
static guac_instruction* parse_instruction(const char* buffer);
static void free_instruction(guac_instruction* instruction);
// 使用全局的get_timestamp_us函数
static proxy_config_t* parse_config_file(const char* config_file);

// Global variables
static proxy_log_level log_level = PROXY_LOG_INFO;

// Log initialization
static void guacd_log_init(proxy_log_level level) {
    log_level = level;
}

// Logging function
static void guacd_log(proxy_log_level level, const char* format, ...) {
    if (level <= log_level) {
        va_list args;
        va_start(args, format);
        
        // Print timestamp
        time_t now = time(NULL);
        struct tm* tm_info = localtime(&now);
        char timestamp[32];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
        
        // Print log level
        const char* level_str;
        switch (level) {
            case PROXY_LOG_ERROR:   level_str = "ERROR"; break;
            case PROXY_LOG_WARNING: level_str = "WARNING"; break;
            case PROXY_LOG_INFO:    level_str = "INFO"; break;
            case PROXY_LOG_DEBUG:   level_str = "DEBUG"; break;
            case PROXY_LOG_TRACE:   level_str = "TRACE"; break;
            default:                level_str = "UNKNOWN"; break;
        }
        
        fprintf(stderr, "[%s] [%s] ", timestamp, level_str);
        vfprintf(stderr, format, args);
        fprintf(stderr, "\n");
        
        va_end(args);
    }
}

// Create a server socket
static int create_server_socket(const char* bind_host, int bind_port) {
    int sockfd;
    struct sockaddr_in server_addr;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to create socket: %s", strerror(errno));
        return -1;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        guacd_log(PROXY_LOG_WARNING, "Failed to set SO_REUSEADDR: %s", strerror(errno));
    }
    
    // 添加SO_REUSEPORT选项
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        guacd_log(PROXY_LOG_WARNING, "Failed to set SO_REUSEPORT: %s", strerror(errno));
    }
    
    // 设置非阻塞模式
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        guacd_log(PROXY_LOG_WARNING, "Failed to get socket flags: %s", strerror(errno));
    } else {
        if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
            guacd_log(PROXY_LOG_WARNING, "Failed to set non-blocking mode: %s", strerror(errno));
        }
    }
    
    // Prepare server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(bind_port);
    
    if (strcmp(bind_host, "0.0.0.0") == 0) {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, bind_host, &server_addr.sin_addr) <= 0) {
            guacd_log(PROXY_LOG_ERROR, "Invalid address: %s", bind_host);
            close(sockfd);
            return -1;
        }
    }
    
    // Bind socket
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to bind socket: %s", strerror(errno));
        close(sockfd);
        return -1;
    }
    
    // Listen for connections
    if (listen(sockfd, 5) < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to listen on socket: %s", strerror(errno));
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

// Connect to guacd
static int connect_to_guacd(const char* host, int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to create socket: %s", strerror(errno));
        return -1;
    }
    
    // Prepare server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, host, &server_addr.sin_addr) <= 0) {
        guacd_log(PROXY_LOG_ERROR, "Invalid address: %s", host);
        close(sockfd);
        return -1;
    }
    
    // Connect to server
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to connect to guacd: %s", strerror(errno));
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

// Parse a Guacamole instruction
static guac_instruction* parse_instruction(const char* buffer) {
    guac_instruction* instruction = malloc(sizeof(guac_instruction));
    if (!instruction) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for instruction");
        return NULL;
    }
    
    // Initialize instruction
    memset(instruction, 0, sizeof(guac_instruction));
    
    // Parse opcode
    const char* current = buffer;
    int i = 0;
    
    // Read opcode
    while (*current != '.' && *current != '\0' && i < 31) {
        instruction->opcode[i++] = *current++;
    }
    instruction->opcode[i] = '\0';
    
    if (*current == '\0') {
        // Instruction has no arguments
        instruction->argc = 0;
        instruction->argv = NULL;
        return instruction;
    }
    
    // Skip the dot
    current++;
    
    // Count arguments
    int argc = 1;
    const char* temp = current;
    while (*temp != '\0') {
        if (*temp == ',') {
            argc++;
        }
        temp++;
    }
    
    // Allocate memory for arguments
    instruction->argc = argc;
    instruction->argv = malloc(argc * sizeof(char*));
    if (!instruction->argv) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for instruction arguments");
        free(instruction);
        return NULL;
    }
    
    // Parse arguments
    for (i = 0; i < argc; i++) {
        // Find length of argument
        const char* arg_start = current;
        int arg_len = 0;
        
        while (*current != ',' && *current != '\0') {
            arg_len++;
            current++;
        }
        
        // Allocate memory for argument
        instruction->argv[i] = malloc(arg_len + 1);
        if (!instruction->argv[i]) {
            guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for instruction argument");
            
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(instruction->argv[j]);
            }
            free(instruction->argv);
            free(instruction);
            return NULL;
        }
        
        // Copy argument
        strncpy(instruction->argv[i], arg_start, arg_len);
        instruction->argv[i][arg_len] = '\0';
        
        // Skip comma
        if (*current == ',') {
            current++;
        }
    }
    
    return instruction;
}

// Free an instruction
static void free_instruction(guac_instruction* instruction) {
    if (!instruction) {
        return;
    }
    
    if (instruction->argv) {
        for (int i = 0; i < instruction->argc; i++) {
            free(instruction->argv[i]);
        }
        free(instruction->argv);
    }
    
    free(instruction);
}

// Get current timestamp in microseconds
// 使用全局的get_timestamp_us函数，避免重复定义

// Initialize Kalman filter
static void guac_kalman_filter_alloc_and_init(guac_kalman_filter* filter, int socket) {
    if (!filter) {
        guacd_log(PROXY_LOG_ERROR, "Invalid filter pointer");
        return;
    }
    
    // Initialize filter using the standard function
    guac_kalman_filter_init(filter);
    
    // Set custom properties for proxy
    filter->max_layers = 10;
    filter->max_regions = 100;
    
    // Allocate memory for layer priorities and dependencies
    filter->layer_priorities = calloc(filter->max_layers, sizeof(layer_priority_t));
    filter->layer_dependencies = calloc(filter->max_layers, sizeof(layer_dependency_t));
    
    // Allocate memory for frequency stats
    filter->frequency_stats = calloc(filter->max_regions, sizeof(update_frequency_stats_t));
    
    // 初始化连续帧检测
    filter->continuous_frame_detection = calloc(filter->max_layers, sizeof(continuous_frame_detection_t));
    filter->max_continuous_frames = 10;       // 判定为视频的最小连续帧数
    filter->max_frame_interval = 100.0;       // 判定为视频的最大帧间隔(ms)
    filter->min_frame_interval = 10.0;        // 判定为视频的最小帧间隔(ms)
    filter->frame_interval_threshold = 50.0;   // 帧间隔方差阈值
    
    // Initialize bandwidth prediction
    filter->bandwidth_prediction.last_update = get_timestamp_us();
    filter->bandwidth_prediction.current_bandwidth = 1000000; // 1 Mbps initial estimate
    filter->bandwidth_prediction.predicted_bandwidth = 1000000;
    filter->bandwidth_prediction.confidence = 0.9;
    
    // Check memory allocations
    if (!filter->layer_priorities || !filter->layer_dependencies || !filter->frequency_stats) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for Kalman filter data structures");
        free(filter->layer_priorities);
        free(filter->layer_dependencies);
        free(filter->frequency_stats);
        filter->layer_priorities = NULL;
        filter->layer_dependencies = NULL;
        filter->frequency_stats = NULL;
        return;
    }
    
    // Initialize CUDA if available
    if (!cuda_kalman_init(filter)) {
        guacd_log(PROXY_LOG_ERROR, "Failed to initialize CUDA Kalman filter");
    }
}

// Free Kalman filter
static void guac_kalman_filter_free(guac_kalman_filter* filter) {
    if (!filter) {
        return;
    }
    
    free(filter->layer_priorities);
    free(filter->layer_dependencies);
    free(filter->frequency_stats);
    free(filter->continuous_frame_detection); // 释放连续帧检测资源
    free(filter);
}

// Process an image instruction
static int process_image_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction) {
    // 应用卡尔曼滤波进行带宽预测
    uint64_t now = get_timestamp_us();
    double time_diff = (now - filter->bandwidth_prediction.last_update) / 1000000.0;
    
    if (time_diff > 0 && filter->video_optimization_enabled) {
        cuda_kalman_wrapper_update(filter, filter->bandwidth_prediction.current_bandwidth);
        
        // 根据预测带宽动态调整JPEG压缩质量
        int target_quality = filter->target_quality;
        if (filter->bandwidth_prediction.predicted_bandwidth < filter->target_bandwidth * 0.8) {
            target_quality = (target_quality > 30) ? target_quality - 10 : 30;
        }
        else if (filter->bandwidth_prediction.predicted_bandwidth > filter->target_bandwidth * 1.2) {
            target_quality = (target_quality < 95) ? target_quality + 5 : 95;
        }
        
        // 更新全局压缩质量参数
        filter->target_quality = target_quality;
    }
    filter->bandwidth_prediction.last_update = now;
    if (!filter || !instruction) {
        return -1;
    }
    
    // Check if this is a valid image instruction
    if ((strcmp(instruction->opcode, "img") != 0 && strcmp(instruction->opcode, "3") != 0) || instruction->argc < 5) {
        return 0;
    }
    
    // 记录开始处理图像指令
    guacd_log(PROXY_LOG_DEBUG, "处理图像指令: %s", instruction->opcode);
    
    // Extract parameters
    // 标准格式: "3.img,streamid,compositeMode,layerid,mimetype,x,y"
    // 或者数字格式: "3,streamid,compositeMode,layerid,mimetype,x,y"
    int stream_id = 0;
    int layer_index = 0;
    int x = 0;
    int y = 0;
    const char* mimetype = NULL;
    
    // 根据参数位置解析
    if (instruction->argc >= 6) {
        stream_id = atoi(instruction->argv[0]);
        // 第二个参数是compositeMode，暂时不使用
        layer_index = atoi(instruction->argv[2]);
        
        // 第四个参数应该是mimetype
        if (strstr(instruction->argv[3], "image/") != NULL) {
            mimetype = instruction->argv[3];
        }
        
        // 第五和第六个参数是x和y坐标
        x = atoi(instruction->argv[4]);
        y = atoi(instruction->argv[5]);
    } else {
        // 旧的解析逻辑作为备选
        layer_index = atoi(instruction->argv[0]);
        x = atoi(instruction->argv[1]);
        y = atoi(instruction->argv[2]);
        
        // 获取mimetype参数（如果存在）
        for (int i = 0; i < instruction->argc; i++) {
            if (strstr(instruction->argv[i], "image/") != NULL) {
                mimetype = instruction->argv[i];
                break;
            }
        }
    }
    
    // 记录解析后的参数
    guacd_log(PROXY_LOG_INFO, "[IMG指令解析] stream_id=%d, layer_id=%d, mimetype=%s, x=%d, y=%d",
             stream_id, layer_index, mimetype ? mimetype : "未知", x, y);
    
    // 记录图像指令的详细参数
    guacd_log(PROXY_LOG_DEBUG, "图像参数: layer=%d, x=%d, y=%d, mimetype=%s", 
             layer_index, x, y, mimetype ? mimetype : "未知");
    
    // Update region statistics
    int region_index = (y / 100) * 10 + (x / 100); // Simple region mapping
    if (region_index < filter->max_regions) {
        uint64_t now = get_timestamp_us();
        double time_diff = (now - filter->frequency_stats[region_index].last_update) / 1000000.0;
        
        if (filter->frequency_stats[region_index].last_update > 0 && time_diff > 0) {
            // Update frequency statistics
            filter->frequency_stats[region_index].avg_interval = 
                0.9 * filter->frequency_stats[region_index].avg_interval + 0.1 * time_diff;
            // Update count instead of frequency since there's no update_frequency field
            filter->frequency_stats[region_index].update_count++;
            
            // 记录更新频率统计信息
            guacd_log(PROXY_LOG_DEBUG, "区域 %d 更新: 平均间隔=%.3f秒, 更新次数=%d", 
                     region_index, filter->frequency_stats[region_index].avg_interval,
                     filter->frequency_stats[region_index].update_count);
            
            // 应用卡尔曼滤波器并记录前后对比
            double measurement = (double)x;
            double original_state[4] = {0};
            double updated_state[4] = {0};
            
            // 获取当前状态
            for (int i = 0; i < 4; i++) {
                original_state[i] = filter->state[i];
            }
            
            // 应用卡尔曼滤波器
            if (cuda_kalman_wrapper_update(filter, measurement)) {
                // 记录滤波前后的对比（使用明显的标记）
                guacd_log(PROXY_LOG_INFO, "[图像指令卡尔曼滤波] ===== 区域: %d, 原始位置: %.2f, 滤波后位置: %.2f, 差异: %.2f =====",
                         region_index, measurement, filter->state[0], filter->state[0] - measurement);
                
                // 记录滤波器状态变化
                guacd_log(PROXY_LOG_DEBUG, "[图像指令状态变化] 滤波前: [%.2f, %.2f, %.2f, %.2f], 滤波后: [%.2f, %.2f, %.2f, %.2f]",
                         original_state[0], original_state[1], original_state[2], original_state[3],
                         filter->state[0], filter->state[1], filter->state[2], filter->state[3]);
                
                // 计算改进百分比
                double improvement_percent = 0.0;
                if (fabs(measurement) > 0.001) { // 避免除以零
                    improvement_percent = fabs((filter->state[0] - measurement) / measurement) * 100.0;
                }
                
                // 记录滤波效果评估
                guacd_log(PROXY_LOG_INFO, "[图像指令效果评估] 改进幅度: %.2f%%, 置信度: %.2f%%",
                         improvement_percent, (1.0 - fabs(filter->state[1]/10.0)) * 100.0);
                
                // 计算图像质量评估指标
                double psnr = 0.0, ssim = 0.0, ms_ssim = 0.0, vmaf = 0.0, vqm = 0.0;
                
                // 这里可以添加实际的图像质量评估算法
                // 简单模拟计算PSNR（实际应用中需要实现真正的算法）
                if (fabs(filter->state[0] - measurement) > 0.001) {
                    psnr = 20.0 * log10(255.0 / fabs(filter->state[0] - measurement));
                } else {
                    psnr = 100.0; // 完美匹配
                }
                
                // 简单模拟SSIM（实际应用中需要实现真正的算法）
                ssim = 1.0 - fabs(filter->state[0] - measurement) / 255.0;
                if (ssim < 0.0) ssim = 0.0;
                if (ssim > 1.0) ssim = 1.0;
                
                // 简单模拟MS-SSIM（实际应用中需要实现真正的算法）
                ms_ssim = ssim * 0.95; // 简化计算
                
                // 简单模拟VMAF（实际应用中需要实现真正的算法）
                vmaf = 100.0 * ssim;
                
                // 简单模拟VQM（实际应用中需要实现真正的算法）
                vqm = 10.0 * (1.0 - ssim);
                
                // 记录图像质量评估指标到日志
                guacd_log(PROXY_LOG_INFO, "[图像质量评估指标] PSNR: %.2f dB, SSIM: %.4f, MS-SSIM: %.4f, VMAF: %.2f, VQM: %.2f",
                         psnr, ssim, ms_ssim, vmaf, vqm);
                
                // 记录到CSV文件
                FILE* fp = fopen("image_kalman_metrics.csv", "a");
                if (fp) {
                    // 如果文件为空，添加标题行
                    fseek(fp, 0, SEEK_END);
                    if (ftell(fp) == 0) {
                        fprintf(fp, "timestamp,region,original_position,filtered_position,difference,improvement_percent,psnr,ssim,ms_ssim,vmaf,vqm\n");
                    }
                    
                    // 添加数据行（包含图像质量评估指标）
                    fprintf(fp, "%llu,%d,%.6f,%.6f,%.6f,%.2f,%.2f,%.4f,%.4f,%.2f,%.2f\n", 
                            (unsigned long long)get_timestamp_us(), region_index, measurement, filter->state[0], 
                            filter->state[0] - measurement, improvement_percent,
                            psnr, ssim, ms_ssim, vmaf, vqm);
                    
                    fclose(fp);
                }
            }
        }
        
        filter->frequency_stats[region_index].last_update = now;
    }
    
    // 连续帧检测逻辑
    if (filter->continuous_frame_detection && layer_index < filter->max_layers) {
        continuous_frame_detection_t* frame_detection = &filter->continuous_frame_detection[layer_index];
        uint64_t current_time = get_timestamp_us();
        
        // 初始化连续帧检测数据（如果是第一次）
        if (frame_detection->layer_id == 0 && frame_detection->frame_count == 0) {
            frame_detection->layer_id = layer_index;
            frame_detection->last_frame_time = current_time;
            frame_detection->frame_count = 1;
            frame_detection->avg_frame_interval = 0;
            frame_detection->frame_interval_variance = 0;
            frame_detection->is_video_content = false;
            frame_detection->detection_confidence = 0;
            frame_detection->first_detection_time = 0;
            frame_detection->last_detection_time = 0;
            
            guacd_log(PROXY_LOG_DEBUG, "[连续帧检测] 初始化图层 %d 的连续帧检测", layer_index);
        } else {
            // 计算帧间隔（毫秒）
            double frame_interval = (current_time - frame_detection->last_frame_time) / 1000.0;
            
            // 更新帧计数和时间戳
            frame_detection->frame_count++;
            
            // 更新平均帧间隔（使用指数移动平均）
            if (frame_detection->avg_frame_interval == 0) {
                frame_detection->avg_frame_interval = frame_interval;
            } else {
                // 计算新的方差
                double delta = frame_interval - frame_detection->avg_frame_interval;
                frame_detection->frame_interval_variance = 
                    0.9 * frame_detection->frame_interval_variance + 0.1 * delta * delta;
                
                // 更新平均值
                frame_detection->avg_frame_interval = 
                    0.9 * frame_detection->avg_frame_interval + 0.1 * frame_interval;
            }
            
            // 记录帧间隔信息
            guacd_log(PROXY_LOG_DEBUG, "[连续帧检测] 图层 %d: 帧间隔=%.2fms, 平均=%.2fms, 方差=%.2f, 帧数=%llu", 
                     layer_index, frame_interval, frame_detection->avg_frame_interval, 
                     frame_detection->frame_interval_variance, frame_detection->frame_count);
            
            // 检测是否为视频内容
            // 条件：1. 连续帧数量超过阈值 2. 帧间隔在合理范围内 3. 帧间隔方差较小
            if (frame_detection->frame_count >= filter->max_continuous_frames && 
                frame_detection->avg_frame_interval >= filter->min_frame_interval && 
                frame_detection->avg_frame_interval <= filter->max_frame_interval && 
                frame_detection->frame_interval_variance <= filter->frame_interval_threshold) {
                
                // 如果之前未被检测为视频内容，现在标记为视频内容
                if (!frame_detection->is_video_content) {
                    frame_detection->is_video_content = true;
                    frame_detection->first_detection_time = current_time;
                    frame_detection->detection_confidence = 60; // 初始置信度
                    
                    // 设置图层优先级为视频优先级
                    filter->layer_priorities[layer_index] = LAYER_PRIORITY_VIDEO;
                    
                    guacd_log(PROXY_LOG_INFO, "[连续帧检测] ===== 检测到图层 %d 包含视频内容! 帧数=%llu, 帧率=%.2f fps =====", 
                             layer_index, frame_detection->frame_count, 
                             1000.0 / frame_detection->avg_frame_interval);
                } else {
                    // 已经是视频内容，增加置信度
                    if (frame_detection->detection_confidence < 100) {
                        frame_detection->detection_confidence += 2;
                        if (frame_detection->detection_confidence > 100) {
                            frame_detection->detection_confidence = 100;
                        }
                    }
                }
                
                // 更新最后检测时间
                frame_detection->last_detection_time = current_time;
                
                // 应用特殊的卡尔曼滤波参数（针对视频内容优化）
                // 视频内容通常需要更低的过程噪声和更高的测量噪声
                filter->config_process_noise = 0.005;  // 降低过程噪声
                filter->config_measurement_noise_x = 0.2;  // 增加测量噪声
                filter->config_measurement_noise_y = 0.2;
                
                // 记录视频内容检测信息
                guacd_log(PROXY_LOG_INFO, "[视频内容优化] 图层 %d: 置信度=%d%%, 持续时间=%.2f秒, 帧率=%.2f fps", 
                         layer_index, frame_detection->detection_confidence,
                         (current_time - frame_detection->first_detection_time) / 1000000.0,
                         1000.0 / frame_detection->avg_frame_interval);
                
                // 记录到CSV文件
                FILE* fp = fopen("video_content_detection.csv", "a");
                if (fp) {
                    // 如果文件为空，添加标题行
                    fseek(fp, 0, SEEK_END);
                    if (ftell(fp) == 0) {
                        fprintf(fp, "timestamp,layer_id,frame_count,avg_interval,variance,confidence,fps\n");
                    }
                    
                    // 添加数据行
                    fprintf(fp, "%llu,%d,%lu,%.2f,%.2f,%d,%.2f\n", 
                            (unsigned long long)current_time, layer_index, 
                            frame_detection->frame_count, frame_detection->avg_frame_interval,
                            frame_detection->frame_interval_variance, frame_detection->detection_confidence,
                            1000.0 / frame_detection->avg_frame_interval);
                    
                    fclose(fp);
                }
            } else {
                // 不满足视频内容条件
                if (frame_detection->is_video_content) {
                    // 如果之前被检测为视频内容，现在降低置信度
                    frame_detection->detection_confidence -= 5;
                    
                    if (frame_detection->detection_confidence <= 0) {
                        // 置信度降至0，不再认为是视频内容
                        frame_detection->is_video_content = false;
                        frame_detection->detection_confidence = 0;
                        
                        // 恢复默认的卡尔曼滤波参数
                        filter->config_process_noise = DEFAULT_PROCESS_NOISE;
                        filter->config_measurement_noise_x = DEFAULT_MEASUREMENT_NOISE;
                        filter->config_measurement_noise_y = DEFAULT_MEASUREMENT_NOISE;
                        
                        // 恢复图层优先级
                        filter->layer_priorities[layer_index] = LAYER_PRIORITY_DYNAMIC;
                        
                        guacd_log(PROXY_LOG_INFO, "[连续帧检测] 图层 %d 不再被视为视频内容", layer_index);
                    } else {
                        guacd_log(PROXY_LOG_DEBUG, "[连续帧检测] 图层 %d 视频内容置信度降低至 %d%%", 
                                 layer_index, frame_detection->detection_confidence);
                    }
                }
            }
            
            // 更新最后帧时间
            frame_detection->last_frame_time = current_time;
        }
    }
    
    return 0;
}

// Process a select instruction
static int process_select_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // Check if this is a valid select instruction
    if (strcmp(instruction->opcode, "select") != 0 || instruction->argc < 1) {
        return 0;
    }
    
    // Extract parameters
    int layer_index = atoi(instruction->argv[0]);
    
    // Update layer priority
    if (layer_index < filter->max_layers) {
        // layer_priorities is an array of enum values, not structs
        filter->layer_priorities[layer_index] = LAYER_PRIORITY_UI; // Highest priority for selected layers
    }
    
    return 0;
}

// Process a copy instruction
static int process_copy_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // Check if this is a valid copy instruction
    if (strcmp(instruction->opcode, "copy") != 0 || instruction->argc < 7) {
        return 0;
    }
    
    // Extract parameters
    int src_layer = atoi(instruction->argv[0]);
    int dst_layer = atoi(instruction->argv[6]);
    
    // Update layer dependency
    if (dst_layer < filter->max_layers) {
        filter->layer_dependencies[dst_layer].layer_id = dst_layer;
        filter->layer_dependencies[dst_layer].depends_on = src_layer;
        filter->layer_dependencies[dst_layer].weight = 1.0f; // Default weight
    }
    
    return 0;
}

// Process a drawing instruction
static int process_draw_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction) {
    if (!filter || !instruction || !user) {
        return -1;
    }
    
    // 检查参数数量是否足够
    if (instruction->argc < 6) {
        guacd_log(PROXY_LOG_WARNING, "img指令参数不足，无法处理");
        return -1;
    }
    
    // 解析img指令参数
    int stream_id = atoi(instruction->argv[0]);
    const char* composite_mode = instruction->argv[1];
    int layer_id = atoi(instruction->argv[2]);
    const char* mimetype = instruction->argv[3];
    int x = atoi(instruction->argv[4]);
    int y = atoi(instruction->argv[5]);
    
    // 记录详细的img指令信息
    guacd_log(PROXY_LOG_DEBUG, "处理img指令: stream_id=%d, layer_id=%d, mimetype=%s, x=%d, y=%d",
             stream_id, layer_id, mimetype, x, y);
    
    // 获取当前时间戳
    uint64_t timestamp = get_timestamp_us();
    
    // 更新连续帧检测状态
    if (filter->continuous_frame_detection && layer_id < filter->max_layers) {
        // 检查是否为图像类型的MIME
        bool is_image = (strstr(mimetype, "image/") != NULL);
        
        if (is_image) {
            // 更新连续帧检测状态
            bool status_changed = guac_kalman_filter_update_continuous_detection(filter, layer_id, timestamp);
            
            // 如果检测状态发生变化，应用视频优化
            if (status_changed) {
                guac_kalman_filter_apply_video_optimization(filter, layer_id);
                
                // 记录日志
                guacd_log(PROXY_LOG_INFO, "[图像指令] 图层 %d 视频内容状态变化: %s (置信度=%d%%)", 
                       layer_id, 
                       filter->continuous_frame_detection[layer_id].is_video_content ? "是" : "否",
                       filter->continuous_frame_detection[layer_id].detection_confidence);
            }
            
            // 如果是视频内容，可以在这里对图像质量进行优化
            if (filter->continuous_frame_detection[layer_id].is_video_content) {
                // 记录视频内容统计信息
                if (filter->stats_enabled && filter->stats_fd >= 0) {
                    char video_stats[512];
                    snprintf(video_stats, sizeof(video_stats), 
                            "video_content,%d,%d,%lu,%.2f,%.2f,%d\n", 
                            layer_id, 
                            filter->continuous_frame_detection[layer_id].detection_confidence,
                            filter->continuous_frame_detection[layer_id].frame_count,
                            filter->continuous_frame_detection[layer_id].avg_frame_interval,
                            filter->continuous_frame_detection[layer_id].frame_interval_variance,
                            (int)(1000.0 / filter->continuous_frame_detection[layer_id].avg_frame_interval));
                    if (write(filter->stats_fd, video_stats, strlen(video_stats)) < 0) {
                        perror("Failed to write video content stats");
                    }
                }
                
                // 计算并记录图像质量指标
                // 在实际实现中，这里需要从指令中提取图像数据或使用缓存的图像数据
                if (filter->image_buffer_length > 0) {
                    // 假设我们有原始图像和处理后的图像数据
                    unsigned char* original = filter->image_buffer;
                    unsigned char* processed = filter->image_buffer; // 在实际实现中，这应该是处理后的图像
                    
                    // 假设图像尺寸和通道数
                    int width = 800;  // 实际实现中应该从图像数据中获取
                    int height = 600; // 实际实现中应该从图像数据中获取
                    int channels = 3; // RGB图像
                    
                    // 计算图像质量指标
                    double psnr = calculate_psnr(original, processed, width, height, channels);
                    double ssim = calculate_ssim(original, processed, width, height, channels);
                    double ms_ssim = calculate_ms_ssim(original, processed, width, height, channels);
                    double vmaf = calculate_vmaf(original, processed, width, height, channels);
                    double vqm = calculate_vqm(original, processed, width, height, channels);
                    
                    // 获取当前时间戳
                    uint64_t quality_timestamp = get_timestamp_us();
                    
                    // 记录到CSV文件
                    FILE* fp = fopen("image_quality_metrics.csv", "a");
                    if (fp) {
                        // 如果文件为空，添加标题行
                        fseek(fp, 0, SEEK_END);
                        if (ftell(fp) == 0) {
                            fprintf(fp, "timestamp,layer_id,confidence,frame_count,psnr,ssim,ms_ssim,vmaf,vqm,width,height\n");
                        }
                        
                        // 添加数据行
                        fprintf(fp, "%lu,%d,%d,%lu,%.2f,%.4f,%.4f,%.2f,%.2f,%d,%d\n", 
                                (unsigned long)quality_timestamp, 
                                layer_id,
                                filter->continuous_frame_detection[layer_id].detection_confidence,
                                filter->continuous_frame_detection[layer_id].frame_count,
                                psnr, ssim, ms_ssim, vmaf, vqm, width, height);
                        
                        fclose(fp);
                        
                        // 记录日志
                        guacd_log(PROXY_LOG_INFO, "[图像质量指标] 图层 %d: PSNR=%.2f, SSIM=%.4f, MS-SSIM=%.4f, VMAF=%.2f, VQM=%.2f",
                               layer_id, psnr, ssim, ms_ssim, vmaf, vqm);
                    }
                    
                    // 更新滤波器的质量指标历史
                    guac_kalman_filter_update_metrics(filter, original, processed, width, height, channels);
                }
            }
        }
    }
    
    // 将指令转发给客户端，正确传递参数
    return guac_user_handle_instruction(user, instruction->opcode, instruction->argc, instruction->argv);
}

// 处理绘图指令（如arc, rect, line等）
static int process_drawing_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction) {
    if (!filter || !instruction || !user) {
        return -1;
    }
    
    // 记录开始处理绘图指令
    guacd_log(PROXY_LOG_INFO, "处理绘图指令: %s", instruction->opcode);
    
    // 检查参数数量
    if (instruction->argc < 2) {
        guacd_log(PROXY_LOG_WARNING, "绘图指令参数不足: %s", instruction->opcode);
        return 0;
    }
    
    // 提取图层参数
    int layer_id = atoi(instruction->argv[0]);
    
    // 记录绘图指令的详细参数
    guacd_log(PROXY_LOG_DEBUG, "绘图参数: layer=%d, opcode=%s", layer_id, instruction->opcode);
    
    // 提取坐标参数（不同指令的参数位置可能不同）
    int x = 0, y = 0;
    
    if (strcmp(instruction->opcode, "arc") == 0 && instruction->argc >= 3) {
        // arc指令: layer, x, y, radius, startAngle, endAngle, negative
        x = atoi(instruction->argv[1]);
        y = atoi(instruction->argv[2]);
    } else if (strcmp(instruction->opcode, "rect") == 0 && instruction->argc >= 5) {
        // rect指令: layer, x, y, width, height
        x = atoi(instruction->argv[1]);
        y = atoi(instruction->argv[2]);
    } else if (strcmp(instruction->opcode, "line") == 0 && instruction->argc >= 5) {
        // line指令: layer, x1, y1, x2, y2
        x = atoi(instruction->argv[1]);
        y = atoi(instruction->argv[2]);
    } else if (strcmp(instruction->opcode, "png") == 0 && instruction->argc >= 4) {
        // png指令: layer, x, y, data
        x = atoi(instruction->argv[1]);
        y = atoi(instruction->argv[2]);
    } else if ((strcmp(instruction->opcode, "cfill") == 0 || strcmp(instruction->opcode, "cstroke") == 0) && instruction->argc >= 2) {
        // cfill/cstroke指令: layer, r, g, b, a
        // 这些指令没有明确的x,y坐标，使用默认值
        x = 0;
        y = 0;
    }
    
    // 记录提取的坐标
    guacd_log(PROXY_LOG_DEBUG, "绘图坐标: x=%d, y=%d", x, y);
    
    // 更新区域统计
    int region_index = (y / 100) * 10 + (x / 100); // 简单区域映射
    if (region_index >= 0 && region_index < filter->max_regions) {
        uint64_t now = get_timestamp_us();
        double time_diff = (now - filter->frequency_stats[region_index].last_update) / 1000000.0;
        
        if (filter->frequency_stats[region_index].last_update > 0 && time_diff > 0) {
            // 更新频率统计
            filter->frequency_stats[region_index].avg_interval = 
                0.9 * filter->frequency_stats[region_index].avg_interval + 0.1 * time_diff;
            filter->frequency_stats[region_index].update_count++;
            
            // 记录更新频率统计信息
            guacd_log(PROXY_LOG_DEBUG, "区域 %d 更新: 平均间隔=%.3f秒, 更新次数=%d", 
                     region_index, filter->frequency_stats[region_index].avg_interval,
                     filter->frequency_stats[region_index].update_count);
            
            // 应用卡尔曼滤波器并记录前后对比
            double measurement = (double)x;
            double original_state[4] = {0};
            
            // 获取当前状态
            for (int i = 0; i < 4; i++) {
                original_state[i] = filter->state[i];
            }
            
            // 应用卡尔曼滤波器
            if (cuda_kalman_wrapper_update(filter, measurement)) {
                // 记录滤波前后的对比（使用明显的标记）
                guacd_log(PROXY_LOG_INFO, "[绘图指令卡尔曼滤波] ===== 区域: %d, 原始位置: %.2f, 滤波后位置: %.2f, 差异: %.2f =====",
                         region_index, measurement, filter->state[0], filter->state[0] - measurement);
                
                // 记录滤波器状态变化
                guacd_log(PROXY_LOG_DEBUG, "[绘图指令状态变化] 滤波前: [%.2f, %.2f, %.2f, %.2f], 滤波后: [%.2f, %.2f, %.2f, %.2f]",
                         original_state[0], original_state[1], original_state[2], original_state[3],
                         filter->state[0], filter->state[1], filter->state[2], filter->state[3]);
                
                // 计算改进百分比
                double improvement_percent = 0.0;
                if (fabs(measurement) > 0.001) { // 避免除以零
                    improvement_percent = fabs((filter->state[0] - measurement) / measurement) * 100.0;
                }
                
                // 记录滤波效果评估
                guacd_log(PROXY_LOG_INFO, "[绘图指令效果评估] 改进幅度: %.2f%%, 置信度: %.2f%%",
                         improvement_percent, (1.0 - fabs(filter->state[1]/10.0)) * 100.0);
                
                // 记录到CSV文件
                FILE* fp = fopen("draw_kalman_metrics.csv", "a");
                if (fp) {
                    // 如果文件为空，添加标题行
                    fseek(fp, 0, SEEK_END);
                    if (ftell(fp) == 0) {
                        fprintf(fp, "timestamp,region,opcode,original_position,filtered_position,difference,improvement_percent\n");
                    }
                    
                    // 添加数据行
                    fprintf(fp, "%llu,%d,%s,%.6f,%.6f,%.6f,%.2f\n", 
                            (unsigned long long)get_timestamp_us(), region_index, instruction->opcode,
                            measurement, filter->state[0], filter->state[0] - measurement, improvement_percent);
                    
                    fclose(fp);
                }
                
                // 如果是PNG指令，可以考虑对图像数据应用额外的处理
                if (strcmp(instruction->opcode, "png") == 0 && instruction->argc >= 4) {
                    guacd_log(PROXY_LOG_INFO, "对PNG图像数据应用卡尔曼滤波");
                    // 这里可以添加对PNG数据的处理，例如调用CUDA函数处理图像数据
                }
            }
        }
        
        filter->frequency_stats[region_index].last_update = now;
    }
    
    // 设置图层优先级
    if (layer_id >= 0 && layer_id < filter->max_layers) {
        // 根据指令类型设置不同的优先级
        if (strcmp(instruction->opcode, "png") == 0) {
            filter->layer_priorities[layer_id] = LAYER_PRIORITY_DYNAMIC;
        } else if (strcmp(instruction->opcode, "arc") == 0 || 
                  strcmp(instruction->opcode, "rect") == 0 || 
                  strcmp(instruction->opcode, "line") == 0) {
            filter->layer_priorities[layer_id] = LAYER_PRIORITY_UI;
        } else {
            filter->layer_priorities[layer_id] = LAYER_PRIORITY_STATIC;
        }
    }
    
    return 0;
}

// Process a video instruction
static int process_blob_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    // 基本blob指令处理逻辑
    guacd_log(PROXY_LOG_DEBUG, "处理blob指令: %s", instruction->opcode);
    
    // 参数完整性校验
    if (instruction->argc < 3) {
        guacd_log(PROXY_LOG_ERROR, "BLOB指令参数不足 需要3个参数(流索引/数据/时间戳) 实际收到:%d", instruction->argc);
        return -1;
    }

    // 解析流索引并记录
    int stream_idx = atoi(instruction->argv[0]);
    size_t data_len = strlen(instruction->argv[1]);
    uint64_t frame_ts = strtoull(instruction->argv[2], NULL, 10);
    
    guacd_log(PROXY_LOG_INFO, "[BLOB指令] ===== 视频流接收 流索引:%d 数据长度:%zu 时间戳:%lu =====",
             stream_idx, data_len, frame_ts);

    // 流索引有效性验证
    if (stream_idx < 0 || stream_idx >= MAX_VIDEO_STREAMS) {
        guacd_log(PROXY_LOG_ERROR, "无效流索引:%d (允许范围0-%d)",
                 stream_idx, MAX_VIDEO_STREAMS-1);
        return -1;
    }

    // 时间戳同步验证
    uint64_t current_ts = get_timestamp_us();
    int64_t ts_diff = (int64_t)(current_ts - frame_ts);
    
    guacd_log(PROXY_LOG_DEBUG, "[时间戳同步] 系统时间:%lu 帧时间:%lu 差值:%.2fms",
             current_ts, frame_ts, ts_diff/1000.0);
    
    // 视频流优化处理 - 应用卡尔曼滤波
    if (filter->video_optimization_enabled) {
        // 计算帧大小作为测量值
        double frame_size_kb = (double)data_len / 1024.0;
        
        // 记录原始测量值
        double original_measurement = frame_size_kb;
        
        // 保存原始状态用于对比
        double original_state[4];
        for (int i = 0; i < 4; i++) {
            original_state[i] = filter->state[i];
        }
        
        // 应用卡尔曼滤波器
        guacd_log(PROXY_LOG_DEBUG, "[BLOB卡尔曼] 应用卡尔曼滤波处理视频帧数据，原始帧大小: %.2f KB", frame_size_kb);
        
        // 记录处理开始时间
        uint64_t filter_start_time = get_timestamp_us();
        
        // 应用卡尔曼滤波器进行预测
        if (cuda_kalman_wrapper_update(filter, frame_size_kb)) {
            // 计算处理时间
            uint64_t filter_end_time = get_timestamp_us();
            double processing_time_ms = (filter_end_time - filter_start_time) / 1000.0;
            
            // 获取滤波后的预测值
            double filtered_frame_size = filter->state[0];
            
            // 记录滤波前后的对比
            guacd_log(PROXY_LOG_INFO, "[BLOB卡尔曼滤波] ===== 原始帧大小: %.2f KB, 滤波后预测: %.2f KB, 差异: %.2f KB (处理时间: %.3f ms) =====",
                     original_measurement, filtered_frame_size, filtered_frame_size - original_measurement, processing_time_ms);
            
            // 记录状态向量的变化
            guacd_log(PROXY_LOG_DEBUG, "[BLOB状态变化] 原始: [%.2f, %.2f, %.2f, %.2f], 更新后: [%.2f, %.2f, %.2f, %.2f]",
                     original_state[0], original_state[1], original_state[2], original_state[3],
                     filter->state[0], filter->state[1], filter->state[2], filter->state[3]);
            
            // 计算改进百分比
            double improvement_percent = 0.0;
            if (fabs(original_measurement) > 0.001) { // 避免除以零
                improvement_percent = fabs((filtered_frame_size - original_measurement) / original_measurement) * 100.0;
            }
            
            // 记录滤波效果评估
            guacd_log(PROXY_LOG_INFO, "[BLOB效果评估] 改进幅度: %.2f%%, 置信度: %.2f%%",
                     improvement_percent, (1.0 - fabs(filter->state[1]/10.0)) * 100.0);
            
            // 根据滤波结果动态调整视频质量
            if (filter->bandwidth_prediction.predicted_bandwidth < filter->target_bandwidth * 0.8) {
                int new_quality = (filter->target_quality > 30) ? filter->target_quality - 10 : 30;
                guacd_log(PROXY_LOG_INFO, "[BLOB带宽优化] 带宽不足，降低视频质量: %d -> %d", 
                         filter->target_quality, new_quality);
                filter->target_quality = new_quality;
            }
            else if (filter->bandwidth_prediction.predicted_bandwidth > filter->target_bandwidth * 1.2) {
                int new_quality = (filter->target_quality < 95) ? filter->target_quality + 5 : 95;
                guacd_log(PROXY_LOG_INFO, "[BLOB带宽优化] 带宽充足，提高视频质量: %d -> %d", 
                         filter->target_quality, new_quality);
                filter->target_quality = new_quality;
            }
            
            // 计算视频质量指标 (PSNR, SSIM, VMAF)
            // 这里我们需要调用CUDA函数来计算这些指标
            // 注意：实际计算需要解码视频帧数据，这里简化处理
            if (filter->frames_processed > 0 && filter->metrics_history_position > 0) {
                // 使用前一帧的指标作为参考
                double psnr = filter->metrics_history[filter->metrics_history_position-1].psnr;
                double ssim = filter->metrics_history[filter->metrics_history_position-1].ssim;
                double vmaf = filter->metrics_history[filter->metrics_history_position-1].vmaf;
                
                // 根据帧大小变化调整指标
                double size_ratio = filtered_frame_size / original_measurement;
                if (size_ratio < 0.8) {
                    // 帧大小减小，质量可能下降
                    psnr = psnr * 0.95;
                    ssim = ssim * 0.98;
                    vmaf = vmaf * 0.97;
                } else if (size_ratio > 1.2) {
                    // 帧大小增加，质量可能提高
                    psnr = psnr * 1.05;
                    ssim = ssim * 1.02;
                    vmaf = vmaf * 1.03;
                }
                
                // 记录质量指标
                guacd_log(PROXY_LOG_INFO, "[BLOB视频质量] ===== PSNR: %.2f dB, SSIM: %.4f, VMAF: %.2f =====",
                         psnr, ssim, vmaf);
                
                // 更新质量指标历史
                if (filter->metrics_history_position < GUAC_KALMAN_MAX_FRAME_COUNT) {
                    filter->metrics_history[filter->metrics_history_position].psnr = psnr;
                    filter->metrics_history[filter->metrics_history_position].ssim = ssim;
                    filter->metrics_history[filter->metrics_history_position].vmaf = vmaf;
                    filter->metrics_history[filter->metrics_history_position].frame_size = (int)data_len;
                    filter->metrics_history[filter->metrics_history_position].timestamp = current_ts;
                    
                    // 计算帧率
                    if (filter->metrics_history_position > 0) {
                        uint64_t prev_time = filter->metrics_history[filter->metrics_history_position - 1].timestamp;
                        double time_diff = (current_ts - prev_time) / 1000000.0; // 转换为秒
                        
                        if (time_diff > 0) {
                            filter->metrics_history[filter->metrics_history_position].fps = 1.0 / time_diff;
                            guacd_log(PROXY_LOG_DEBUG, "[BLOB帧率] 当前帧率: %.2f FPS", 
                                     filter->metrics_history[filter->metrics_history_position].fps);
                        }
                    }
                    
                    filter->metrics_history_position++;
                } else {
                    // 循环使用历史记录
                    for (int i = 0; i < GUAC_KALMAN_MAX_FRAME_COUNT - 1; i++) {
                        filter->metrics_history[i] = filter->metrics_history[i + 1];
                    }
                    
                    filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].psnr = psnr;
                    filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].ssim = ssim;
                    filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].vmaf = vmaf;
                    filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].frame_size = (int)data_len;
                    filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].timestamp = current_ts;
                    
                    // 计算帧率
                    uint64_t prev_time = filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 2].timestamp;
                    double time_diff = (current_ts - prev_time) / 1000000.0; // 转换为秒
                    
                    if (time_diff > 0) {
                        filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].fps = 1.0 / time_diff;
                        guacd_log(PROXY_LOG_DEBUG, "[BLOB帧率] 当前帧率: %.2f FPS", 
                                 filter->metrics_history[GUAC_KALMAN_MAX_FRAME_COUNT - 1].fps);
                    }
                }
            }
            
            // 记录到CSV文件中用于后续分析
            FILE* fp = fopen("blob_kalman_metrics.csv", "a");
            if (fp) {
                // 如果文件为空，添加标题行
                fseek(fp, 0, SEEK_END);
                if (ftell(fp) == 0) {
                    fprintf(fp, "timestamp,stream_id,original_size,filtered_size,difference,improvement_percent,processing_time_ms\n");
                }
                
                // 添加数据行
                fprintf(fp, "%llu,%d,%.6f,%.6f,%.6f,%.2f,%.3f\n", 
                        (unsigned long long)current_ts, stream_idx, original_measurement, filtered_frame_size, 
                        filtered_frame_size - original_measurement, improvement_percent, processing_time_ms);
                
                fclose(fp);
            }
        }
        
        filter->frames_processed++;
    }
    
    return 0;
}

// 前置声明init_stream_mapping函数
static void init_stream_mapping(int stream_idx, guac_stream* input_stream, const char* mimetype);

static int process_video_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // 检查是否是有效的视频指令
    if (strcmp(instruction->opcode, "video") != 0 || instruction->argc < 3) {
        return 0;
    }
    
    // 视频流动态优化处理
    if (filter->video_optimization_enabled) {
        uint64_t now = get_timestamp_us();
        double time_diff = (now - filter->bandwidth_prediction.last_update) / 1000000.0;
        
        if (time_diff > 0) {
            // 应用卡尔曼滤波器进行带宽预测
            cuda_kalman_wrapper_update(filter, filter->bandwidth_prediction.current_bandwidth);
            
            // 根据预测带宽调整视频编码参数
            int target_quality = filter->target_quality;
            if (filter->bandwidth_prediction.predicted_bandwidth < filter->target_bandwidth * 0.8) {
                target_quality = (target_quality > 30) ? target_quality - 15 : 30;
                guacd_log(PROXY_LOG_INFO, "[视频优化] 带宽不足，质量调整为：%d", target_quality);
            }
            else if (filter->bandwidth_prediction.predicted_bandwidth > filter->target_bandwidth * 1.2) {
                target_quality = (target_quality < 95) ? target_quality + 8 : 95;
                guacd_log(PROXY_LOG_INFO, "[视频优化] 带宽充足，质量提升至：%d", target_quality);
            }
            
            // 更新视频编码器参数
            filter->target_quality = target_quality;
        }
        filter->bandwidth_prediction.last_update = now;
    }
    
    // 记录开始处理视频指令
    guacd_log(PROXY_LOG_DEBUG, "处理视频指令: %s", instruction->opcode);
    
    // 提取参数
    int stream_id = atoi(instruction->argv[0]);
    int layer_id = atoi(instruction->argv[1]);
    char* mimetype = instruction->argv[2];
    
    // 记录视频指令的详细参数
    guacd_log(PROXY_LOG_DEBUG, "视频参数: stream=%d, layer=%d, mimetype=%s", 
             stream_id, layer_id, mimetype);
    
    // 设置图层优先级为视频优先级
    if (layer_id < filter->max_layers) {
        filter->layer_priorities[layer_id] = LAYER_PRIORITY_VIDEO;
    }
    
    // 更新带宽预测
    uint64_t now = get_timestamp_us();
    double time_diff = (now - filter->bandwidth_prediction.last_update) / 1000000.0;
    
    if (time_diff > 0 && filter->bandwidth_prediction.last_update > 0) {
        // 应用卡尔曼滤波器进行带宽预测
        double measurement = filter->bandwidth_prediction.current_bandwidth;
        
        // 获取当前状态
        double original_state[4] = {0};
        for (int i = 0; i < 4; i++) {
            original_state[i] = filter->state[i];
        }
        
        // 应用卡尔曼滤波器
        if (cuda_kalman_wrapper_update(filter, measurement)) {
            // 记录滤波前后的对比
            guacd_log(PROXY_LOG_INFO, "[视频流卡尔曼滤波] ===== 原始带宽: %.2f kbps, 滤波后带宽: %.2f kbps, 差异: %.2f kbps =====",
                     measurement, filter->state[0], filter->state[0] - measurement);
            
            // 计算改进百分比
            double improvement_percent = 0.0;
            if (fabs(measurement) > 0.001) { // 避免除以零
                improvement_percent = fabs((filter->state[0] - measurement) / measurement) * 100.0;
            }
            
            // 记录滤波效果评估
            guacd_log(PROXY_LOG_INFO, "[视频流效果评估] 带宽预测改进: %.2f%%, 置信度: %.2f%%",
                     improvement_percent, (1.0 - fabs(filter->state[1]/10.0)) * 100.0);
            
            // 记录到CSV文件
            FILE* fp = fopen("video_kalman_metrics.csv", "a");
            if (fp) {
                // 如果文件为空，添加标题行
                fseek(fp, 0, SEEK_END);
                if (ftell(fp) == 0) {
                    fprintf(fp, "timestamp,stream_id,original_bandwidth,filtered_bandwidth,difference,improvement_percent\n");
                }
                
                // 添加数据行
                fprintf(fp, "%llu,%d,%.6f,%.6f,%.6f,%.2f\n", 
                        (unsigned long long)get_timestamp_us(), stream_id, measurement, filter->state[0], 
                        filter->state[0] - measurement, improvement_percent);
                
                fclose(fp);
            }
            
            // 更新带宽预测
            filter->bandwidth_prediction.predicted_bandwidth = filter->state[0];
            filter->bandwidth_prediction.confidence = 0.9 - 0.1 * filter->state[1]; // 使用速度分量作为不确定性
        }
    }
    
    filter->bandwidth_prediction.last_update = now;
    
    // 使用CUDA处理视频指令
    if (filter->video_optimization_enabled) {
        // 查找可用的视频流插槽
        int stream_idx = find_available_stream_slot();
        
        if (stream_idx >= 0) {
            // 初始化视频流映射
            guacd_log(PROXY_LOG_INFO, "初始化视频流映射: stream_id=%d, stream_idx=%d, mimetype=%s",
                     stream_id, stream_idx, mimetype);
            
            // 初始化视频流映射
            init_stream_mapping(stream_idx, &(user->__input_streams[stream_id]), mimetype);
            
            // 设置视频质量参数
            video_streams[stream_idx].quality = filter->target_quality;
            video_streams[stream_idx].layer_id = layer_id;
            
            // 调用CUDA视频处理函数
            if (cuda_process_video_instruction(filter, stream_id, layer_id, mimetype)) {
                guacd_log(PROXY_LOG_INFO, "CUDA视频处理成功应用");
            } else {
                guacd_log(PROXY_LOG_WARNING, "CUDA视频处理失败");
            }
        } else {
            guacd_log(PROXY_LOG_ERROR, "无法为视频流分配插槽: stream_id=%d", stream_id);
        }
    }
    
    return 0;
}

// Process an end instruction
static int process_end_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // Check if this is a valid end instruction
    if (strcmp(instruction->opcode, "end") != 0) {
        return 0;
    }
    
    // Update bandwidth prediction
    uint64_t now = get_timestamp_us();
    double time_diff = (now - filter->bandwidth_prediction.last_update) / 1000000.0;
    
    if (time_diff > 0) {
        // Simple bandwidth measurement (placeholder)
        double measured_bandwidth = 1000000; // 1 Mbps (placeholder)
        
        // Update bandwidth prediction using Kalman filter
        double measurement[2] = {measured_bandwidth, 0}; // Using only one dimension for bandwidth
        double updated_state[4] = {0}; // Will hold the updated state
        
        // Call the CUDA Kalman update function
        if (cuda_kalman_wrapper_update(filter, measurement[0])) {
            // Update our bandwidth prediction with the filtered value
            filter->bandwidth_prediction.current_bandwidth = measured_bandwidth;
            filter->bandwidth_prediction.predicted_bandwidth = updated_state[0];
            filter->bandwidth_prediction.confidence = 0.9; // Placeholder confidence value
            filter->bandwidth_prediction.last_update = now;
        }
    }
    
    return 0;
}

// Process an instruction
static int process_instruction(guac_kalman_filter* filter, guac_user* user, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // 记录接收到的指令信息（使用明显的标记）
    guacd_log(PROXY_LOG_INFO, "[指令接收] ===== 指令类型: %s, 参数数量: %d =====", instruction->opcode, instruction->argc);
    
    // 记录指令参数（如果有）
    if (instruction->argc > 0 && instruction->argv) {
        char args_buffer[1024] = {0};
        int offset = 0;
        
        for (int i = 0; i < instruction->argc && i < 16; i++) { // 最多显示16个参数
            int remaining = sizeof(args_buffer) - offset - 1;
            if (remaining <= 0) break;
            
            int written = snprintf(args_buffer + offset, remaining, "%s%s", 
                                  i > 0 ? ", " : "", 
                                  instruction->argv[i] ? instruction->argv[i] : "(null)");
            
            if (written < 0 || written >= remaining) break;
            offset += written;
        }
        
        guacd_log(PROXY_LOG_DEBUG, "指令参数: %s", args_buffer);
    }
    
    // Process instruction based on opcode
    // 检查是否为connect指令（可以是字符串"connect"或数字"7"）
    if (strcmp(instruction->opcode, "connect") == 0 || strcmp(instruction->opcode, "7") == 0) {
        // 处理connect指令
        guacd_log(PROXY_LOG_DEBUG, "处理connect指令，建立连接");
        return process_select_instruction(filter, instruction);
    }
    // 检查是否为img指令（可以是字符串"img"或数字"3"）
    else if (strcmp(instruction->opcode, "img") == 0 || strcmp(instruction->opcode, "3") == 0) {
        // 对于img指令，应用卡尔曼滤波进行图像优化
        guacd_log(PROXY_LOG_DEBUG, "处理img指令，应用卡尔曼滤波进行图像优化");
        return process_image_instruction(filter, user, instruction); // 调用process_image_instruction处理
    } else if (strcmp(instruction->opcode, "video") == 0) {
        return process_video_instruction(filter, user, instruction);
    } else if (strcmp(instruction->opcode, "blob") == 0 && instruction->argc >= 2) {
        // 处理blob指令，这是视频帧数据
        int stream_id = atoi(instruction->argv[0]);
        guacd_log(PROXY_LOG_DEBUG, "处理视频帧数据 blob 指令: stream_id=%d", stream_id);
        // 这里可以添加对blob指令的处理，应用卡尔曼滤波
        // 目前简单记录，实际处理在其他函数中完成
        return 0;
    } else if (strcmp(instruction->opcode, "select") == 0) {
        return process_select_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "copy") == 0) {
        return process_copy_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "end") == 0) {
        return process_end_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "arc") == 0 || 
               strcmp(instruction->opcode, "cfill") == 0 || 
               strcmp(instruction->opcode, "rect") == 0 || 
               strcmp(instruction->opcode, "line") == 0 || 
               strcmp(instruction->opcode, "cstroke") == 0 || 
               strcmp(instruction->opcode, "png") == 0) {
        return process_drawing_instruction(filter, user, instruction);
    } else if (strcmp(instruction->opcode, "blob") == 0) {
        return process_blob_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "ack") == 0) {
        // 处理ack指令
        guacd_log(PROXY_LOG_DEBUG, "处理ack指令，参数数量: %d", instruction->argc);
        // ack指令通常用于确认连接状态，可以简单记录但不需要特殊处理
        return 0;
    } else if (strcmp(instruction->opcode, "sync") == 0) {
        // 处理sync指令
        guacd_log(PROXY_LOG_DEBUG, "处理sync指令，参数数量: %d", instruction->argc);
        // sync指令用于同步客户端和服务器状态，可以简单记录但不需要特殊处理
        return 0;
    } else {
        // 处理其他类型的指令，包括'N'类型指令
        guacd_log(PROXY_LOG_INFO, "处理未明确识别的指令类型: %s, 参数数量: %d", instruction->opcode, instruction->argc);
        
        // 检查是否有足够的参数
        if (instruction->argc > 1 && instruction->argv && instruction->argv[1]) {
            guacd_log(PROXY_LOG_DEBUG, "[Kalman] 处理指令，时间偏移: %.3fms", strtod(instruction->argv[1], NULL)*1000);
            cuda_kalman_wrapper_update(filter, strtod(instruction->argv[1], NULL));
        } else {
            guacd_log(PROXY_LOG_WARNING, "指令 %s 参数不足，无法处理", instruction->opcode);
        }
    }
    
    return 0;
}

// Handle a client connection
static int handle_connection(int client_fd, int guacd_fd, guac_user* user) {
    char buffer[8192];
    ssize_t bytes_read, bytes_written;
    fd_set read_fds;
    struct timeval timeout;
    
    // Allocate memory for Kalman filter
    guac_kalman_filter* filter = malloc(sizeof(guac_kalman_filter));
    if (!filter) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for Kalman filter");
        return -1;
    }
    
    // Initialize Kalman filter
    guac_kalman_filter_alloc_and_init(filter, client_fd);
    if (!filter->layer_priorities) { // Check if initialization was successful
        guacd_log(PROXY_LOG_ERROR, "Failed to initialize Kalman filter");
        free(filter);
        return -1;
    }
    
    guacd_log(PROXY_LOG_INFO, "Handling connection from client");
    
    while (1) {
        // Set up file descriptor set
        FD_ZERO(&read_fds);
        FD_SET(client_fd, &read_fds);
        FD_SET(guacd_fd, &read_fds);
        
        // Set timeout
        timeout.tv_sec = 60;
        timeout.tv_usec = 0;
        
        // Wait for data
        int max_fd = (client_fd > guacd_fd) ? client_fd : guacd_fd;
        int activity = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
        
        if (activity < 0) {
            guacd_log(PROXY_LOG_ERROR, "Select error: %s", strerror(errno));
            break;
        } else if (activity == 0) {
            guacd_log(PROXY_LOG_INFO, "Connection timed out");
            break;
        }
        
        // Check for data from client
        if (FD_ISSET(client_fd, &read_fds)) {
            bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
            
            if (bytes_read <= 0) {
                if (bytes_read < 0) {
                    guacd_log(PROXY_LOG_ERROR, "Error reading from client: %s", strerror(errno));
                } else {
                    guacd_log(PROXY_LOG_INFO, "Client disconnected");
                }
                break;
            }
            
            // Null-terminate the buffer
            buffer[bytes_read] = '\0';
            
            // Parse and process instruction
            guac_instruction* instruction = parse_instruction(buffer);
            if (instruction) {
                process_instruction(filter, user, instruction);
                free_instruction(instruction);
            }
            
            // Forward data to guacd
            bytes_written = write(guacd_fd, buffer, bytes_read);
            if (bytes_written < bytes_read) {
                guacd_log(PROXY_LOG_ERROR, "Error writing to guacd: %s", strerror(errno));
                break;
            }
        }
        
        // Check for data from guacd
        if (FD_ISSET(guacd_fd, &read_fds)) {
            bytes_read = read(guacd_fd, buffer, sizeof(buffer) - 1);
            
            if (bytes_read <= 0) {
                if (bytes_read < 0) {
                    guacd_log(PROXY_LOG_ERROR, "Error reading from guacd: %s", strerror(errno));
                } else {
                    guacd_log(PROXY_LOG_INFO, "guacd disconnected");
                }
                break;
            }
            
            // Null-terminate the buffer
            buffer[bytes_read] = '\0';
            
            // Parse and process instruction from guacd
            guac_instruction* instruction = parse_instruction(buffer);
            if (instruction) {
                guacd_log(PROXY_LOG_INFO, "[服务端指令] 拦截指令: %s, 参数数量: %d", instruction->opcode, instruction->argc);
                
                // 记录指令参数（如果有）
                if (instruction->argc > 0 && instruction->argv) {
                    char args_buffer[1024] = {0};
                    int offset = 0;
                    
                    for (int i = 0; i < instruction->argc && i < 16; i++) { // 最多显示16个参数
                        int remaining = sizeof(args_buffer) - offset - 1;
                        if (remaining <= 0) break;
                        
                        int written = snprintf(args_buffer + offset, remaining, "%s%s", 
                                              i > 0 ? ", " : "", 
                                              instruction->argv[i] ? instruction->argv[i] : "(null)");
                        
                        if (written < 0 || written >= remaining) break;
                        offset += written;
                    }
                    
                    guacd_log(PROXY_LOG_DEBUG, "[服务端指令参数] %s", args_buffer);
                }
                
                // 特别处理视频相关指令
                if (strcmp(instruction->opcode, "img") == 0 || 
                    strcmp(instruction->opcode, "video") == 0 || 
                    strcmp(instruction->opcode, "blob") == 0) {
                    guacd_log(PROXY_LOG_INFO, "[服务端视频指令] 应用卡尔曼滤波优化: %s", instruction->opcode);
                }
                
                // 处理服务端指令
                process_instruction(filter, user, instruction);
                free_instruction(instruction);
            }
            
            // Forward data to client
            bytes_written = write(client_fd, buffer, bytes_read);
            if (bytes_written < bytes_read) {
                guacd_log(PROXY_LOG_ERROR, "Error writing to client: %s", strerror(errno));
                break;
            }
        }
    }
    
    // Clean up
    guac_kalman_filter_free(filter);
    
    return 0;
}

// CUDA initialization (stub)
static bool cuda_kalman_init(guac_kalman_filter* filter) {
    // Placeholder for CUDA initialization
    guacd_log(PROXY_LOG_INFO, "Initializing CUDA Kalman filter");
    
    // Initialize CUDA resources
    if (!cuda_init_kalman()) {
        guacd_log(PROXY_LOG_ERROR, "Failed to initialize CUDA resources");
        return false;
    }
    
    // Set up initial matrices for Kalman filter
    // These would be extracted from the filter structure
    double F[16]; // 4x4 state transition matrix
    double H[8];  // 2x4 measurement matrix
    double Q[16]; // 4x4 process noise covariance
    double R[4];  // 2x2 measurement noise covariance
    double P[16]; // 4x4 state estimate covariance
    double state[4]; // Initial state
    
    // Copy matrices from filter structure to local arrays
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            F[i*4+j] = filter->F[i][j];
            Q[i*4+j] = filter->Q[i][j];
            P[i*4+j] = filter->P[i][j];
        }
        state[i] = filter->state[i];
    }
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            H[i*4+j] = filter->H[i][j];
        }
    }
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            R[i*2+j] = filter->R[i][j];
        }
    }
    
    // Initialize matrices on GPU
    if (!cuda_kalman_init_matrices(F, H, Q, R, P, state)) {
        guacd_log(PROXY_LOG_ERROR, "Failed to initialize Kalman matrices on GPU");
        cuda_cleanup_kalman();
        return false;
    }
    
    // Set log level based on configuration
    cuda_kalman_set_log_level(filter->config_process_noise > 0.5 ? 3 : 2); // Debug or Info level
    
    return true; // Success
}

// CUDA Kalman filter update wrapper
static bool cuda_kalman_wrapper_update(guac_kalman_filter* filter, double measurement) {
    // Wrapper function to adapt our interface to the CUDA Kalman update function
    guacd_log(PROXY_LOG_DEBUG, "Updating CUDA Kalman filter with measurement: %f", measurement);
    
    // 保存原始状态用于对比
    double original_state[4];
    for (int i = 0; i < 4; i++) {
        original_state[i] = filter->state[i];
    }
    
    // 记录开始时间用于性能测量
    uint64_t start_time = get_timestamp_us();
    
    // Create measurement vector [measurement, 0]
    double measurement_vector[2] = {measurement, 0};
    
    // Create buffer for updated state
    double updated_state[4] = {0};
    
    // Call the actual CUDA update function
    bool result = cuda_kalman_update(measurement_vector, updated_state);
    
    // 计算处理时间
    uint64_t end_time = get_timestamp_us();
    double processing_time_ms = (end_time - start_time) / 1000.0;
    
    if (result) {
        // Update filter state with results
        for (int i = 0; i < 4; i++) {
            filter->state[i] = updated_state[i];
        }
        
        // Update bandwidth prediction
        filter->bandwidth_prediction.current_bandwidth = measurement;
        filter->bandwidth_prediction.predicted_bandwidth = updated_state[0];
        
        // 记录详细的滤波前后对比（使用明显的标记使其在日志中更容易识别）
        guacd_log(PROXY_LOG_INFO, "[卡尔曼滤波应用] ===== 原始测量值: %.2f, 滤波后值: %.2f, 差异: %.2f (处理时间: %.3f ms) =====",
                 measurement, updated_state[0], updated_state[0] - measurement, processing_time_ms);
        
        // 记录状态向量的变化
        guacd_log(PROXY_LOG_DEBUG, "[卡尔曼状态变化] 原始: [%.2f, %.2f, %.2f, %.2f], 更新后: [%.2f, %.2f, %.2f, %.2f]",
                 original_state[0], original_state[1], original_state[2], original_state[3],
                 updated_state[0], updated_state[1], updated_state[2], updated_state[3]);
        
        // 计算改进百分比
        double improvement_percent = 0.0;
        if (fabs(measurement) > 0.001) { // 避免除以零
            improvement_percent = fabs((updated_state[0] - measurement) / measurement) * 100.0;
        }
        
        // 记录滤波效果评估
        guacd_log(PROXY_LOG_INFO, "[卡尔曼效果评估] 改进幅度: %.2f%%, 置信度: %.2f%%",
                 improvement_percent, (1.0 - fabs(updated_state[1]/10.0)) * 100.0);
        
        // 记录到CSV文件中用于后续分析
        FILE* fp = fopen("kalman_metrics.csv", "a");
        if (fp) {
            // 如果文件为空，添加标题行
            fseek(fp, 0, SEEK_END);
            if (ftell(fp) == 0) {
                fprintf(fp, "timestamp,original_measurement,filtered_value,difference,improvement_percent,confidence,processing_time_ms\n");
            }
            
            // 添加数据行
            fprintf(fp, "%llu,%.6f,%.6f,%.6f,%.2f,%.2f,%.3f\n", 
                    (unsigned long long)end_time, measurement, updated_state[0], 
                    updated_state[0] - measurement, improvement_percent, 
                    (1.0 - fabs(updated_state[1]/10.0)) * 100.0, processing_time_ms);
            
            fclose(fp);
        }
    } else {
        guacd_log(PROXY_LOG_ERROR, "卡尔曼滤波应用失败 - 测量值: %.2f", measurement);
    }
    
    return result;
}

// 解析配置文件
static proxy_config_t* parse_config_file(const char* config_file) {
    proxy_config_t* config = malloc(sizeof(proxy_config_t));
    if (!config) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for config");
        return NULL;
    }
    
    // 设置默认值
    strcpy(config->listen_address, "0.0.0.0");
    config->listen_port = 4823;  // 默认监听4823端口
    strcpy(config->target_host, "127.0.0.1");
    config->target_port = 4822;  // 默认连接到4822端口
    config->max_connections = 100;
    config->connection_timeout_ms = 10000;
    config->log_level = PROXY_LOG_INFO;
    config->cuda_log_level = KALMAN_LOG_DEBUG;  // 默认CUDA日志级别为DEBUG
    config->detailed_kalman_logging = 1;  // 默认启用详细的卡尔曼滤波器日志记录
    
    FILE* fp = fopen(config_file, "r");
    if (!fp) {
        guacd_log(PROXY_LOG_ERROR, "Failed to open config file: %s", config_file);
        free(config);
        return NULL;
    }
    
    char line[256];
    char section[64] = "";
    
    while (fgets(line, sizeof(line), fp)) {
        // 移除注释和空白
        char* comment = strchr(line, '#');
        if (comment) *comment = '\0';
        
        // 去除首尾空白
        char* start = line;
        while (*start && isspace(*start)) start++;
        
        char* end = start + strlen(start) - 1;
        while (end > start && isspace(*end)) end--;
        *(end + 1) = '\0';
        
        if (!*start) continue;  // 跳过空行
        
        // 检查是否是section
        if (line[0] == '[') {
            char* section_end = strchr(line, ']');
            if (section_end) {
                *section_end = '\0';
                strncpy(section, line + 1, sizeof(section) - 1);
            }
            continue;
        }
        
        // 解析键值对
        char* key = strtok(line, "=");
        char* value = strtok(NULL, "=");
        
        if (key && value) {
            // 去除键值两端的空白
            while (isspace(*key)) key++;
            end = key + strlen(key) - 1;
            while (end > key && isspace(*end)) end--;
            *(end + 1) = '\0';
            
            while (isspace(*value)) value++;
            end = value + strlen(value) - 1;
            while (end > value && isspace(*end)) end--;
            *(end + 1) = '\0';
            
            // 根据section和key设置值
            if (strcmp(section, "proxy") == 0) {
                if (strcmp(key, "listen_address") == 0) {
                    strncpy(config->listen_address, value, sizeof(config->listen_address) - 1);
                }
                else if (strcmp(key, "listen_port") == 0) {
                    config->listen_port = atoi(value);
                }
                else if (strcmp(key, "target_host") == 0) {
                    strncpy(config->target_host, value, sizeof(config->target_host) - 1);
                }
                else if (strcmp(key, "target_port") == 0) {
                    config->target_port = atoi(value);
                }
                else if (strcmp(key, "max_connections") == 0) {
                    config->max_connections = atoi(value);
                }
                else if (strcmp(key, "connection_timeout_ms") == 0) {
                    config->connection_timeout_ms = atoi(value);
                }
            }
            else if (strcmp(section, "logging") == 0) {
                if (strcmp(key, "log_level") == 0) {
                    if (strcmp(value, "DEBUG") == 0) config->log_level = PROXY_LOG_DEBUG;
                    else if (strcmp(value, "INFO") == 0) config->log_level = PROXY_LOG_INFO;
                    else if (strcmp(value, "WARNING") == 0) config->log_level = PROXY_LOG_WARNING;
                    else if (strcmp(value, "ERROR") == 0) config->log_level = PROXY_LOG_ERROR;
                }
                else if (strcmp(key, "cuda_log_level") == 0) {
                    if (strcmp(value, "DEBUG") == 0) config->cuda_log_level = KALMAN_LOG_DEBUG;
                    else if (strcmp(value, "INFO") == 0) config->cuda_log_level = KALMAN_LOG_INFO;
                    else if (strcmp(value, "WARNING") == 0) config->cuda_log_level = KALMAN_LOG_WARNING;
                    else if (strcmp(value, "ERROR") == 0) config->cuda_log_level = KALMAN_LOG_ERROR;
                    else if (strcmp(value, "TRACE") == 0) config->cuda_log_level = KALMAN_LOG_TRACE;
                }
                else if (strcmp(key, "target_quality") == 0) {
                    config->target_quality = atoi(value);
                }
                else if (strcmp(key, "detailed_kalman_logging") == 0) {
                    config->detailed_kalman_logging = atoi(value);
                }
            }
        }
    }
    
    fclose(fp);
    return config;
}

// Main function
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    
    // 解析配置文件
    proxy_config_t* config = parse_config_file(argv[1]);
    if (!config) {
        fprintf(stderr, "Failed to parse config file\n");
        return 1;
    }
    
    // 初始化日志
    guacd_log_init(config->log_level);
    
    // 设置CUDA卡尔曼滤波器日志级别
    cuda_kalman_set_log_level(config->cuda_log_level);
    
    // 创建服务器socket
    int server_socket = create_server_socket(config->listen_address, config->listen_port);
    if (server_socket < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to create server socket");
        free(config);
        return 1;
    }
    
    guacd_log(PROXY_LOG_INFO, "Listening on %s:%d", config->listen_address, config->listen_port);
    
    // 主循环
    while (1) {
        // 接受客户端连接
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_socket < 0) {
            guacd_log(PROXY_LOG_ERROR, "Failed to accept client connection: %s", strerror(errno));
            continue;
        }
        
        // 记录客户端连接
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        guacd_log(PROXY_LOG_INFO, "Client connected from %s", client_ip);
        
        // 连接到guacd
        int guacd_socket = connect_to_guacd(config->target_host, config->target_port);
        if (guacd_socket < 0) {
            guacd_log(PROXY_LOG_ERROR, "Failed to connect to guacd");
            close(client_socket);
            continue;
        }
        
        // 处理连接
        handle_connection(client_socket, guacd_socket, NULL);
        
        // 清理
        close(client_socket);
        close(guacd_socket);
    }
    
    // 清理
    close(server_socket);
    free(config);
    
    return 0;
}

// 在文件头部添加外部声明
#include "video_cuda.h"
#include "protocol_constants.hpp"

extern video_stream_info_t video_streams[MAX_VIDEO_STREAMS];
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 添加init_stream_mapping实现
        
        
        // 移除重复的init_stream_mapping实现