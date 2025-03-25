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

#include <guacamole/client.h>
#include <guacamole/error.h>
#include <guacamole/protocol.h>
#include <guacamole/socket.h>
#include <guacamole/timestamp.h>

#include "kalman_cuda.h"
#include "video_cuda.h"

// Forward declarations
typedef struct guac_kalman_filter guac_kalman_filter;
typedef struct guac_instruction guac_instruction;

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
static int handle_connection(int client_fd, int guacd_fd);
static void guac_kalman_filter_alloc_and_init(guac_kalman_filter* filter, int socket);
static void guac_kalman_filter_free(guac_kalman_filter* filter);
static int process_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_image_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_video_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_select_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_copy_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_end_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_draw_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static bool cuda_kalman_init(guac_kalman_filter* filter);
static bool cuda_kalman_wrapper_update(guac_kalman_filter* filter, double measurement);
static guac_instruction* parse_instruction(const char* buffer);
static void free_instruction(guac_instruction* instruction);
static uint64_t get_timestamp_us(void);
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
static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
}

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
    free(filter);
}

// Process an image instruction
static int process_image_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // Check if this is a valid image instruction
    if (strcmp(instruction->opcode, "img") != 0 || instruction->argc < 5) {
        return 0;
    }
    
    // 记录开始处理图像指令
    guacd_log(PROXY_LOG_DEBUG, "处理图像指令: %s", instruction->opcode);
    
    // Extract parameters
    int layer_index = atoi(instruction->argv[0]);
    int x = atoi(instruction->argv[1]);
    int y = atoi(instruction->argv[2]);
    
    // 记录图像指令的详细参数
    guacd_log(PROXY_LOG_DEBUG, "图像参数: layer=%d, x=%d, y=%d", layer_index, x, y);
    
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
                
                // 记录到CSV文件
                FILE* fp = fopen("image_kalman_metrics.csv", "a");
                if (fp) {
                    // 如果文件为空，添加标题行
                    fseek(fp, 0, SEEK_END);
                    if (ftell(fp) == 0) {
                        fprintf(fp, "timestamp,region,original_position,filtered_position,difference,improvement_percent\n");
                    }
                    
                    // 添加数据行
                    fprintf(fp, "%llu,%d,%.6f,%.6f,%.6f,%.2f\n", 
                            (unsigned long long)get_timestamp_us(), region_index, measurement, filter->state[0], 
                            filter->state[0] - measurement, improvement_percent);
                    
                    fclose(fp);
                }
            }
        }
        
        filter->frequency_stats[region_index].last_update = now;
        // Remove the region_index assignment as this field doesn't exist
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
static int process_draw_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
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
static int process_video_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // Check if this is a valid video instruction
    if (strcmp(instruction->opcode, "video") != 0 || instruction->argc < 3) {
        return 0;
    }
    
    // 记录开始处理视频指令
    guacd_log(PROXY_LOG_DEBUG, "处理视频指令: %s", instruction->opcode);
    
    // Extract parameters
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
            // 记录滤波前后的对比（使用明显的标记）
            guacd_log(PROXY_LOG_INFO, "[视频流卡尔曼滤波] ===== 原始带宽: %.2f kbps, 滤波后带宽: %.2f kbps, 差异: %.2f kbps =====",
                     measurement, filter->state[0], filter->state[0] - measurement);
            
            // 记录滤波器状态变化
            guacd_log(PROXY_LOG_DEBUG, "[视频流状态变化] 滤波前: [%.2f, %.2f, %.2f, %.2f], 滤波后: [%.2f, %.2f, %.2f, %.2f]",
                     original_state[0], original_state[1], original_state[2], original_state[3],
                     filter->state[0], filter->state[1], filter->state[2], filter->state[3]);
            
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
                    fprintf(fp, "timestamp,original_bandwidth,filtered_bandwidth,difference,improvement_percent\n");
                }
                
                // 添加数据行
                fprintf(fp, "%llu,%.6f,%.6f,%.6f,%.2f\n", 
                        (unsigned long long)get_timestamp_us(), measurement, filter->state[0], 
                        filter->state[0] - measurement, improvement_percent);
                
                fclose(fp);
            }
            
            // 更新带宽预测
            filter->bandwidth_prediction.predicted_bandwidth = filter->state[0];
            filter->bandwidth_prediction.confidence = 0.9 - 0.1 * filter->state[1]; // 使用速度分量作为不确定性
            
            // 根据预测带宽调整视频质量
            if (filter->video_optimization_enabled) {
                int target_quality = filter->target_quality;
                
                // 如果预测带宽低于目标带宽的80%，降低质量
                if (filter->target_bandwidth > 0 && 
                    filter->bandwidth_prediction.predicted_bandwidth < filter->target_bandwidth * 0.8) {
                    target_quality = filter->target_quality - 10;
                    if (target_quality < 30) target_quality = 30; // 最低质量限制
                    
                    guacd_log(PROXY_LOG_INFO, "带宽不足，降低视频质量: %d -> %d", 
                             filter->target_quality, target_quality);
                }
                // 如果预测带宽高于目标带宽的120%，提高质量
                else if (filter->target_bandwidth > 0 && 
                         filter->bandwidth_prediction.predicted_bandwidth > filter->target_bandwidth * 1.2) {
                    target_quality = filter->target_quality + 5;
                    if (target_quality > 95) target_quality = 95; // 最高质量限制
                    
                    guacd_log(PROXY_LOG_INFO, "带宽充足，提高视频质量: %d -> %d", 
                             filter->target_quality, target_quality);
                }
                
                // 应用新的质量设置
                if (target_quality != filter->target_quality) {
                    filter->target_quality = target_quality;
                    guacd_log(PROXY_LOG_INFO, "更新视频质量目标: %d", filter->target_quality);
                }
            }
        }
    }
    
    filter->bandwidth_prediction.last_update = now;
    
    // 使用CUDA处理视频指令
    if (filter->video_optimization_enabled) {
        // 调用CUDA视频处理函数
        guacd_log(PROXY_LOG_INFO, "使用CUDA处理视频指令: stream=%d, layer=%d, mimetype=%s",
                 stream_id, layer_id, mimetype);
        
        // 调用新实现的CUDA视频处理函数
        if (cuda_process_video_instruction(filter, stream_id, layer_id, mimetype)) {
            guacd_log(PROXY_LOG_INFO, "CUDA视频处理成功应用");
        } else {
            guacd_log(PROXY_LOG_WARNING, "CUDA视频处理失败");
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
static int process_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
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
    if (strcmp(instruction->opcode, "img") == 0) {
        return process_image_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "video") == 0) {
        return process_video_instruction(filter, instruction);
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
        return process_draw_instruction(filter, instruction);
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
        guacd_log(PROXY_LOG_DEBUG, "未处理的指令类型: %s", instruction->opcode);
    }
    
    return 0;
}

// Handle a client connection
static int handle_connection(int client_fd, int guacd_fd) {
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
                process_instruction(filter, instruction);
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
            bytes_read = read(guacd_fd, buffer, sizeof(buffer));
            
            if (bytes_read <= 0) {
                if (bytes_read < 0) {
                    guacd_log(PROXY_LOG_ERROR, "Error reading from guacd: %s", strerror(errno));
                } else {
                    guacd_log(PROXY_LOG_INFO, "guacd disconnected");
                }
                break;
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
        handle_connection(client_socket, guacd_socket);
        
        // 清理
        close(client_socket);
        close(guacd_socket);
    }
    
    // 清理
    close(server_socket);
    free(config);
    
    return 0;
}