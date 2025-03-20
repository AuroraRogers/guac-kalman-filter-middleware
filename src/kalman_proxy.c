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

#include <guacamole/client.h>
#include <guacamole/error.h>
#include <guacamole/protocol.h>
#include <guacamole/socket.h>
#include <guacamole/timestamp.h>

// CUDA support
#ifndef CUDA_DISABLED
extern int cuda_kalman_init(void* filter);
extern int cuda_kalman_update(void* filter, double measurement);
#endif

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

// Structure definitions
typedef struct {
    double last_update;
    double bandwidth_estimate;
    double process_noise;
    double measurement_noise;
    double error_covariance;
} bandwidth_prediction_t;

typedef struct {
    int layer_index;
    int priority;
} layer_priority_t;

typedef struct {
    int layer_index;
    int depends_on;
} layer_dependency_t;

typedef struct {
    int region_index;
    double last_update;
    double update_interval;
    double update_frequency;
} update_frequency_stats_t;

// Kalman filter structure
struct guac_kalman_filter {
    int socket;
    int max_layers;
    int max_regions;
    layer_priority_t* layer_priorities;
    layer_dependency_t* layer_dependencies;
    update_frequency_stats_t* frequency_stats;
    bandwidth_prediction_t bandwidth_prediction;
    int use_cuda;
};

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
static guac_kalman_filter* guac_kalman_filter_init(int socket);
static void guac_kalman_filter_free(guac_kalman_filter* filter);
static int process_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_image_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_select_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_copy_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int process_end_instruction(guac_kalman_filter* filter, guac_instruction* instruction);
static int cuda_kalman_init(guac_kalman_filter* filter);
static int cuda_kalman_update(guac_kalman_filter* filter, double measurement);
static guac_instruction* parse_instruction(const char* buffer);
static void free_instruction(guac_instruction* instruction);
static uint64_t get_timestamp_us(void);

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
        guacd_log(PROXY_LOG_WARNING, "Failed to set socket options: %s", strerror(errno));
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
static guac_kalman_filter* guac_kalman_filter_init(int socket) {
    guac_kalman_filter* filter = malloc(sizeof(guac_kalman_filter));
    if (!filter) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for Kalman filter");
        return NULL;
    }
    
    // Initialize filter
    filter->socket = socket;
    filter->max_layers = 10;
    filter->max_regions = 100;
    filter->use_cuda = 1;
    
    // Allocate memory for layer priorities and dependencies
    filter->layer_priorities = calloc(filter->max_layers, sizeof(layer_priority_t));
    filter->layer_dependencies = calloc(filter->max_layers, sizeof(layer_dependency_t));
    
    // Allocate memory for frequency stats
    filter->frequency_stats = calloc(filter->max_regions, sizeof(update_frequency_stats_t));
    
    // Initialize bandwidth prediction
    filter->bandwidth_prediction.last_update = get_timestamp_us();
    filter->bandwidth_prediction.bandwidth_estimate = 1000000; // 1 Mbps initial estimate
    filter->bandwidth_prediction.process_noise = 0.01;
    filter->bandwidth_prediction.measurement_noise = 0.1;
    filter->bandwidth_prediction.error_covariance = 1.0;
    
    // Check memory allocations
    if (!filter->layer_priorities || !filter->layer_dependencies || !filter->frequency_stats) {
        guacd_log(PROXY_LOG_ERROR, "Failed to allocate memory for Kalman filter data structures");
        guac_kalman_filter_free(filter);
        return NULL;
    }
    
    // Initialize CUDA if available
    if (filter->use_cuda) {
        if (!cuda_kalman_init(filter)) {
            guac_kalman_filter_free(filter);
            return NULL;
        }
    }
    
    return filter;
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
    
    // Extract parameters
    int layer_index = atoi(instruction->argv[0]);
    int x = atoi(instruction->argv[1]);
    int y = atoi(instruction->argv[2]);
    
    // Update region statistics
    int region_index = (y / 100) * 10 + (x / 100); // Simple region mapping
    if (region_index < filter->max_regions) {
        uint64_t now = get_timestamp_us();
        double time_diff = (now - filter->frequency_stats[region_index].last_update) / 1000000.0;
        
        if (filter->frequency_stats[region_index].last_update > 0 && time_diff > 0) {
            // Update frequency statistics
            filter->frequency_stats[region_index].update_interval = 
                0.9 * filter->frequency_stats[region_index].update_interval + 0.1 * time_diff;
            filter->frequency_stats[region_index].update_frequency = 
                1.0 / filter->frequency_stats[region_index].update_interval;
        }
        
        filter->frequency_stats[region_index].last_update = now;
        filter->frequency_stats[region_index].region_index = region_index;
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
        filter->layer_priorities[layer_index].layer_index = layer_index;
        filter->layer_priorities[layer_index].priority = 10; // High priority for selected layers
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
        filter->layer_dependencies[dst_layer].layer_index = dst_layer;
        filter->layer_dependencies[dst_layer].depends_on = src_layer;
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
        
        // Update Kalman filter
        if (filter->use_cuda) {
            cuda_kalman_update(filter, measured_bandwidth);
        } else {
            // CPU implementation of Kalman filter
            double k = filter->bandwidth_prediction.error_covariance / 
                       (filter->bandwidth_prediction.error_covariance + filter->bandwidth_prediction.measurement_noise);
            
            filter->bandwidth_prediction.bandwidth_estimate = 
                filter->bandwidth_prediction.bandwidth_estimate + 
                k * (measured_bandwidth - filter->bandwidth_prediction.bandwidth_estimate);
            
            filter->bandwidth_prediction.error_covariance = 
                (1 - k) * filter->bandwidth_prediction.error_covariance + 
                filter->bandwidth_prediction.process_noise;
        }
        
        filter->bandwidth_prediction.last_update = now;
    }
    
    return 0;
}

// Process an instruction
static int process_instruction(guac_kalman_filter* filter, guac_instruction* instruction) {
    if (!filter || !instruction) {
        return -1;
    }
    
    // Process instruction based on opcode
    if (strcmp(instruction->opcode, "img") == 0) {
        return process_image_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "select") == 0) {
        return process_select_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "copy") == 0) {
        return process_copy_instruction(filter, instruction);
    } else if (strcmp(instruction->opcode, "end") == 0) {
        return process_end_instruction(filter, instruction);
    }
    
    return 0;
}

// Handle a client connection
static int handle_connection(int client_fd, int guacd_fd) {
    char buffer[8192];
    ssize_t bytes_read, bytes_written;
    fd_set read_fds;
    struct timeval timeout;
    
    // Initialize Kalman filter
    guac_kalman_filter* filter = guac_kalman_filter_init(client_fd);
    if (!filter) {
        guacd_log(PROXY_LOG_ERROR, "Failed to initialize Kalman filter");
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

// CUDA initialization
static int cuda_kalman_init(guac_kalman_filter* filter) {
    guacd_log(PROXY_LOG_INFO, "Initializing CUDA Kalman filter");
    
#ifndef CUDA_DISABLED
    // Call the CUDA implementation
    return cuda_kalman_init((void*)filter);
#else
    // CUDA is disabled, use CPU implementation
    guacd_log(PROXY_LOG_INFO, "CUDA is disabled, using CPU implementation");
    filter->use_cuda = 0;
    return 1; // Success
#endif
}

// CUDA Kalman filter update
static int cuda_kalman_update(guac_kalman_filter* filter, double measurement) {
#ifndef CUDA_DISABLED
    // Call the CUDA implementation if CUDA is enabled
    if (filter->use_cuda) {
        return cuda_kalman_update((void*)filter, measurement);
    }
#endif

    // CPU implementation of Kalman filter
    double k = filter->bandwidth_prediction.error_covariance / 
               (filter->bandwidth_prediction.error_covariance + filter->bandwidth_prediction.measurement_noise);
    
    filter->bandwidth_prediction.bandwidth_estimate = 
        filter->bandwidth_prediction.bandwidth_estimate + 
        k * (measurement - filter->bandwidth_prediction.bandwidth_estimate);
    
    filter->bandwidth_prediction.error_covariance = 
        (1 - k) * filter->bandwidth_prediction.error_covariance + 
        filter->bandwidth_prediction.process_noise;
    
    return 1; // Success
}

// Main function
int main(int argc, char** argv) {
    // Initialize logging
    guacd_log_init(PROXY_LOG_DEBUG);
    
    // Create server socket
    int server_socket = create_server_socket("0.0.0.0", 4822);
    if (server_socket < 0) {
        guacd_log(PROXY_LOG_ERROR, "Failed to create server socket");
        return 1;
    }
    
    guacd_log(PROXY_LOG_INFO, "Listening on port 4822");
    
    // Main loop
    while (1) {
        // Accept client connection
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_socket < 0) {
            guacd_log(PROXY_LOG_ERROR, "Failed to accept client connection: %s", strerror(errno));
            continue;
        }
        
        // Log client connection
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        guacd_log(PROXY_LOG_INFO, "Client connected from %s", client_ip);
        
        // Connect to guacd
        int guacd_socket = connect_to_guacd("127.0.0.1", 4823);
        if (guacd_socket < 0) {
            guacd_log(PROXY_LOG_ERROR, "Failed to connect to guacd");
            close(client_socket);
            continue;
        }
        
        // Handle connection
        handle_connection(client_socket, guacd_socket);
        
        // Clean up
        close(client_socket);
        close(guacd_socket);
    }
    
    // Clean up
    close(server_socket);
    
    return 0;
}