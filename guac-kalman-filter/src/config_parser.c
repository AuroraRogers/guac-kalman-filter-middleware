#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "config_parser.h"

/**
 * Trim whitespace from the beginning and end of a string
 * 
 * @param str String to trim
 * @return Pointer to the trimmed string
 */
static char* trim(char* str) {
    if (!str) return NULL;
    
    // Trim leading whitespace
    while (isspace((unsigned char)*str)) {
        str++;
    }
    
    if (*str == '\0') return str;
    
    // Trim trailing whitespace
    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }
    
    // Null terminate the string
    *(end + 1) = '\0';
    
    return str;
}

/**
 * Set default configuration values
 * 
 * @param config Pointer to the configuration structure to initialize
 */
void set_default_config(proxy_config_t* config) {
    if (!config) return;
    
    // Proxy settings
    strcpy(config->listen_address, "0.0.0.0");
    config->listen_port = 4823;
    strcpy(config->target_host, "127.0.0.1");
    config->target_port = 4822;
    config->max_connections = 100;
    config->connection_timeout_ms = 10000;
    
    // Kalman filter settings
    config->kalman_enabled = 1;
    config->process_noise = 0.01;
    config->measurement_noise_x = 0.1;
    config->measurement_noise_y = 0.1;
    strcpy(config->stats_file, "metrics.csv");
    
    // Video optimization settings
    config->optimization_enabled = 1;
    config->target_quality = 80;
    config->target_bandwidth = 1000000;
    
    // Logging settings
    strcpy(config->log_level, "INFO");
    strcpy(config->log_file, "kalman-proxy.log");
}

/**
 * Parse a configuration file and populate the config structure
 * 
 * @param config_file Path to the configuration file
 * @param config Pointer to the configuration structure to populate
 * @return 0 on success, -1 on failure
 */
int parse_config_file(const char* config_file, proxy_config_t* config) {
    if (!config_file || !config) {
        return -1;
    }
    
    // Set default values
    set_default_config(config);
    
    // Open the configuration file
    FILE* file = fopen(config_file, "r");
    if (!file) {
        fprintf(stderr, "Failed to open configuration file: %s\n", config_file);
        return -1;
    }
    
    char line[1024];
    char section[64] = "";
    
    // Parse the configuration file
    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        char* trimmed = trim(line);
        if (trimmed[0] == '#' || trimmed[0] == '\0' || trimmed[0] == '\n') {
            continue;
        }
        
        // Check for section header
        if (trimmed[0] == '[' && strchr(trimmed, ']')) {
            char* end = strchr(trimmed, ']');
            *end = '\0';
            strcpy(section, trimmed + 1);
            continue;
        }
        
        // Parse key-value pair
        char* equals = strchr(trimmed, '=');
        if (!equals) {
            continue;
        }
        
        *equals = '\0';
        char* key = trim(trimmed);
        char* value = trim(equals + 1);
        
        // Process key-value pair based on section
        if (strcmp(section, "proxy") == 0) {
            if (strcmp(key, "listen_address") == 0) {
                strncpy(config->listen_address, value, sizeof(config->listen_address) - 1);
            } else if (strcmp(key, "listen_port") == 0) {
                config->listen_port = atoi(value);
            } else if (strcmp(key, "target_host") == 0) {
                strncpy(config->target_host, value, sizeof(config->target_host) - 1);
            } else if (strcmp(key, "target_port") == 0) {
                config->target_port = atoi(value);
            } else if (strcmp(key, "max_connections") == 0) {
                config->max_connections = atoi(value);
            } else if (strcmp(key, "connection_timeout_ms") == 0) {
                config->connection_timeout_ms = atoi(value);
            }
        } else if (strcmp(section, "kalman") == 0) {
            if (strcmp(key, "enabled") == 0) {
                config->kalman_enabled = (strcmp(value, "true") == 0 || atoi(value) == 1);
            } else if (strcmp(key, "process_noise") == 0) {
                config->process_noise = atof(value);
            } else if (strcmp(key, "measurement_noise_x") == 0) {
                config->measurement_noise_x = atof(value);
            } else if (strcmp(key, "measurement_noise_y") == 0) {
                config->measurement_noise_y = atof(value);
            } else if (strcmp(key, "stats_file") == 0) {
                strncpy(config->stats_file, value, sizeof(config->stats_file) - 1);
            }
        } else if (strcmp(section, "video") == 0) {
            if (strcmp(key, "optimization_enabled") == 0) {
                config->optimization_enabled = (strcmp(value, "true") == 0 || atoi(value) == 1);
            } else if (strcmp(key, "target_quality") == 0) {
                config->target_quality = atoi(value);
            } else if (strcmp(key, "target_bandwidth") == 0) {
                config->target_bandwidth = atoi(value);
            }
        } else if (strcmp(section, "logging") == 0) {
            if (strcmp(key, "log_level") == 0) {
                strncpy(config->log_level, value, sizeof(config->log_level) - 1);
            } else if (strcmp(key, "log_file") == 0) {
                strncpy(config->log_file, value, sizeof(config->log_file) - 1);
            }
        }
    }
    
    fclose(file);
    return 0;
}