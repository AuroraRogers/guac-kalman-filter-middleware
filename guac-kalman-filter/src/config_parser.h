#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

/**
 * Structure to hold proxy configuration settings
 */
typedef struct {
    // Proxy settings
    char listen_address[64];
    int listen_port;
    char target_host[64];
    int target_port;
    int max_connections;
    int connection_timeout_ms;
    
    // Kalman filter settings
    int kalman_enabled;
    double process_noise;
    double measurement_noise_x;
    double measurement_noise_y;
    char stats_file[256];
    
    // Video optimization settings
    int optimization_enabled;
    int target_quality;
    int target_bandwidth;
    
    // Logging settings
    char log_level[16];
    char log_file[256];
} proxy_config_t;

/**
 * Parse a configuration file and populate the config structure
 * 
 * @param config_file Path to the configuration file
 * @param config Pointer to the configuration structure to populate
 * @return 0 on success, -1 on failure
 */
int parse_config_file(const char* config_file, proxy_config_t* config);

/**
 * Set default configuration values
 * 
 * @param config Pointer to the configuration structure to initialize
 */
void set_default_config(proxy_config_t* config);

#endif /* CONFIG_PARSER_H */