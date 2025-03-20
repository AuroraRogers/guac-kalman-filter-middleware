#include "filter.hpp"
#include "protocol.hpp"

#include <iostream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <thread>
#include <atomic>

// 全局协议处理器实例
std::unique_ptr<guac::ProtocolHandler> g_handler;
std::atomic<bool> g_running(true);

// 信号处理函数
void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
    
    if (g_handler) {
        g_handler->stop();
    }
}

// 打印使用帮助
void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  --listen-host HOST     Host to listen on (default: 0.0.0.0)\n"
              << "  --listen-port PORT     Port to listen on (default: 4823)\n"
              << "  --server-host HOST     Guacamole server host (default: localhost)\n"
              << "  --server-port PORT     Guacamole server port (default: 4822)\n"
              << "  --width WIDTH          Initial width for filter (default: 1920)\n"
              << "  --height HEIGHT        Initial height for filter (default: 1080)\n"
              << "  --cpu-only             Disable CUDA acceleration\n"
              << "  --help                 Display this help and exit\n";
}

int main(int argc, char* argv[]) {
    // 默认参数
    std::string listen_host = "0.0.0.0";
    int listen_port = 4823;
    std::string server_host = "localhost";
    int server_port = 4822;
    int width = 1920;
    int height = 1080;
    bool use_cuda = true;
    bool use_ssl = false;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--listen-host" && i + 1 < argc) {
            listen_host = argv[++i];
        } else if (arg == "--listen-port" && i + 1 < argc) {
            listen_port = std::atoi(argv[++i]);
        } else if (arg == "--server-host" && i + 1 < argc) {
            server_host = argv[++i];
        } else if (arg == "--server-port" && i + 1 < argc) {
            server_port = std::atoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        } else if (arg == "--cpu-only") {
            use_cuda = false;
        } else if (arg == "--ssl") {
            std::cout << "Note: SSL option is deprecated and will be ignored" << std::endl;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // 设置信号处理
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    try {
        std::cout << "Guacamole Kalman Filter Proxy" << std::endl;
        std::cout << "=============================" << std::endl;
        
        // 创建滤波器
        std::cout << "Creating filter..." << std::endl;
        auto filter = guac::Filter::create(width, height, use_cuda);
        
        // 创建协议处理器
        std::cout << "Creating protocol handler..." << std::endl;
        g_handler = std::make_unique<guac::ProtocolHandler>(std::move(filter), server_host, server_port);
        
        // 启动服务
        std::cout << "Starting service..." << std::endl;
        if (!g_handler->start(listen_host, listen_port)) {
            std::cerr << "Failed to start service" << std::endl;
            return 1;
        }
        
        std::cout << "Service started successfully" << std::endl;
        std::cout << "Listening on " << listen_host << ":" << listen_port << std::endl;
        std::cout << "Forwarding to " << server_host << ":" << server_port << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        // 主循环
        while (g_running && g_handler->is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // 停止服务
        std::cout << "Stopping service..." << std::endl;
        g_handler->stop();
        g_handler.reset();
        
        std::cout << "Service stopped" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
