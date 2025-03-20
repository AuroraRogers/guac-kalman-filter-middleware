#include "protocol.hpp"
#include "protocol_utils.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>  // 添加netdb.h用于getaddrinfo和gethostbyname等
#include <unordered_map> // 添加unordered_map
#include <mutex>    // 添加mutex支持
#include <string>   // 确保string被包含

namespace guac {

// 创建TCP套接字并连接到服务器的辅助函数
static int create_tcp_socket(const std::string& hostname, int port) {
    // 解析主机名
    struct addrinfo hints, *servinfo, *p;
    int rv;
    
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;     // IPv4或IPv6均可
    hints.ai_socktype = SOCK_STREAM; // TCP流套接字
    
    // 将端口转换为字符串
    char port_str[8];
    snprintf(port_str, sizeof(port_str), "%d", port);
    
    if ((rv = getaddrinfo(hostname.c_str(), port_str, &hints, &servinfo)) != 0) {
        std::cerr << "getaddrinfo: " << gai_strerror(rv) << std::endl;
        return -1;
    }
    
    // 尝试连接到可能的每个地址直到成功
    int sockfd;
    for (p = servinfo; p != nullptr; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            std::cerr << "socket: " << strerror(errno) << std::endl;
            continue;
        }
        
        if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockfd);
            std::cerr << "connect: " << strerror(errno) << std::endl;
            continue;
        }
        
        break; // 连接成功
    }
    
    if (p == nullptr) {
        std::cerr << "Failed to connect to " << hostname << ":" << port << std::endl;
        freeaddrinfo(servinfo);
        return -1;
    }
    
    freeaddrinfo(servinfo);
    
    // 设置套接字为非阻塞模式
    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
    
    return sockfd;
}

// 创建监听套接字
static int create_server_socket(const std::string& hostname, int port) {
    // 创建套接字
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        std::cerr << "socket: " << strerror(errno) << std::endl;
        return -1;
    }
    
    // 允许地址重用
    int yes = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == -1) {
        std::cerr << "setsockopt: " << strerror(errno) << std::endl;
        close(sockfd);
        return -1;
    }
    
    // 绑定到指定地址和端口
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    if (hostname == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, hostname.c_str(), &addr.sin_addr);
    }
    
    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        std::cerr << "bind: " << strerror(errno) << std::endl;
        close(sockfd);
        return -1;
    }
    
    // 开始监听
    if (listen(sockfd, 5) == -1) {
        std::cerr << "listen: " << strerror(errno) << std::endl;
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

// 指令处理函数类型定义
typedef int guac_user_callback(guac_user* user, int argc, char** argv);

// 保存原始指令处理函数的函数指针
typedef struct {
    guac_user_callback* func;
    int registered;
} HandlerInfo;

// 指令处理函数映射
static std::unordered_map<std::string, HandlerInfo> instruction_handlers;

// 用户上下文数据
struct UserContext {
    ProtocolHandler* handler;
    std::unique_ptr<Filter> filter;
    bool png_handler_registered;
    
    UserContext() : handler(nullptr), png_handler_registered(false) {}
};

// 协议处理器实现
struct ProtocolHandler::Impl {
    std::unique_ptr<Filter> filter;
    std::atomic<bool> running{false};
    ProtocolHandler* parent; 
    
    // 服务器套接字
    guac_socket* server_socket = nullptr;
    guac_client* client = nullptr;
    
    // 监听线程
    std::thread listener_thread;
    
    // 用户映射
    std::unordered_map<guac_user*, UserContext*> user_contexts;
    std::mutex user_mutex;
    
    std::string server_host;
    int server_port;
    
    explicit Impl(ProtocolHandler* parent, std::unique_ptr<Filter> filter, const std::string& server_host, int server_port)
        : filter(std::move(filter)), parent(parent), server_host(server_host), server_port(server_port) {
    }
    
    ~Impl() {
        stop();
    }
    
    // 连接处理函数
    static int guac_filter_join_handler(guac_user* user, int /* argc */, char** /* argv */) {
        auto* impl = static_cast<ProtocolHandler::Impl*>(user->client->data);
        
        // 为用户创建上下文
        auto* context = new UserContext();
        context->handler = impl->parent; 
        context->filter = Filter::create(640, 480, false); 
        user->data = context;
        
        {
            std::lock_guard<std::mutex> lock(impl->user_mutex);
            impl->user_contexts[user] = context;
        }
        
        std::cout << "[PROTOCOL] User joined: " << user->user_id << std::endl;
        
        // 注册自定义指令处理函数
        // 注意：在1.5.5中，我们需要使用guac_user_handle_instruction来注册处理函数
        // 这个函数在这个简化实现中被省略了
        
        std::cout << "[PROTOCOL] Registered custom handlers" << std::endl;
        
        return 0;
    }
    
    bool connect_to_server() {
        // 创建套接字
        struct sockaddr_in server_addr;
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        
        if (server_fd < 0) {
            std::cerr << "[PROTOCOL] Socket creation failed: " << strerror(errno) << std::endl;
            return false;
        }
        
        std::cout << "[PROTOCOL-DEBUG] Socket created successfully (fd=" << server_fd << ")" << std::endl;
        
        // 设置服务器地址
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        
        // 解析主机名
        struct hostent* server = gethostbyname(server_host.c_str());
        if (server == nullptr) {
            std::cerr << "[PROTOCOL] Failed to resolve host: " << server_host << std::endl;
            close(server_fd);
            return false;
        }
        
        std::cout << "[PROTOCOL-DEBUG] Host " << server_host << " resolved successfully" << std::endl;
        
        // 复制IP地址
        memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
        
        // 连接到服务器
        std::cout << "[PROTOCOL-DEBUG] Attempting to connect to " << server_host << ":" << server_port << "..." << std::endl;
        if (connect(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "[PROTOCOL] Connection failed: " << strerror(errno) << std::endl;
            close(server_fd);
            return false;
        }
        
        std::cout << "[PROTOCOL-DEBUG] Connected to server successfully" << std::endl;
        
        // 创建guac socket
        server_socket = guac_socket_open(server_fd);
        if (server_socket == nullptr) {
            std::cerr << "[PROTOCOL] Failed to create guac socket" << std::endl;
            close(server_fd);
            return false;
        }
        
        std::cout << "[PROTOCOL-DEBUG] Guac socket created successfully" << std::endl;
        return true;
    }
    
    // 执行Guacamole协议握手
    bool perform_guacd_handshake(guac_socket* socket, const std::string& protocol = "rdp") {
        std::cout << "[PROTOCOL] Starting Guacamole protocol handshake with guacd" << std::endl;
        
        // 确保server_socket已经创建
        if (!server_socket) {
            std::cerr << "[PROTOCOL] Cannot perform handshake: server_socket is null" << std::endl;
            return false;
        }
        
        // 步骤1: 发送"select"指令给guacd，选择要使用的协议
        std::cout << "[PROTOCOL] Sending select command for protocol: " << protocol << std::endl;
        
        // 使用标准API发送select指令
        int select_result = guac_protocol_send_select(server_socket, protocol.c_str());
        if (select_result != 0) {
            std::cerr << "[PROTOCOL] Failed to send select command to guacd: " 
                      << guac_status_string(guac_error) << std::endl;
            return false;
        }
        
        // 确保数据被发送出去
        int flush_result = guac_socket_flush(server_socket);
        if (flush_result != 0) {
            std::cerr << "[PROTOCOL] Failed to flush socket after select command: " 
                      << guac_status_string(guac_error) << std::endl;
            return false;
        }
        
        std::cout << "[PROTOCOL] Select command sent successfully" << std::endl;
        
        // 创建解析器来读取guacd的响应
        guac_parser* handshake_parser = guac_parser_alloc();
        if (!handshake_parser) {
            std::cerr << "[PROTOCOL] Failed to allocate parser for handshake" << std::endl;
            return false;
        }
        
        // 步骤2: 读取guacd发送的"args"指令
        std::cout << "[PROTOCOL] Waiting for args instruction from guacd..." << std::endl;
        // 使用适当的超时值(以微秒为单位，30秒超时)
        const int GUACD_TIMEOUT_USEC = 30000000;
        int result = guac_parser_read(handshake_parser, server_socket, GUACD_TIMEOUT_USEC);
        if (result < 0) {
            std::cerr << "[PROTOCOL] Failed to receive instruction from guacd: " 
                      << guac_status_string(guac_error) << std::endl;
            guac_parser_free(handshake_parser);
            return false;
        }
        
        // 检查收到的是否是"args"指令
        if (strcmp(handshake_parser->opcode, "args") != 0) {
            std::cerr << "[PROTOCOL] Expected 'args' instruction but received: " 
                      << handshake_parser->opcode << std::endl;
            guac_parser_free(handshake_parser);
            return false;
        }
        
        std::cout << "[PROTOCOL] Received args instruction with " << handshake_parser->argc 
                  << " arguments" << std::endl;
        
        // 步骤3: 构建connect命令的参数列表
        std::vector<const char*> connect_args;
        connect_args.reserve(handshake_parser->argc + 1); // +1 for nullptr terminator
        
        // 创建一个静态的参数字符串存储区，避免临时字符串被销毁
        static std::vector<std::string> arg_storage;
        arg_storage.clear();
        arg_storage.reserve(handshake_parser->argc);
        
        // 打印并准备参数
        for (int i = 0; i < handshake_parser->argc; i++) {
            std::cout << "[PROTOCOL] Argument " << i << " required: " 
                      << handshake_parser->argv[i] << std::endl;
            
            std::string arg_value;
            
            // 为每个参数提供合适的值
            if (strcmp(handshake_parser->argv[i], "hostname") == 0) {
                arg_value = "localhost"; // 连接到本地RDP服务器
            }
            else if (strcmp(handshake_parser->argv[i], "port") == 0) {
                arg_value = "3389";  // 标准RDP端口
            }
            else if (strcmp(handshake_parser->argv[i], "username") == 0) {
                arg_value = "guacadmin";  // 用户名
            }
            else if (strcmp(handshake_parser->argv[i], "password") == 0) {
                arg_value = "guacadmin";  // 密码
            }
            else if (strcmp(handshake_parser->argv[i], "ignore-cert") == 0) {
                arg_value = "true";  // 忽略证书警告
            }
            else if (strcmp(handshake_parser->argv[i], "security") == 0) {
                arg_value = "nla";   // 网络级别认证 (可选: rdp, tls, nla, any)
            }
            else if (strcmp(handshake_parser->argv[i], "width") == 0) {
                arg_value = "1024";  // 显示宽度
            }
            else if (strcmp(handshake_parser->argv[i], "height") == 0) {
                arg_value = "768";   // 显示高度
            }
            else if (strcmp(handshake_parser->argv[i], "dpi") == 0) {
                arg_value = "96";    // 显示DPI
            }
            else if (strcmp(handshake_parser->argv[i], "color-depth") == 0) {
                arg_value = "24";    // 颜色深度 (8, 16, 24)
            }
            else if (strcmp(handshake_parser->argv[i], "clipboard-encoding") == 0) {
                arg_value = "UTF-8"; // 剪贴板编码
            }
            else if (strcmp(handshake_parser->argv[i], "disable-audio") == 0) {
                arg_value = "true";  // 禁用音频
            }
            else if (strcmp(handshake_parser->argv[i], "enable-drive") == 0) {
                arg_value = "false"; // 禁用驱动器重定向
            }
            else if (strcmp(handshake_parser->argv[i], "enable-printing") == 0) {
                arg_value = "false"; // 禁用打印功能
            }
            else if (strcmp(handshake_parser->argv[i], "server-layout") == 0) {
                arg_value = "en-us-qwerty"; // 键盘布局
            }
            else if (strcmp(handshake_parser->argv[i], "timezone") == 0) {
                arg_value = "Asia/Shanghai"; // 时区
            }
            else {
                // 其他参数使用默认值（空字符串）
                arg_value = "";
                std::cout << "[PROTOCOL] Using empty value for unknown parameter: " 
                          << handshake_parser->argv[i] << std::endl;
            }
            
            // 保存参数值的字符串，并将其C字符串指针添加到参数列表
            arg_storage.push_back(arg_value);
            connect_args.push_back(arg_storage.back().c_str());
            
            std::cout << "[PROTOCOL-DEBUG] Parameter " << handshake_parser->argv[i] 
                      << " = \"" << arg_value << "\"" << std::endl;
        }
        
        // 添加nullptr作为列表结束标记
        connect_args.push_back(nullptr);
        
        // 发送connect指令
        std::cout << "[PROTOCOL] Sending connect command with " << handshake_parser->argc 
                  << " arguments" << std::endl;
        
        // 使用参数列表发送connect指令
        int connect_result = guac_protocol_send_connect(server_socket, connect_args.data());
        if (connect_result != 0) {
            std::cerr << "[PROTOCOL] Failed to send connect command to guacd: " 
                      << guac_status_string(guac_error) << std::endl;
            guac_parser_free(handshake_parser);
            return false;
        }
        
        // 确保数据被发送出去
        flush_result = guac_socket_flush(server_socket);
        if (flush_result != 0) {
            std::cerr << "[PROTOCOL] Failed to flush socket after connect command: " 
                      << guac_status_string(guac_error) << std::endl;
            guac_parser_free(handshake_parser);
            return false;
        }
        
        std::cout << "[PROTOCOL] Connect command sent successfully" << std::endl;
        
        // 释放资源
        guac_parser_free(handshake_parser);
        
        std::cout << "[PROTOCOL] Handshake with guacd completed successfully" << std::endl;
        return true;
    }
    
    // 启动服务
    bool start(const std::string& listen_host, int listen_port) {
        if (running) {
            std::cerr << "[PROTOCOL] Service already running" << std::endl;
            return false;
        }
        
        try {
            std::cout << "[PROTOCOL] Starting service on " << listen_host << ":" << listen_port << std::endl;
            std::cout << "[PROTOCOL] Forwarding to " << server_host << ":" << server_port << std::endl;
            
            // 创建连接到服务器的套接字
            int socket_fd = create_tcp_socket(server_host, server_port);
            if (socket_fd == -1) {
                throw std::runtime_error("无法创建到服务器的连接");
            }
            
            guac_socket* socket = guac_socket_open(socket_fd);
            if (!socket) {
                close(socket_fd);
                throw std::runtime_error("Failed to open socket to server: " + 
                                          std::string(guac_status_string(guac_error)));
            }
            
            // 创建客户端
            guac_client* client = guac_client_alloc();
            if (!client) {
                guac_socket_free(socket);
                throw std::runtime_error("Failed to allocate client: " + 
                                          std::string(guac_status_string(guac_error)));
            }
            
            // 设置客户端属性
            client->socket = socket;
            client->data = this;
            
            // 设置连接处理函数
            client->join_handler = guac_filter_join_handler;
            
            // 保存客户端引用
            this->client = client;
            
            // 创建监听套接字
            int server_fd = create_server_socket(listen_host, listen_port);
            if (server_fd == -1) {
                guac_client_free(client);
                throw std::runtime_error("无法创建监听套接字");
            }
            
            // 启动监听线程
            running = true;
            listener_thread = std::thread([this, server_fd, client]() {
                std::cout << "[PROTOCOL] Listening thread started" << std::endl;
                
                fd_set read_fds;
                struct timeval tv;
                
                // 主循环
                while (running) {
                    // 设置文件描述符集
                    FD_ZERO(&read_fds);
                    FD_SET(server_fd, &read_fds);
                    
                    // 设置超时为0.5秒
                    tv.tv_sec = 0;
                    tv.tv_usec = 500000;
                    
                    int result = select(server_fd + 1, &read_fds, NULL, NULL, &tv);
                    
                    if (result == -1) {
                        if (errno == EINTR) continue;  // 被信号中断，继续循环
                        std::cerr << "[PROTOCOL] select(): " << strerror(errno) << std::endl;
                        break;
                    }
                    
                    if (result > 0 && FD_ISSET(server_fd, &read_fds)) {
                        // 接受新连接
                        struct sockaddr_in client_addr;
                        socklen_t addr_len = sizeof(client_addr);
                        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
                        
                        if (client_fd == -1) {
                            std::cerr << "[PROTOCOL] accept(): " << strerror(errno) << std::endl;
                            continue;
                        }
                        
                        char client_ip[INET_ADDRSTRLEN];
                        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
                        std::cout << "[PROTOCOL] Accepted connection from " << client_ip << ":" << ntohs(client_addr.sin_port) << std::endl;
                        
                        // 为新连接创建guac_socket
                        guac_socket* client_socket = guac_socket_open(client_fd);
                        if (!client_socket) {
                            std::cerr << "[PROTOCOL] Failed to create guac_socket" << std::endl;
                            close(client_fd);
                            continue;
                        }
                        
                        // 处理新连接 - 解析Guacamole协议指令
                        guac_parser* parser = guac_parser_alloc();
                        if (!parser) {
                            std::cerr << "[PROTOCOL] Failed to create parser" << std::endl;
                            guac_socket_free(client_socket);
                            close(client_fd);
                            continue;
                        }
                        
                        std::cout << "[PROTOCOL] Client socket and parser created successfully" << std::endl;
                        
                        // 打印客户端的连接请求详情
                        std::cout << "[CONNECTION] Handling new client connection from " 
                                  << inet_ntoa(client_addr.sin_addr) << ":" 
                                  << ntohs(client_addr.sin_port) << std::endl;
                        
                        // 连接到guacd服务器
                        if (!connect_to_server()) {
                            std::cerr << "[CONNECTION] Failed to connect to guacd server at " 
                                      << server_host << ":" << server_port << std::endl;
                            close(client_fd);
                            guac_parser_free(parser);
                            guac_socket_free(client_socket);
                            continue;
                        }
                        
                        // 执行Guacamole协议握手
                        if (!perform_guacd_handshake(client_socket)) {
                            std::cerr << "[CONNECTION] Failed to perform handshake with guacd server" << std::endl;
                            close(client_fd);
                            guac_parser_free(parser);
                            guac_socket_free(client_socket);
                            guac_socket_free(server_socket);
                            continue;
                        }
                        
                        std::cout << "[CONNECTION] Successfully connected to guacd server at "
                                  << server_host << ":" << server_port << std::endl;
                        
                        // 实现协议转发逻辑
                        std::cout << "[PROTOCOL] Starting protocol forwarding" << std::endl;
                        
                        // 创建一个线程处理数据转发
                        std::thread forwarding_thread([this, client_socket, parser]() {
                            try {
                                // 假设client是已经连接到服务器的客户端
                                guac_socket* server_socket = this->client->socket;
                                
                                // 不再直接操作client_fd，改用client_socket
                                
                                bool active = true;
                                std::cout << "[PROTOCOL] Forwarding established" << std::endl;
                                
                                // 记录上次检查服务器数据的时间
                                auto last_server_check = std::chrono::steady_clock::now();
                                
                                // 连接初始化标志
                                bool connection_initialized = false;
                                
                                // 为客户端数据读取创建解析器
                                guac_parser* client_parser = guac_parser_alloc();
                                if (!client_parser) {
                                    std::cerr << "[PROTOCOL] Failed to allocate client parser" << std::endl;
                                    return;
                                }
                                
                                // 转发循环
                                while (active && running) {
                                    // 处理来自客户端的指令
                                    int client_result = guac_parser_read(client_parser, client_socket, 50000); // 50毫秒超时
                                    
                                    if (client_result > 0 && client_parser->opcode) {
                                        std::cout << "[PROTOCOL] Client -> Server: " << client_parser->opcode 
                                                << " instruction with " << client_parser->argc << " arguments" << std::endl;
                                        
                                        // 打印完整的指令内容用于调试
                                        std::string debug_instruction = client_parser->opcode;
                                        for (int i = 0; i < client_parser->argc && i < 3; i++) {
                                            debug_instruction += ",";
                                            if (client_parser->argv[i] && strlen(client_parser->argv[i]) < 20) {
                                                debug_instruction += client_parser->argv[i];
                                            } else {
                                                debug_instruction += "[data]";
                                            }
                                        }
                                        if (client_parser->argc > 3) {
                                            debug_instruction += ",...";
                                        }
                                        debug_instruction += ";";
                                        std::cout << "[DETAILED] Client instruction: " << debug_instruction << std::endl;
                                        
                                        // 转发指令到服务器
                                        int send_result = forward_instruction(server_socket, client_parser);
                                        
                                        if (send_result == 0) {
                                            guac_socket_flush(server_socket);
                                        } else {
                                            std::cerr << "[PROTOCOL] Failed to forward instruction to server: " 
                                                    << guac_status_string(guac_error) << std::endl;
                                            break;
                                        }
                                    } else if (client_result < 0 && guac_error != GUAC_STATUS_WOULD_BLOCK && guac_error != GUAC_STATUS_TIMEOUT) {
                                        std::cerr << "[PROTOCOL] Error reading from client: " 
                                                << guac_status_string(guac_error) << std::endl;
                                        break;
                                    }
                                    
                                    // 定期检查服务器数据
                                    auto now = std::chrono::steady_clock::now();
                                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        now - last_server_check).count();
                                    
                                    // 每5毫秒检查一次服务器数据
                                    if (elapsed >= 5) {
                                        last_server_check = now;
                                        
                                        // 尝试读取服务器指令
                                        int read_result = guac_parser_read(parser, server_socket, 0);
                                        
                                        if (read_result > 0 && parser->opcode) {
                                            std::cout << "[PROTOCOL] Received instruction from server: " << parser->opcode 
                                                    << " (argc=" << parser->argc << ")" << std::endl;
                                            
                                            // 打印详细的指令信息用于调试
                                            std::string debug_instruction = parser->opcode;
                                            for (int i = 0; i < parser->argc && i < 3; i++) {
                                                debug_instruction += ",";
                                                if (parser->argv[i] && strlen(parser->argv[i]) < 20) {
                                                    debug_instruction += parser->argv[i];
                                                } else {
                                                    debug_instruction += "[data]";
                                                }
                                            }
                                            if (parser->argc > 3) {
                                                debug_instruction += ",...";
                                            }
                                            debug_instruction += ";";
                                            std::cout << "[DETAILED] Server instruction: " << debug_instruction << std::endl;
                                            
                                            // 处理图像指令 - 不直接修改parser->argv，而是创建新的指令
                                            if (strcmp(parser->opcode, "png") == 0 && parser->argc >= 5) {
                                                std::cout << "[PROTOCOL] Processing PNG image instruction" << std::endl;
                                                
                                                // 创建参数副本以保持安全
                                                std::vector<std::string> arg_copies;
                                                std::vector<const char*> arg_ptrs;
                                                
                                                for (int i = 0; i < parser->argc; i++) {
                                                    arg_copies.push_back(parser->argv[i]);
                                                    arg_ptrs.push_back(arg_copies.back().c_str());
                                                }
                                                
                                                // 使用Guacamole API发送指令
                                                int send_result = forward_instruction(
                                                    client_socket,
                                                    parser
                                                );
                                                
                                                if (send_result != 0) {
                                                    std::cerr << "[PROTOCOL] Failed to send PNG instruction to client: " 
                                                            << guac_status_string(guac_error) << std::endl;
                                                } else {
                                                    guac_socket_flush(client_socket);
                                                }
                                            } else {
                                                // 转发其他指令
                                                int send_result = forward_instruction(
                                                    client_socket,
                                                    parser
                                                );
                                                
                                                if (send_result != 0) {
                                                    std::cerr << "[PROTOCOL] Failed to forward instruction to client: " 
                                                            << guac_status_string(guac_error) << std::endl;
                                                } else {
                                                    guac_socket_flush(client_socket);
                                                }
                                            }
                                            
                                            // 连接初始化阶段的特殊处理
                                            if (!connection_initialized) {
                                                if (strcmp(parser->opcode, "ready") == 0) {
                                                    connection_initialized = true;
                                                    std::cout << "[PROTOCOL] Connection fully initialized!" << std::endl;
                                                }
                                            }
                                        } else if (read_result < 0) {
                                            if (guac_error == GUAC_STATUS_WOULD_BLOCK || guac_error == GUAC_STATUS_TIMEOUT) {
                                                // 这些不是错误，只是暂时没有数据
                                            } else {
                                                std::cerr << "[PROTOCOL] Error reading from server: " 
                                                        << guac_status_string(guac_error) << std::endl;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                // 清理资源
                                guac_parser_free(client_parser);
                                std::cout << "[PROTOCOL] Forwarding ended" << std::endl;
                            }
                            catch (const std::exception& e) {
                                std::cerr << "[PROTOCOL] Forwarding error: " << e.what() << std::endl;
                            }
                            
                            // 清理资源
                            guac_parser_free(parser);
                            guac_socket_free(client_socket);
                        });
                        
                        // 分离线程以允许独立运行
                        forwarding_thread.detach();
                    }
                }
                
                // 关闭监听套接字
                close(server_fd);
                std::cout << "[PROTOCOL] Listening thread terminated" << std::endl;
                running = false;
            });
            
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[PROTOCOL] Failed to start service: " << e.what() << std::endl;
            stop();
            return false;
        }
    }
    
    // 停止服务
    void stop() {
        if (!running)
            return;
        
        std::cout << "[PROTOCOL] Stopping service" << std::endl;
        running = false;
        
        // 等待监听线程结束
        if (listener_thread.joinable())
            listener_thread.join();
        
        // 清理用户上下文
        {
            std::lock_guard<std::mutex> lock(user_mutex);
            for (auto& pair : user_contexts) {
                delete pair.second;
            }
            user_contexts.clear();
        }
        
        // 释放客户端和套接字
        if (client) {
            guac_client_free(client);
            client = nullptr;
        }
        
        std::cout << "[PROTOCOL] Service stopped" << std::endl;
    }
};

// 公共接口实现
ProtocolHandler::ProtocolHandler(std::unique_ptr<Filter> filter, const std::string& server_host, int server_port)
    : pImpl(std::make_unique<Impl>(this, std::move(filter), server_host, server_port)) {
}

ProtocolHandler::~ProtocolHandler() = default;

bool ProtocolHandler::start(const std::string& listen_host, int listen_port) {
    return pImpl->start(listen_host, listen_port);
}

void ProtocolHandler::stop() {
    pImpl->stop();
}

bool ProtocolHandler::is_running() const {
    return pImpl->running;
}

} // namespace guac
