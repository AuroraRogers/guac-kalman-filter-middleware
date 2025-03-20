#pragma once

#include <memory>
#include <string>
#include <thread>
#include <atomic>

#include "filter.hpp"

namespace guac {

/**
 * Guacamole 协议处理器
 * 作为中间层，拦截并处理图像数据
 */
class ProtocolHandler {
public:
    /**
     * 构造函数
     * 
     * @param filter 用于处理图像的滤波器
     * @param server_host 上游 guacd 服务器地址
     * @param server_port 上游 guacd 服务器端口
     */
    ProtocolHandler(std::unique_ptr<Filter> filter, const std::string& server_host, int server_port);
    
    /**
     * 析构函数
     */
    ~ProtocolHandler();
    
    /**
     * 启动协议处理服务
     * 
     * @param listen_host 监听主机地址
     * @param listen_port 监听端口
     * @return 启动成功返回 true，失败返回 false
     */
    bool start(const std::string& listen_host, int listen_port);
    
    /**
     * 停止协议处理服务
     */
    void stop();
    
    /**
     * 获取服务运行状态
     * 
     * @return 服务正在运行返回 true，否则返回 false
     */
    bool is_running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace guac
