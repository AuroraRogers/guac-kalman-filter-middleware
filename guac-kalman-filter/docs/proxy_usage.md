# Guacamole Kalman Filter 代理模式使用说明

本文档说明如何使用Kalman Filter中间件的代理模式。

## 架构概述

代理模式允许Kalman Filter中间件作为一个独立的服务运行，位于Guacamole客户端和服务器之间：

```
Guacamole客户端 <---> Kalman代理 <---> Guacamole服务器
```

代理服务拦截所有通信，应用滤波和优化，然后转发到真实的Guacamole服务器。这种方式无需修改现有的Guacamole安装。

## 安装步骤

### 1. 构建Kalman Filter中间件

```bash
cd guac-kalman-filter
mkdir build && cd build
cmake ..
make
sudo make install
```

安装后，以下文件会被创建：
- 库文件: `/usr/local/lib/libguac-kalman-filter.so`
- 代理程序: `/usr/local/bin/guac-kalman-proxy`
- 配置文件: `/etc/guacamole/kalman-proxy.conf`

### 2. 配置代理服务

编辑配置文件 `/etc/guacamole/kalman-proxy.conf`（或使用本地配置文件）：

```ini
[proxy]
# 监听端口（Guacamole客户端将连接到这里）
listen_address = 0.0.0.0
listen_port = 4823

# 实际的Guacamole服务器地址
target_host = localhost
target_port = 4822
```

> **注意**: 代理使用端口4823（标准Guacamole端口），而实际的Guacamole服务器应该配置为使用不同的端口（这里是4822）。

### 3. 修改Guacamole服务器配置

编辑Guacamole服务器配置（通常是`/etc/guacamole/guacd.conf`）来更改端口：

```ini
[server]
bind_host = 127.0.0.1
bind_port = 4823
```

此配置使Guacamole服务器只监听本地连接，且使用非标准端口，确保所有连接通过代理。

### 4. 启动服务

首先启动修改后的Guacamole服务器：

```bash
guacd -c /etc/guacamole/guacd.conf
```

然后启动Kalman代理：

```bash
guac-kalman-proxy /etc/guacamole/kalman-proxy.conf
```

## 客户端配置

Guacamole客户端（通常是Web应用）不需要任何修改。只需确保它配置为连接到代理的地址和端口，而不是直接连接到Guacamole服务器。

## 故障排除

### 常见问题

1. **连接被拒绝**：确保代理服务正在运行且监听在正确的端口上。

   ```bash
   netstat -tuln | grep 4822
   ```

2. **连接超时**：确保实际的Guacamole服务器正在运行。

   ```bash
   netstat -tuln | grep 4823
   ```

3. **视频质量问题**：调整配置文件中的`[video]`部分参数。

### 日志查看

查看代理日志：

```bash
tail -f kalman-proxy.log
```

## 性能监控

启用统计功能后，代理会生成CSV格式的统计数据文件（由`stats_file`参数指定）。可以使用电子表格软件或数据分析工具查看这些统计数据。

## 安全考虑

- 代理服务默认监听所有网络接口。在生产环境中，应该限制为只监听必要的网络接口。
- 考虑在代理和Guacamole服务器之间配置SSL/TLS以加密通信。
