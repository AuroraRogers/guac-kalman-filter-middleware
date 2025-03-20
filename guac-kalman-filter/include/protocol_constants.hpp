#pragma once

namespace guac {

// 定义超时常量（微秒）
constexpr int GUACD_TIMEOUT_USEC = 1000000;

// 定义RDP连接参数（NULL标记参数结束）
constexpr const char* GUAC_RDP_ARGS[] = { NULL };

} // namespace guac
