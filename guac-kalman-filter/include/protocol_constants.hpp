#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 指令类型常量
#define GUAC_INSTRUCTION_MOUSE 4
#define GUAC_INSTRUCTION_VIDEO 5
#define GUAC_INSTRUCTION_IMAGE 3
#define GUAC_INSTRUCTION_BLOB 6
#define GUAC_BLOB_STREAM_HEADER 0x42
#define GUAC_BLOB_STREAM_DATA  0x44

// 定义超时时常量（微秒）
#define GUACD_TIMEOUT_USEC 1000000

// 定义RDP连接参数（NULL标记参数结束）
static const char* GUAC_RDP_ARGS[] = { NULL };

#ifdef __cplusplus
}
#endif
