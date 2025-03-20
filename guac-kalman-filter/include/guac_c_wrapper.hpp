#ifndef GUAC_C_WRAPPER_HPP
#define GUAC_C_WRAPPER_HPP

// 这个头文件包装了 C 库 libguac 的头文件，确保正确处理 C/C++ ABI 兼容性

#ifdef __cplusplus
extern "C" {
#endif

// 包含 guacamole C 库头文件
#include <guacamole/client.h>
#include <guacamole/protocol.h>
#include <guacamole/socket.h>
#include <guacamole/user.h>
#include <guacamole/parser.h>
#include <guacamole/error.h>

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // GUAC_C_WRAPPER_HPP
