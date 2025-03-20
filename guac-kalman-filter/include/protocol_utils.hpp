#ifndef PROTOCOL_UTILS_HPP
#define PROTOCOL_UTILS_HPP

#include "guac_c_wrapper.hpp"
#include <string>
#include <cstdarg>

/**
 * 发送一个完整的Guacamole指令到指定的socket
 * 
 * @param socket    Guacamole socket，目标连接
 * @param opcode    指令操作码
 * @param argc      参数数量
 * @param argv      参数数组
 * @return          成功返回0，失败返回非零
 */
int send_instruction(guac_socket* socket, const char* opcode, int argc, const char** argv);

/**
 * 发送一个格式化的Guacamole指令
 * 
 * @param socket    Guacamole socket，目标连接
 * @param opcode    指令操作码
 * @param argc      参数数量
 * @param ...       变长参数列表，每个参数都是const char*
 * @return          成功返回0，失败返回非零
 */
int send_formatted_instruction(guac_socket* socket, const char* opcode, int argc, ...);

/**
 * 将一条指令从客户端转发到服务器
 * 
 * @param target_socket     目标socket
 * @param source_parser     源解析器
 * @return                  成功返回0，失败返回非零
 */
int forward_instruction(guac_socket* target_socket, guac_parser* source_parser);

/**
 * 创建一个指令的字符串表示，用于调试
 * 
 * @param opcode    指令操作码
 * @param argc      参数数量
 * @param argv      参数数组
 * @return          指令的可读字符串表示
 */
std::string format_instruction_for_debug(const char* opcode, int argc, const char** argv);

// 定义额外的状态码，适用于非阻塞读取
#define GUAC_STATUS_NO_INPUT GUAC_STATUS_WOULD_BLOCK

#endif // PROTOCOL_UTILS_HPP
