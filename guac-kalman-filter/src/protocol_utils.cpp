#include "protocol_utils.hpp"
#include <sstream>
#include <cstring>

// 实现指令发送函数
int send_instruction(guac_socket* socket, const char* opcode, int argc, const char** argv) {
    // 首先发送操作码
    int ret = guac_socket_write_string(socket, opcode);
    if (ret)
        return ret;

    // 发送参数
    for (int i = 0; i < argc; i++) {
        // 写入逗号
        ret = guac_socket_write_string(socket, ",");
        if (ret)
            return ret;

        // 写入参数
        ret = guac_socket_write_string(socket, argv[i]);
        if (ret)
            return ret;
    }

    // 写入分号
    ret = guac_socket_write_string(socket, ";");
    if (ret)
        return ret;

    return 0;
}

// 构建并发送一个完整的Guacamole指令，包括格式化
int send_formatted_instruction(guac_socket* socket, const char* opcode, int argc, ...) {
    va_list args;
    va_start(args, argc);
    
    const char* argv[argc];
    for (int i = 0; i < argc; i++)
        argv[i] = va_arg(args, const char*);
    
    va_end(args);
    
    return send_instruction(socket, opcode, argc, argv);
}

// 将一条指令从客户端转发到服务器
int forward_instruction(guac_socket* target_socket, guac_parser* source_parser) {
    if (!target_socket || !source_parser || !source_parser->opcode)
        return -1;
        
    return send_instruction(
        target_socket,
        source_parser->opcode,
        source_parser->argc,
        (const char**)source_parser->argv
    );
}

// 创建一个指令的字符串表示，用于调试
std::string format_instruction_for_debug(const char* opcode, int argc, const char** argv) {
    std::ostringstream debug_stream;
    debug_stream << opcode;
    
    for (int i = 0; i < argc && i < 4; i++) {
        debug_stream << ",";
        if (argv[i] && strlen(argv[i]) < 20) {
            debug_stream << argv[i];
        } else {
            debug_stream << "[data]";
        }
    }
    
    if (argc > 4) {
        debug_stream << ",...";
    }
    
    debug_stream << ";";
    return debug_stream.str();
}
