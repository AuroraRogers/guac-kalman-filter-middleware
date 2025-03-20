#include "config.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* 引入Guacamole相关头文件 */
#include <guacamole/client.h>
#include <guacamole/socket.h>
#include <guacamole/protocol.h>

// 简单编译测试函数
int main() {
    printf("Hello from Kalman Filter Proxy!\n");
    printf("Config test: %s\n", VERSION);
    
    // 显示一些基本信息
    printf("测试指令示例:\n");
    printf("  SELECT: 6.select,rdp;\n");
    printf("  CONNECT: 7.connect,rdp,hostname,port;\n");
    
    return 0;
} 