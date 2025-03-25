# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# 设置语言标准在设置编译器之前
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 检查是否在Windows上使用WSL
if(WIN32 AND EXISTS "/usr/bin/g++")
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CMAKE_C_COMPILER /usr/bin/gcc)
    message(STATUS "Using WSL compilers")
endif()

# Project name and version - 注意要在编译器设置后声明项目
project(guac-kalman-filter VERSION 0.1 LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 检测系统特性
include(CheckIncludeFile)
CHECK_INCLUDE_FILE("netinet/in.h" HAVE_NETINET_IN_H)
CHECK_INCLUDE_FILE("sys/socket.h" HAVE_SYS_SOCKET_H)
CHECK_INCLUDE_FILE("fcntl.h" HAVE_FCNTL_H)
CHECK_INCLUDE_FILE("arpa/inet.h" HAVE_ARPA_INET_H)
CHECK_INCLUDE_FILE("unistd.h" HAVE_UNISTD_H)
CHECK_INCLUDE_FILE("pthread.h" HAVE_PTHREAD_H)

# 创建配置文件内容
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/config.h.in "
#ifndef CONFIG_H
#define CONFIG_H

/* 版本信息 */
#define VERSION \"@PROJECT_VERSION@\"
#define PACKAGE_VERSION \"@PROJECT_VERSION@\"

/* 系统头文件检测 */
#cmakedefine HAVE_NETINET_IN_H
#cmakedefine HAVE_SYS_SOCKET_H
#cmakedefine HAVE_FCNTL_H
#cmakedefine HAVE_ARPA_INET_H
#cmakedefine HAVE_UNISTD_H
#cmakedefine HAVE_PTHREAD_H

#endif /* CONFIG_H */
")

# 配置生成config.h文件
configure_file(
    ${CMAKE_CURRENT_BINARY_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

# 确保编译器能找到生成的config.h
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Find required dependencies
find_package(PkgConfig REQUIRED)
# 手动指定 libguac 路径
set(LIBGUAC_INCLUDE_DIRS /usr/local/include)
set(LIBGUAC_LIBRARIES /usr/local/lib/libguac.so)

# 检查是否找到库
if(NOT EXISTS "${LIBGUAC_LIBRARIES}")
    message(FATAL_ERROR "libguac.so not found at ${LIBGUAC_LIBRARIES}. Please specify the correct path.")
endif()

# 查找CUDA和线程库
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# 可选的视频处理依赖项
find_package(OpenCV QUIET)
find_package(FFMPEG QUIET COMPONENTS avcodec avformat avutil swscale)

# 检查库
if(OpenCV_FOUND)
    add_definitions(-DHAVE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
endif()

if(FFMPEG_FOUND)
    add_definitions(-DHAVE_FFMPEG)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    message(STATUS "Found FFmpeg: ${FFMPEG_avcodec_VERSION}")
endif()

# 设置包含目录
include_directories(${LIBGUAC_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 添加源文件
set(SOURCES
    src/kalman_proxy.c
    src/kalman_filter.c
    src/kalman_cuda.cu
    src/kalman_cuda_impl.cpp
    src/video_cuda.cu
    src/video_cuda_impl.cpp
)

# 创建共享库
cuda_add_library(guac-kalman-filter SHARED
    ${SOURCES}
)

# 设置库的属性
set_target_properties(guac-kalman-filter PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# 链接库
target_link_libraries(guac-kalman-filter
    ${LIBGUAC_LIBRARIES}
    ${CUDA_LIBRARIES}
    pthread
)

# 设置CUDA编译选项
if(CUDA_FOUND)
    set_target_properties(guac-kalman-filter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # 设置CUDA架构
    set_target_properties(guac-kalman-filter PROPERTIES
        CUDA_ARCHITECTURES "60;61;70;75;80;86"
    )
endif()

# 安装目标
install(TARGETS guac-kalman-filter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY src/
    DESTINATION include/guac-kalman-filter
    FILES_MATCHING PATTERN "*.h"
)

# 创建可执行文件
add_executable(kalman-proxy src/kalman_proxy.c)
target_link_libraries(kalman-proxy
    guac-kalman-filter
    ${LIBGUAC_LIBRARIES}
    ${CUDA_LIBRARIES}
    pthread
)

# 安装可执行文件
install(TARGETS kalman-proxy
    RUNTIME DESTINATION bin
)

# 安装路径
if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()

if(NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR bin)
endif()

if(NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
    set(CMAKE_INSTALL_INCLUDEDIR include)
endif()

if(NOT DEFINED CMAKE_INSTALL_SYSCONFDIR)
    set(CMAKE_INSTALL_SYSCONFDIR etc)
endif()

if(NOT DEFINED CMAKE_INSTALL_DOCDIR)
    set(CMAKE_INSTALL_DOCDIR share/doc/${CMAKE_PROJECT_NAME})
endif()

# 安装配置文件
install(DIRECTORY conf/
    DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/guacamole
    FILES_MATCHING PATTERN "*.conf"
)

# 安装文档
install(DIRECTORY docs/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    FILES_MATCHING PATTERN "*.md"
)

# 安装启动脚本已被注释，因为start-proxy.bat文件已被删除
# install(PROGRAMS start-proxy.bat
#     DESTINATION ${CMAKE_INSTALL_BINDIR}
#)

# 添加测试目标
add_executable(simple_proxy
    src/simple_proxy.c
)

target_link_libraries(simple_proxy PRIVATE 
    ${CMAKE_PROJECT_NAME}
    ${LIBGUAC_LIBRARIES}
    m
    Threads::Threads
)
