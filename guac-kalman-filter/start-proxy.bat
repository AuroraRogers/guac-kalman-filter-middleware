@echo off
REM 启动 Guacamole Kalman Filter 代理服务

echo 正在启动 Guacamole Kalman Filter 代理服务...

REM 设置可执行文件路径
set PROXY_EXE=.\build\guac-kalman-proxy.exe
set CONFIG_FILE=.\conf\kalman-proxy.conf

REM 检查可执行文件是否存在
if not exist %PROXY_EXE% (
    echo 错误: 代理可执行文件不存在!
    echo 请先构建代理程序：
    echo mkdir build
    echo cd build
    echo cmake ..
    echo cmake --build .
    exit /b 1
)

REM 检查配置文件是否存在
if not exist %CONFIG_FILE% (
    echo 错误: 配置文件不存在!
    echo 请确认 %CONFIG_FILE% 文件存在
    exit /b 1
)

echo 使用配置文件: %CONFIG_FILE%
echo.
echo 代理服务运行中...按 Ctrl+C 停止.
echo.

REM 启动代理服务
%PROXY_EXE% %CONFIG_FILE%

echo 代理服务已停止.
exit /b 0
