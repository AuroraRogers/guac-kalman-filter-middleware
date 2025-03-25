#pragma once

#include <cstdint>
#include <memory>
#include <vector>

// 前向声明
struct guac_parser;

namespace guac {

/**
 * 图像数据结构
 */
struct ImageData {
    std::vector<uint8_t> data;  ///< 图像原始数据
    int width;                  ///< 图像宽度
    int height;                 ///< 图像高度
    int stride;                 ///< 每行字节数
    int channels;               ///< 颜色通道数 (通常是 4 - RGBA)
};

/**
 * 滤波器接口
 */
class Filter {
public:
    virtual ~Filter() = default;
    
    /**
     * 处理图像数据
     * 
     * @param image 待处理的图像数据
     * @return 处理成功返回 true，失败返回 false
     */
    virtual bool process(ImageData& image) = 0;
    
    /**
     * 重置滤波器状态
     */
    virtual void reset() = 0;
    
    /**
     * 克隆当前滤波器实例
     * 
     * @return 新的滤波器实例
     */
    virtual std::unique_ptr<Filter> clone() const = 0;
    
    /**
     * 将滤波器应用于Guacamole指令
     * 
     * @param parser Guacamole解析器
     * @return 处理成功返回 true，失败返回 false
     */
    virtual bool apply_to_instruction(guac_parser* parser) = 0;
    
    /**
     * 创建滤波器实例
     * 
     * @param width 初始图像宽度
     * @param height 初始图像高度
     * @param use_cuda 是否使用 CUDA 加速
     * @return 滤波器实例
     */
    static std::unique_ptr<Filter> create(int width, int height, bool use_cuda = true);
};

} // namespace guac
