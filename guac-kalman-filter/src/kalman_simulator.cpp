#include "filter.hpp"
#include "kalman.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <random>
#include <cstring>
#include <chrono>

// 定义STB_IMAGE_IMPLEMENTATION和STB_IMAGE_WRITE_IMPLEMENTATION以包含完整实现
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// 模拟添加噪声的函数
void add_noise(guac::ImageData& image, float noise_level = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, noise_level * 255.0f);
    
    for (size_t i = 0; i < image.data.size(); ++i) {
        float noise = dist(gen);
        int value = static_cast<int>(image.data[i]) + static_cast<int>(noise);
        image.data[i] = static_cast<uint8_t>(std::max(0, std::min(255, value)));
    }
}

// 加载图像的函数
bool load_image(const std::string& filename, guac::ImageData& image) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
    
    if (!data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return false;
    }
    
    image.width = width;
    image.height = height;
    image.channels = 4; // 总是使用RGBA
    image.stride = width * 4;
    image.data.resize(width * height * 4);
    
    // 复制图像数据
    std::memcpy(image.data.data(), data, width * height * 4);
    
    // 释放STB分配的内存
    stbi_image_free(data);
    
    return true;
}

// 保存图像的函数
bool save_image(const std::string& filename, const guac::ImageData& image) {
    // 保存为PNG
    int result = stbi_write_png(
        filename.c_str(),
        image.width,
        image.height,
        image.channels,
        image.data.data(),
        image.stride
    );
    
    return result != 0;
}

// 生成一个测试图像（如果不能加载外部图像）
guac::ImageData generate_test_image(int width, int height) {
    guac::ImageData image;
    image.width = width;
    image.height = height;
    image.channels = 4; // RGBA
    image.stride = width * 4;
    image.data.resize(width * height * 4);
    
    // 创建一个简单的渐变测试图案
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            
            // 创建彩色渐变
            uint8_t r = static_cast<uint8_t>((float)x / width * 255);
            uint8_t g = static_cast<uint8_t>((float)y / height * 255);
            uint8_t b = static_cast<uint8_t>(((float)x + y) / (width + height) * 255);
            
            image.data[idx + 0] = r;
            image.data[idx + 1] = g;
            image.data[idx + 2] = b;
            image.data[idx + 3] = 255; // Alpha
        }
    }
    
    return image;
}

// 创建模拟的Guacamole PNG指令
std::string simulate_guacamole_png_instruction(int stream_index, int layer, int x, int y, const guac::ImageData& image) {
    // 在实际情况下，这里会使用Base64编码图像数据
    // 为简单起见，我们只返回原始数据的大小
    std::string instruction = std::to_string(stream_index) + ".png," +
                             std::to_string(layer) + "," +
                             std::to_string(x) + "," +
                             std::to_string(y) + "," +
                             "base64_data_size=" + std::to_string(image.data.size()) + ";";
    
    return instruction;
}

// 主测试程序
int main(int argc, char* argv[]) {
    try {
        // 解析命令行参数
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " <input_image.png> <output_image.png> [use_cuda=1] [add_noise=1]" << std::endl;
            std::cout << "If no input image is specified or can't be loaded, a test pattern will be generated." << std::endl;
            return 1;
        }
        
        std::string input_file = argv[1];
        std::string output_file = argv[2];
        bool use_cuda = (argc > 3) ? std::stoi(argv[3]) != 0 : true;
        bool add_noise_to_image = (argc > 4) ? std::stoi(argv[4]) != 0 : true;
        
        std::cout << "Kalman Filter Simulator" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Input image: " << input_file << std::endl;
        std::cout << "Output image: " << output_file << std::endl;
        std::cout << "Using CUDA: " << (use_cuda ? "Yes" : "No") << std::endl;
        std::cout << "Add noise: " << (add_noise_to_image ? "Yes" : "No") << std::endl;
        
        // 加载或生成输入图像
        guac::ImageData input_image;
        if (!load_image(input_file, input_image)) {
            std::cout << "Could not load input image. Generating test pattern instead." << std::endl;
            input_image = generate_test_image(640, 480);
        }
        
        std::cout << "Image dimensions: " << input_image.width << "x" << input_image.height 
                  << " (" << input_image.channels << " channels)" << std::endl;
        
        // 保存原始图像的副本（用于比较）
        std::string original_file = output_file.substr(0, output_file.find_last_of('.')) + "_original.png";
        if (save_image(original_file, input_image)) {
            std::cout << "Saved original image to: " << original_file << std::endl;
        }
        
        // 添加噪声（模拟传输过程中的干扰）
        if (add_noise_to_image) {
            add_noise(input_image, 0.1f);
            
            // 保存添加噪声后的图像
            std::string noisy_file = output_file.substr(0, output_file.find_last_of('.')) + "_noisy.png";
            if (save_image(noisy_file, input_image)) {
                std::cout << "Saved noisy image to: " << noisy_file << std::endl;
            }
        }
        
        // 创建Kalman滤波器
        guac::KalmanParams params;
        params.process_noise = 1e-4f;     // 较小的过程噪声（对系统模型更信任）
        params.measurement_noise = 1e-1f; // 较大的测量噪声（对测量值不太信任）
        params.error_cov_post = 1.0f;     // 初始后验估计误差协方差
        
        std::cout << "Creating Kalman filter with parameters:" << std::endl;
        std::cout << "  Process noise: " << params.process_noise << std::endl;
        std::cout << "  Measurement noise: " << params.measurement_noise << std::endl;
        std::cout << "  Error covariance: " << params.error_cov_post << std::endl;
        
        std::unique_ptr<guac::Filter> filter;
        
        if (use_cuda) {
            std::cout << "Using CUDA-accelerated Kalman filter" << std::endl;
            filter = std::make_unique<guac::CudaKalmanFilter>(
                input_image.width, input_image.height, params);
        } else {
            std::cout << "Using CPU Kalman filter" << std::endl;
            filter = std::make_unique<guac::KalmanFilter>(
                input_image.width, input_image.height, params);
        }
        
        // 模拟Guacamole协议场景
        std::cout << "Simulating Guacamole protocol PNG instruction..." << std::endl;
        std::string png_instruction = simulate_guacamole_png_instruction(0, 0, 0, 0, input_image);
        std::cout << "PNG instruction: " << png_instruction << std::endl;
        
        // 创建一个副本用于处理
        guac::ImageData processed_image = input_image;
        
        // 应用Kalman滤波器
        std::cout << "Applying Kalman filter..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        if (!filter->process(processed_image)) {
            std::cerr << "Failed to process image with Kalman filter" << std::endl;
            return 1;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Kalman filter processing completed in " << duration.count() << "ms" << std::endl;
        
        // 保存结果
        std::cout << "Saving processed image to: " << output_file << std::endl;
        if (!save_image(output_file, processed_image)) {
            std::cerr << "Failed to save output image: " << output_file << std::endl;
            return 1;
        }
        
        std::cout << "Done!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
