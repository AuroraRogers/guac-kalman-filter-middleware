// stb_image.h - v2.28 - public domain image loader
// 注：这是一个占位文件。在实际使用中，您应该获取完整的stb_image.h库。
// 下载地址：https://github.com/nothings/stb/blob/master/stb_image.h

#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STBI_INCLUDE_STB_IMAGE_H

// 这里只包含简单的接口声明
// 实际项目中请下载完整的stb_image.h

typedef unsigned char stbi_uc;
typedef unsigned short stbi_us;

#ifdef __cplusplus
extern "C" {
#endif

// 主要的图像加载函数
extern stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
extern void stbi_image_free(void *retval_from_stbi_load);

#ifdef __cplusplus
}
#endif

#endif // STBI_INCLUDE_STB_IMAGE_H
