// stb_image_write.h - v1.16 - public domain image writer
// 注：这是一个占位文件。在实际使用中，您应该获取完整的stb_image_write.h库。
// 下载地址：https://github.com/nothings/stb/blob/master/stb_image_write.h

#ifndef INCLUDE_STB_IMAGE_WRITE_H
#define INCLUDE_STB_IMAGE_WRITE_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// 主要的图像保存功能
extern int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
extern int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_STB_IMAGE_WRITE_H
