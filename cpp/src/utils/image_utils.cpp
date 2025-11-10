/**
 * @file image_utils.cpp
 * @brief Image processing utilities implementation
 */

#include "image_utils.h"
#include <cstring>
#include <algorithm>

namespace ocfa {
namespace utils {

void BGRToRGB(const uint8_t* bgr, uint8_t* rgb, int width, int height) {
    int total = width * height;
    for (int i = 0; i < total; ++i) {
        rgb[i * 3 + 0] = bgr[i * 3 + 2]; // R = B
        rgb[i * 3 + 1] = bgr[i * 3 + 1]; // G = G
        rgb[i * 3 + 2] = bgr[i * 3 + 0]; // B = R
    }
}

void ResizeImage(const uint8_t* src, int src_width, int src_height, int channels,
                 uint8_t* dst, int dst_width, int dst_height) {
    float x_ratio = static_cast<float>(src_width) / dst_width;
    float y_ratio = static_cast<float>(src_height) / dst_height;

    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            // Bilinear interpolation
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x1 = static_cast<int>(src_x);
            int y1 = static_cast<int>(src_y);
            int x2 = std::min(x1 + 1, src_width - 1);
            int y2 = std::min(y1 + 1, src_height - 1);

            float dx = src_x - x1;
            float dy = src_y - y1;

            for (int c = 0; c < channels; ++c) {
                float val =
                    (1 - dx) * (1 - dy) * src[(y1 * src_width + x1) * channels + c] +
                    dx * (1 - dy) * src[(y1 * src_width + x2) * channels + c] +
                    (1 - dx) * dy * src[(y2 * src_width + x1) * channels + c] +
                    dx * dy * src[(y2 * src_width + x2) * channels + c];

                dst[(y * dst_width + x) * channels + c] = static_cast<uint8_t>(val + 0.5f);
            }
        }
    }
}

void NormalizeAndConvertToCHW(const uint8_t* image, float* output,
                              int width, int height, int channels,
                              const float* mean, const float* std) {
    int total_pixels = width * height;

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int hwc_idx = (h * width + w) * channels + c;
                int chw_idx = c * total_pixels + h * width + w;

                float pixel = static_cast<float>(image[hwc_idx]);
                output[chw_idx] = (pixel / 255.0f - mean[c]) / std[c];
            }
        }
    }
}

} // namespace utils
} // namespace ocfa
