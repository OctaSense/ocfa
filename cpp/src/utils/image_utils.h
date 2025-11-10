/**
 * @file image_utils.h
 * @brief Image processing utilities
 */

#ifndef OCFA_IMAGE_UTILS_H
#define OCFA_IMAGE_UTILS_H

#include <cstdint>
#include <vector>

namespace ocfa {
namespace utils {

/**
 * @brief Convert BGR image to RGB
 */
void BGRToRGB(const uint8_t* bgr, uint8_t* rgb, int width, int height);

/**
 * @brief Resize image (simple bilinear interpolation)
 */
void ResizeImage(const uint8_t* src, int src_width, int src_height, int channels,
                 uint8_t* dst, int dst_width, int dst_height);

/**
 * @brief Normalize image and convert to CHW format
 * @param image Input image (HWC format, uint8)
 * @param output Output tensor (CHW format, float32, normalized)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (1 or 3)
 * @param mean Mean values for normalization (per channel)
 * @param std Standard deviation for normalization (per channel)
 */
void NormalizeAndConvertToCHW(const uint8_t* image, float* output,
                              int width, int height, int channels,
                              const float* mean, const float* std);

} // namespace utils
} // namespace ocfa

#endif // OCFA_IMAGE_UTILS_H
