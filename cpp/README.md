# OCFA Face SDK - C++ Implementation

C++ implementation of OCFA Face Recognition SDK for embedded Linux platforms.

## Features

- RGB-IR dual-modal liveness detection
- Face feature extraction (512-dim ArcFace)
- 1:1 and 1:N face matching
- In-memory feature database
- ONNX Runtime and NNIE inference engine support
- Optimized for Hi3516CV610 (ARM A17, NNIE NPU)

## Architecture

```
cpp/
├── include/              # Public API headers
│   ├── ocfa_face_sdk.h  # Main SDK API
│   ├── ocfa_types.h     # Data structures
│   └── ocfa_errors.h    # Error codes
├── src/
│   ├── core/            # Core SDK implementation
│   ├── inference/       # Inference engines (ONNX/NNIE)
│   └── utils/           # Image/math utilities
└── examples/            # Usage examples
```

## Dependencies

### Required
- C++17 compiler (GCC 7+ or Clang 5+)
- CMake 3.16+
- pthread

### Optional (choose one)
- **ONNX Runtime** (for x86/ARM testing)
- **HiSilicon NNIE SDK** (for Hi3516CV610 deployment)

## Building

### 1. x86 Build with ONNX Runtime

```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export ONNXRUNTIME_DIR=$PWD/onnxruntime-linux-x64-1.16.0

# Build SDK
cd cpp
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ONNXRUNTIME=ON \
    -DUSE_NNIE=OFF
make -j4
```

### 2. ARM Cross-compilation with NNIE

```bash
# Set up cross-compilation toolchain
export TOOLCHAIN_PATH=/path/to/arm-gcc-toolchain
export NNIE_SDK_PATH=/path/to/hi3516cv610-sdk

# Build SDK
cd cpp
mkdir build-arm && cd build-arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ONNXRUNTIME=OFF \
    -DUSE_NNIE=ON
make -j4
```

## Model Preparation

The SDK requires two ONNX models:

1. **minifasnet_int8.onnx** - RGB-IR liveness detection
2. **arcface_r34_int8.onnx** - Face feature extraction

### Generate from Python

```bash
cd ../python/tools

# Export PyTorch models to ONNX
python model_export.py \
    --model-type arcface \
    --output ../../models/arcface_r34.onnx

python model_export.py \
    --model-type minifasnet \
    --output ../../models/minifasnet.onnx

# Quantize to INT8
python quantization.py \
    --input ../../models/arcface_r34.onnx \
    --output ../../models/arcface_r34_int8.onnx \
    --calibration-data /path/to/calibration/images

python quantization.py \
    --input ../../models/minifasnet.onnx \
    --output ../../models/minifasnet_int8.onnx \
    --calibration-data /path/to/calibration/images \
    --dual-stream
```

### Convert to NNIE .wk format

```bash
# Use NNIE mapper tool (Hi3516CV610 SDK)
$NNIE_SDK_PATH/bin/nnie_mapper \
    --type=onnx \
    --model=../../models/arcface_r34_int8.onnx \
    --output=../../models/arcface_r34_int8.wk \
    --target=hi3516cv610
```

## Usage Example

```cpp
#include "ocfa_face_sdk.h"
#include <cstdio>

int main() {
    // Initialize SDK
    ocfa_config_t config;
    config.model_dir = "../../models";
    config.liveness_threshold = 0.90f;
    config.quality_threshold = 0.50f;
    config.num_threads = 2;

    if (ocfa_init(&config) != OCFA_SUCCESS) {
        printf("Failed to initialize SDK\n");
        return 1;
    }

    // Capture RGB and IR images (1280x720)
    uint8_t* rgb_image = capture_rgb_image();
    uint8_t* ir_image = capture_ir_image();

    // Perform recognition
    ocfa_recognition_result_t result;
    int ret = ocfa_recognize(rgb_image, ir_image, 1280, 720, &result);

    if (ret == OCFA_SUCCESS && result.feature_extracted) {
        printf("Liveness: %.3f\n", result.liveness_score);
        printf("Quality: %.3f\n", result.quality_score);

        // Add to database
        uint8_t user_id[16] = {...};
        ocfa_add_user(user_id, result.feature);

        // Search database
        uint8_t matched_id[16];
        float similarity;
        ocfa_search_user(result.feature, matched_id, &similarity);
        printf("Match: %02x... (%.3f)\n", matched_id[0], similarity);
    }

    // Cleanup
    ocfa_release();
    return 0;
}
```

## API Reference

### Initialization

```cpp
int ocfa_init(const ocfa_config_t* config);
int ocfa_release(void);
const char* ocfa_get_version(void);
```

### Recognition Pipeline

```cpp
// Full pipeline
int ocfa_recognize(const uint8_t* rgb_image, const uint8_t* ir_image,
                   int width, int height, ocfa_recognition_result_t* result);

// Individual steps
int ocfa_detect_liveness(const uint8_t* rgb, const uint8_t* ir,
                         int w, int h, float* score);
int ocfa_assess_quality(const uint8_t* rgb, int w, int h, float* score);
int ocfa_extract_feature(const uint8_t* rgb, const uint8_t* ir,
                         int w, int h, float* feature);
```

### Feature Management

```cpp
float ocfa_compare_feature(const float* feat1, const float* feat2);

int ocfa_add_user(const uint8_t user_id[16], const float* feature);
int ocfa_update_user(const uint8_t user_id[16], const float* feature);
int ocfa_remove_user(const uint8_t user_id[16]);

// 1:1 search
int ocfa_search_user(const float* query, uint8_t user_id[16], float* sim);

// 1:N search
int ocfa_search_users(const float* query, float threshold,
                      ocfa_search_result_t* results, int max_results);
```

## Performance Benchmarks

### Hi3516CV610 (NNIE INT8)

| Operation | Time | Notes |
|-----------|------|-------|
| Liveness detection | ~20ms | MiniFASNet RGB-IR fusion |
| Quality assessment | ~5ms | Laplacian variance |
| Feature extraction | ~30ms | ArcFace-R34 |
| Feature comparison | <1ms | Cosine similarity |
| **Total pipeline** | **~55ms** | End-to-end |

### x86 (ONNX Runtime)

| Operation | Time | Notes |
|-----------|------|-------|
| Liveness detection | ~15ms | CPU inference |
| Feature extraction | ~25ms | CPU inference |
| **Total pipeline** | **~45ms** | Intel i7 |

## Memory Usage

- SDK initialization: ~50MB (models loaded)
- Per-user database entry: 2KB (512-dim float32)
- Recognition pipeline: ~5MB (temporary buffers)

## Troubleshooting

### Build Errors

**ONNX Runtime not found:**
```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime
cmake .. -DONNXRUNTIME_INCLUDE_DIR=$ONNXRUNTIME_DIR/include \
         -DONNXRUNTIME_LIB=$ONNXRUNTIME_DIR/lib/libonnxruntime.so
```

**NNIE SDK not found:**
```bash
export NNIE_SDK_PATH=/path/to/hi3516cv610-sdk
```

### Runtime Errors

**Model loading failed:**
- Verify model files exist in `model_dir`
- Check file permissions
- Ensure ONNX Runtime version compatibility (1.14+)

**Inference failed:**
- Check input image dimensions (must match model requirements)
- Verify image format (RGB: BGR uint8, IR: grayscale uint8)
- Ensure sufficient memory available

## License

Copyright (c) 2025 OCTA Team

## Contact

For issues and questions, please contact the OCTA team.
