# OCFA Face SDK - Quick Start Guide

快速开始使用 OCFA 人脸识别 SDK。

## 环境要求

### Python 开发环境
- Python 3.7+
- PyTorch 1.12+
- ONNX Runtime 1.14+

### C++ 部署环境
- C++17 编译器 (GCC 7+ 或 Clang 5+)
- CMake 3.16+
- ONNX Runtime (x86 测试) 或 NNIE SDK (Hi3516CV610 部署)

## 5 分钟快速测试

### 1. 克隆仓库

```bash
cd /path/to/workspace
# 假设您已经在 octas/face 目录
```

### 2. Python 环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r python/requirements.txt
```

### 3. 生成测试模型

由于完整的 InsightFace 模型较大，我们先创建虚拟模型进行功能测试：

```bash
cd python/tools
python create_dummy_models.py
```

这会在 `models/` 目录生成：
- `arcface_r34_int8.onnx` - 人脸特征提取模型（虚拟）
- `minifasnet_int8.onnx` - 活体检测模型（虚拟）

⚠️ **注意**: 虚拟模型仅用于功能测试，不会产生有意义的识别结果。

### 4. Python 测试

```bash
# 回到项目根目录
cd ../..

# 运行 Python 示例
python python/examples/demo_basic.py
```

预期输出：
```
OCFA Face SDK - Basic Example
Version: 1.0.0

Initializing SDK...
SDK initialized successfully

Testing recognition pipeline...
[功能演示输出]
```

### 5. C++ 编译和测试

```bash
# 创建构建目录
cd cpp
mkdir build && cd build

# 配置 CMake (使用 ONNX Runtime)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ONNXRUNTIME=ON \
    -DUSE_NNIE=OFF \
    -DBUILD_EXAMPLES=ON

# 编译
make -j4

# 运行示例
./examples/demo_basic
./examples/demo_recognition
./examples/benchmark_neon
```

预期输出（demo_recognition）：
```
OCFA Face SDK - Recognition Example
Version: 1.0.0

Initializing SDK...
SDK initialized successfully

Creating test images (1280x720)...
Test images created

Performing face recognition...
Recognition successful!
  Liveness: 0.XXX (threshold: 0.90) - [PASSED/FAILED]
  Quality:  0.XXX (threshold: 0.50) - [PASSED/FAILED]
  Feature:  EXTRACTED
  Total time: XX ms
```

## 使用真实模型

### 方式 1: 下载预训练 ONNX 模型

```bash
cd models
python download_models.py
```

这会下载：
- `arcface_r100.onnx` (249 MB) - 高精度模型
- `arcface_mobilefacenet.onnx` (3.8 MB) - 轻量级模型

### 方式 2: 从 InsightFace 转换

```bash
# 1. 下载 PyTorch 权重
cd models
# 下载 arcface_r34_ms1mv2.pth 到此目录

# 2. 导出为 ONNX
cd ../python/tools
python model_export.py \
    --model-type arcface \
    --model-path ../../models/arcface_r34_ms1mv2.pth \
    --output ../../models/arcface_r34.onnx

# 3. INT8 量化
python quantization.py \
    --input ../../models/arcface_r34.onnx \
    --output ../../models/arcface_r34_int8.onnx \
    --calibration-data /path/to/calibration/images \
    --calibration-count 100
```

详细模型获取方法见 [models/README.md](models/README.md)。

## Python SDK 使用示例

```python
from ocfa import OCFAFaceSDK
import numpy as np

# 初始化 SDK
sdk = OCFAFaceSDK(config_path="configs/default_config.json")

# 准备图像 (1280x720, BGR format)
rgb_image = np.zeros((720, 1280, 3), dtype=np.uint8)  # 从相机读取
ir_image = np.zeros((720, 1280), dtype=np.uint8)      # 从红外相机读取

# 人脸识别
result = sdk.recognize(rgb_image, ir_image)

if result.success:
    print(f"活体检测: {result.liveness_score:.3f}")
    print(f"质量评估: {result.quality_score:.3f}")

    if result.liveness_passed and result.quality_passed:
        # 添加到数据库
        user_id = b'user_12345678901'  # 16 字节
        sdk.add_user(user_id, result.feature)

        # 1:N 搜索
        matches = sdk.search_users(result.feature, threshold=0.70, max_results=5)
        for user_id, similarity in matches:
            print(f"匹配用户: {user_id.hex()}, 相似度: {similarity:.3f}")
```

## C++ SDK 使用示例

```cpp
#include "ocfa_face_sdk.h"

int main() {
    // 初始化配置
    ocfa_config_t config;
    config.model_dir = "../../models";
    config.liveness_threshold = 0.90f;
    config.quality_threshold = 0.50f;
    config.num_threads = 2;

    // 初始化 SDK
    if (ocfa_init(&config) != OCFA_SUCCESS) {
        printf("SDK 初始化失败\n");
        return 1;
    }

    // 准备图像数据 (从相机读取)
    uint8_t* rgb_image = ...; // 1280x720x3, BGR
    uint8_t* ir_image = ...;  // 1280x720x1, Grayscale

    // 执行识别
    ocfa_recognition_result_t result;
    int ret = ocfa_recognize(rgb_image, ir_image, 1280, 720, &result);

    if (ret == OCFA_SUCCESS && result.liveness_passed && result.quality_passed) {
        // 添加用户
        uint8_t user_id[16] = {...};
        ocfa_add_user(user_id, result.feature);

        // 搜索用户
        uint8_t matched_id[16];
        float similarity;
        ocfa_search_user(result.feature, matched_id, &similarity);

        printf("匹配用户，相似度: %.3f\n", similarity);
    }

    // 释放资源
    ocfa_release();
    return 0;
}
```

## 性能测试

### Python 性能测试

```bash
cd python/tests
python test_performance.py
```

### C++ NEON 优化测试

```bash
cd cpp/build
./examples/benchmark_neon
```

在 ARM 平台上会显示 NEON 优化的加速效果：

```
=== Cosine Similarity (512-dim) ===
Standard CosineSimilarity  : 2.150 µs
NEON CosineSimilarity      : 0.680 µs
Speedup: 3.16x
```

## 常见问题

### Q1: 模型加载失败

**错误**: `Failed to initialize SDK: Model loading failed`

**解决**:
1. 检查模型文件是否存在于 `models/` 目录
2. 验证模型文件名: `arcface_r34_int8.onnx`, `minifasnet_int8.onnx`
3. 检查文件权限
4. 使用 `create_dummy_models.py` 生成测试模型

### Q2: ONNX Runtime 找不到

**错误**: `ONNX Runtime not found`

**解决**:
```bash
# 下载 ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz

# 设置环境变量
export ONNXRUNTIME_DIR=$PWD/onnxruntime-linux-x64-1.16.0

# 重新编译
cd cpp/build
cmake .. -DONNXRUNTIME_INCLUDE_DIR=$ONNXRUNTIME_DIR/include \
         -DONNXRUNTIME_LIB=$ONNXRUNTIME_DIR/lib/libonnxruntime.so
make
```

### Q3: 识别结果不准确

**原因**: 使用了虚拟模型 (`create_dummy_models.py` 生成)

**解决**:
- 下载真实的 InsightFace 预训练模型
- 参考 [models/README.md](models/README.md) 获取模型

### Q4: Hi3516CV610 部署

参考 [cpp/README.md](cpp/README.md) 的 "ARM Cross-compilation with NNIE" 章节。

## 下一步

1. **阅读完整文档**
   - [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 系统架构
   - [sdk.md](docs/sdk.md) - SDK 接口规范
   - [flow.md](docs/flow.md) - 识别流程

2. **获取真实模型**
   - 按照 [models/README.md](models/README.md) 下载并转换模型

3. **集成到应用**
   - Python: 参考 `python/examples/`
   - C++: 参考 `cpp/examples/`

4. **性能优化**
   - 启用 NEON 优化 (ARM 平台)
   - 使用 NNIE 加速 (Hi3516CV610)
   - INT8 量化

## 技术支持

如遇问题，请查看：
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 实现总结
- [FAQ.md](docs/FAQ.md) - 常见问题
- GitHub Issues: [项目仓库]

---

**祝您使用愉快！** 🚀
