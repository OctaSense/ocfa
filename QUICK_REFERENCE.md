# OCFA Face SDK - 快速参考卡

**版本**: v1.0.0
**更新**: 2025-01-11

---

## 模型选择

### 🎯 我应该选择哪个模型？

| 需求 | 推荐 | 配置 |
|------|------|------|
| **最佳精度** | R50 | `default_config.json` |
| **最快速度** | R34 | `r34_config.json` |
| **不确定** | R50 | `default_config.json` |
| **资源受限** | R50 INT8 | `default_config.json` |
| **安全第一** | R50 + livecheck=1 | `default_config.json` |

---

## 使用代码示例

### Python - 默认配置 (R50)

```python
from ocfa import OCFAFaceSDK
import cv2

# 加载图像
rgb = cv2.imread('face_rgb.jpg')
ir = cv2.imread('face_ir.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化 SDK
sdk = OCFAFaceSDK()

# 识别
result = sdk.recognize(rgb, ir)

if result.success:
    print(f"✓ 特征提取成功")
    print(f"  - 活体: {result.liveness_passed}")
    print(f"  - 质量: {result.quality_passed}")
    print(f"  - 时间: {result.total_time_ms:.1f} ms")
```

### Python - R34 配置 (高性能)

```python
sdk = OCFAFaceSDK(config_path='configs/r34_config.json')
result = sdk.recognize(rgb, ir)
```

### Python - 旁路活体检测 (快速模式)

```python
# 跳过活体检测，仅特征提取
result = sdk.recognize(rgb, ir, livecheck=0)  # 快 ~20ms
```

### C++ - 基础用法

```cpp
#include "ocfa_face_sdk.h"

// 初始化
ocfa_init("configs/default_config.json");

// 识别
ocfa_recognition_result_t result;
ocfa_recognize(rgb_data, ir_data, width, height, 1, &result);

// 搜索
uint8_t matched_id[16];
float similarity;
ocfa_search_user(result.feature, matched_id, &similarity);

// 清理
ocfa_release();
```

---

## 性能数据

### 模型对比

```
        大小      推理时间    精度     推荐
R50     166 MB    31.7 ms     99.83%   ✅ 默认
R34     130 MB    ~20ms*      99.83%   🚀 ARM 设备
R50 INT8 42 MB    15-25ms*    99.75%   💾 受限
```

*在嵌入式设备 (Hi3516CV610) 上的预期值

### 完整流程延迟 (livecheck=0)

```
预处理          ~5 ms
质量评估        ~3 ms
特征提取 (RGB)  ~25 ms
特征提取 (IR)   ~25 ms  (可并行)
特征融合        ~1 ms
─────────────────────
总计 (~35 ms)   ~18 fps
```

---

## 配置文件

### 可用配置

| 文件 | 用途 | 模型 |
|------|------|------|
| `default_config.json` | 开发/生产 | R50 FP32 |
| `r34_config.json` | 性能优化 | R34 FP32 |
| `x86_config.json` | x86 开发 | R50 FP32 |
| `hi3516cv610_config.json` | 硬件专用 | R50 INT8 |

### 快速切换配置

```python
# 方式 1: 使用配置文件
sdk = OCFAFaceSDK(config_path='configs/r34_config.json')

# 方式 2: 动态修改
from ocfa import load_config
config = load_config('configs/default_config.json')
config['models']['feature_model_rgb'] = 'models/arcface_r34_ms1mv3.onnx'
sdk = OCFAFaceSDK(config=config)
```

---

## 活体检测 (Livecheck) 参数

### livecheck 值说明

| 值 | 含义 | 性能 | 安全性 | 用途 |
|----|------|------|--------|------|
| 0 | 旁路 | ⚡ 快 | ⚠️ 低 | 受控环境 |
| 1 | RGB+IR | 正常 | ✅ 高 | 一般场景 |

### 使用示例

```python
# 快速识别 (受控环境)
result = sdk.recognize(rgb, ir, livecheck=0)

# 安全识别 (门禁/支付)
result = sdk.recognize(rgb, ir, livecheck=1)

# 混合模式
if operation_type == "login":
    result = sdk.recognize(rgb, ir, livecheck=0)
elif operation_type == "payment":
    result = sdk.recognize(rgb, ir, livecheck=1)
```

---

## 常见任务

### 1:1 验证 (人脸比对)

```python
result1 = sdk.recognize(rgb_image1, ir_image1, livecheck=0)
result2 = sdk.recognize(rgb_image2, ir_image2, livecheck=0)

if result1.success and result2.success:
    similarity = sdk.compare_features(result1.feature, result2.feature)
    if similarity > 0.60:
        print("✓ 相同人")
    else:
        print("✗ 不同人")
```

### 1:N 搜索 (人脸识别)

```python
# 添加用户
user_id = bytes.fromhex('0123456789abcdef')
sdk.add_user(user_id, feature)

# 搜索用户
result = sdk.recognize(rgb_image, ir_image, livecheck=0)
matches = sdk.search_users(result.feature, threshold=0.60)

for user_id, similarity in matches:
    print(f"用户: {user_id.hex()}, 相似度: {similarity:.3f}")
```

### 性能测试

```bash
# 运行性能对比
python test_r34_vs_r50.py

# 运行基础测试
python test_basic_pipeline.py

# 运行活体检测演示
python test_livecheck.py
```

---

## 故障排除

### "找不到模型文件"

```bash
# 检查模型是否存在
ls -lh models/*.onnx

# 预期输出
arcface_r34_ms1mv3.onnx         (130 MB)
buffalo_l/w600k_r50.onnx        (166 MB)
w600k_r50_int8_dynamic.onnx     (42 MB)
minifasnet_v2.onnx              (143 KB)
```

### "R34 和 R50 结果不同"

这是正常的! 两个模型架构不同，特征分布也不同。
- 不要混合使用两个模型的特征
- 始终使用同一模型进行比较

### "R34 推理给出错误"

确保:
1. 模型文件存在: `models/arcface_r34_ms1mv3.onnx`
2. 配置正确指向该模型
3. 使用 Python 3.7+ 和最新的 ONNX Runtime

### "INT8 量化出错"

这在 macOS 上是已知问题。解决方案:
- 使用 FP32 版本 (开发环境)
- 在部署设备上使用设备特定的量化工具
- 不影响功能，仅性能有差异

---

## 文件导航

| 文件 | 内容 | 何时阅读 |
|------|------|---------|
| **README.md** | 项目概览 | 开始时 |
| **ARCFACE_MODELS.md** | 模型详细对比 | 选择模型时 |
| **QUICK_REFERENCE.md** | 本文档 | 快速查看 |
| **models/README.md** | 模型库详情 | 管理模型时 |
| **LIVECHECK_REFACTORING.md** | 活体检测重构 | 深入了解时 |
| **test_r34_vs_r50.py** | 性能脚本 | 基准测试时 |

---

## 一句话建议

🎯 **不知道选什么？用 R50**
⚡ **需要快速？用 R34**
📱 **ARM 设备？都支持**
🔐 **需要安全？用 livecheck=1**

---

## 版本说明

- **v1.0.0** - 初始版本，支持 R50 和 R34
- **v1.1** (计划) - 真实数据验证
- **v1.2** (计划) - 双流活体检测
- **v2.0** (计划) - 自适应模型选择

---

**最后更新**: 2025-01-11
**维护者**: OCFA 团队
