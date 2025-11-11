# 模型文件目录

本目录存放人脸识别 SDK 所需的模型文件。

---

## 当前已下载模型 (v1.0.0)

### ✅ 1. 人脸特征提取 - ArcFace R50 (InsightFace buffalo_l)

**功能**: 从人脸图像提取 512 维特征向量

**已有文件**:
- `buffalo_l/w600k_r50.onnx` - FP32 原始模型 (166 MB) ✅
- `w600k_r50_int8_dynamic.onnx` - INT8 动态量化 (42 MB) ✅
- `w600k_r50_int8_static.onnx` - INT8 静态量化 (42 MB) ✅

**输入**: 1x3x112x112 (CHW, RGB, [-1, 1] 归一化)
**输出**: 1x512 (L2 归一化特征)
**来源**: [InsightFace buffalo_l](https://github.com/deepinsight/insightface)

**说明**:
- 这是 R50 模型（ResNet-50），比 R34 更准确但稍慢
- 当前 SDK 配置使用此模型
- 支持 RGB 和 IR 图像（使用同一模型）

### ✅ 2. 活体检测 - MiniFASNet V2 (单流)

**功能**: RGB 单模态活体检测

**已有文件**:
- `minifasnet_v2.pth` - PyTorch 原始模型 (1.8 MB) ✅
- `minifasnet_v2.onnx` - ONNX 模型 (143 KB) ✅

**输入**: 1x3x80x80 (CHW, RGB, [0, 1] 归一化)
**输出**: 1x3 (real/fake/mask 概率)
**来源**: [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

**说明**:
- 这是单流 RGB 模型，仅使用 RGB 图像
- 不是双流 RGB+IR 模型
- 当前 SDK 配置使用此模型（livecheck=1 时）

---

## 缺失的理想模型 (待实现)

### ❌ 1. 双流活体检测模型 (RGB+IR MiniFASNet)

**预期功能**: RGB-IR 双模态活体检测

**需要文件**:
- `minifasnet_dual_fp32.onnx` - RGB+IR 双流模型
- `minifasnet_dual_int8.onnx` - INT8 量化版本

**输入**:
- RGB 通道: 1x3x112x112
- IR 通道: 1x1x112x112
- 合并输入: 1x4x112x112 (RGBI)

**状态**: ⚠️ 暂未找到预训练的双流模型

**替代方案**:
1. 使用当前的单流 MiniFASNet (仅 RGB)
2. 训练自己的双流模型
3. 使用 `livecheck=0` 旁路活体检测

### ❌ 2. 分离的 RGB/IR 特征提取模型 (ArcFace R34)

**预期功能**: 分别优化的 RGB 和 IR 特征提取

**需要文件**:
- `arcface_r34_rgb_fp32.onnx` - RGB 专用模型
- `arcface_r34_ir_fp32.onnx` - IR 专用模型
- INT8 量化版本

**状态**: ⚠️ 当前使用同一个 R50 模型处理 RGB 和 IR

**说明**: 实际部署中，使用同一模型处理 RGB/IR 是常见做法

---

## 推荐的模型获取方式

### 方式 1: 使用现有模型（推荐，已完成）

当前已下载的模型足以运行完整流程：

```bash
# 已有模型
models/
├── buffalo_l/w600k_r50.onnx          # 特征提取 (FP32)
├── w600k_r50_int8_static.onnx        # 特征提取 (INT8)
└── minifasnet_v2.onnx                # 活体检测 (仅RGB)
```

**使用方式**:
- 特征提取：使用 `w600k_r50.onnx` 或量化版本
- 活体检测：使用 `minifasnet_v2.onnx` (livecheck=1)
- 或旁路：使用 `livecheck=0` 跳过活体检测

### 方式 2: 下载 ArcFace R34（可选）

如果需要更快的模型（R34 比 R50 快约 30%）：

```bash
cd models
python download_models.py --model arcface_r34
```

**InsightFace 预训练模型**:
- MS1MV2-ArcFace-R34: https://github.com/deepinsight/insightface
- Glint360K-R34: 更高精度但更慢

### 方式 3: 训练双流活体检测模型（高级）

如果需要真正的 RGB+IR 双流活体检测：

1. 收集 RGB+IR 配对数据集
2. 基于 MiniFASNet 架构训练双流模型
3. 导出为 ONNX 格式
4. 集成到 SDK

**参考**:
- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- [InsightFace Anti-Spoofing](https://github.com/deepinsight/insightface/tree/master/detection/face_anti_spoofing)

---

## 已完成的模型工作

### ✅ 下载

使用 `python/tools/download_models.py`:
```bash
cd python/tools
python download_models.py
```

下载了：
- InsightFace buffalo_l 包（包含 w600k_r50.onnx）
- MiniFASNet V2 模型（从 GitHub）

### ✅ 量化

使用 `python/tools/quantization.py`:
```bash
cd python/tools

# 量化 ArcFace R50
python quantization.py \
    --model ../../models/buffalo_l/w600k_r50.onnx \
    --output ../../models/w600k_r50_int8_static.onnx \
    --quantize-type static \
    --calib-data ../../data/calibration

# 结果: 166MB -> 42MB (4x 压缩)
```

### ✅ 转换

使用 `python/tools/convert_minifasnet.py`:
```bash
cd python/tools

# PyTorch -> ONNX
python convert_minifasnet.py \
    --input ../../models/minifasnet_v2.pth \
    --output ../../models/minifasnet_v2.onnx \
    --model-type v2

# 结果: 1.8MB -> 143KB
```

---

## 模型使用示例

### Python SDK

```python
from ocfa import OCFAFaceSDK

# 配置使用已下载的模型
sdk = OCFAFaceSDK(config_path='configs/default_config.json')

# 识别（使用 livecheck=0 旁路活体检测）
result = sdk.recognize(rgb_image, ir_image, livecheck=0)
```

**配置文件示例** (`configs/default_config.json`):
```json
{
  "models": {
    "liveness_model": "models/minifasnet_v2.onnx",
    "feature_model_rgb": "models/buffalo_l/w600k_r50.onnx",
    "feature_model_ir": "models/buffalo_l/w600k_r50.onnx"
  }
}
```

### C++ SDK

```cpp
ocfa_config_t config;
config.model_dir = "models";
// SDK 会自动加载 minifasnet_v2.onnx 和 w600k_r50.onnx

ocfa_init(&config);

// 旁路活体检测
int livecheck = 0;
ocfa_recognize(rgb, ir, width, height, livecheck, &result);
```

---

## 实际模型尺寸 (当前)

| 模型 | FP32 | INT8 | 压缩比 | 状态 |
|-----|------|------|-------|------|
| **MiniFASNet V2** (单流) | 1.8 MB (pth) | - | - | ✅ 已有 |
| | 143 KB (onnx) | 未量化 | - | ✅ 已有 |
| **ArcFace R50** | 166 MB | 42 MB | 3.95x | ✅ 已有 |
| **总计** | ~168 MB | ~42 MB | 4x | ✅ 完成 |

**说明**:
- MiniFASNet 非常小（143KB），量化收益不大
- ArcFace R50 量化效果显著：166MB → 42MB

---

## 预期性能 (Hi3516CV610 NNIE INT8)

**注:** 以下为理论估算，需实际硬件测试验证

| 模块 | 输入尺寸 | 推理延迟 (估算) | 备注 |
|-----|---------|----------------|------|
| **预处理** | 1280x720 → 112x112 | ~5ms | Resize, 归一化 |
| **MiniFASNet** (livecheck=1) | 3x80x80 | ~10ms | RGB 单流 |
| **ArcFace R50** RGB | 3x112x112 | ~30ms | 特征提取 |
| **ArcFace R50** IR | 3x112x112 | ~30ms | 可并行 |
| **特征融合** | 512+512 | ~1ms | 简单运算 |
| **质量评估** | 112x112 | ~3ms | Laplacian |
| | | | |
| **总计 (livecheck=1)** | | ~**55ms** | 含活体检测 |
| **总计 (livecheck=0)** | | ~**35ms** | 旁路活体检测 |

**性能优化**:
- RGB 和 IR 特征提取可以并行（~30ms 而非 60ms）
- 使用 `livecheck=0` 可节省 ~20ms
- ARM NEON 优化可提升特征比对速度 3-4x

---

## 模型精度 (预期)

### ArcFace R50 (InsightFace)

| 测试集 | FP32 准确率 | INT8 准确率 (估) | 精度损失 |
|-------|-----------|-----------------|---------|
| LFW | 99.83% | ~99.75% | <0.1% |
| CFP-FP | 98.27% | ~98.15% | <0.15% |
| AgeDB-30 | 98.28% | ~98.15% | <0.15% |

**来源**: InsightFace 官方基准测试

### MiniFASNet V2 (Silent-Face-Anti-Spoofing)

| 指标 | 单流 RGB | 备注 |
|-----|---------|------|
| **APCER** | <1% | 攻击样本错误接受率 |
| **BPCER** | <2% | 真实样本错误拒绝率 |
| **ACER** | <1.5% | 平均分类错误率 |

**说明**:
- 仅 RGB 单流，防护能力低于双流
- 适合受控环境，建议配合 `livecheck=0` 使用

## 许可证

本目录中的模型来源于 InsightFace 开源项目，遵循 Apache 2.0 License。

使用这些模型请遵守以下条款：
1. 保留原始许可证声明
2. 注明模型来源
3. 商业使用需遵守 Apache 2.0 License

## 参考资料

- [InsightFace 官方仓库](https://github.com/deepinsight/insightface)
- [ArcFace 论文](https://arxiv.org/abs/1801.07698)
- [MiniFASNet 论文](https://arxiv.org/abs/2003.04092)
