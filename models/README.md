# 模型文件目录

本目录存放人脸识别 SDK 所需的模型文件。

## 模型列表

### 1. 活体检测模型 - MiniFASNet

**功能**: RGB-IR 双模态活体检测，防御照片/视频/屏幕攻击

**模型文件**:
- `minifasnet_dual_fp32.pth` - PyTorch 原始模型 (FP32)
- `minifasnet_dual_fp32.onnx` - ONNX 模型 (FP32)
- `minifasnet_dual_int8.onnx` - ONNX INT8 量化模型
- `minifasnet_dual_int8.wk` - Hi3516CV610 NNIE 模型

**输入**:
- RGB 图像: 1x3x112x112 (CHW, BGR, [0, 255])
- IR 图像: 1x1x112x112 (CHW, Gray, [0, 255])

**输出**:
- 活体置信度: 1x2 (real/fake)

**来源**: [InsightFace MiniFASNet](https://github.com/deepinsight/insightface/tree/master/detection/face_anti_spoofing)

### 2. 人脸特征提取模型 - ArcFace-R34

**功能**: 从人脸图像提取 512 维特征向量

**模型文件**:
- `arcface_r34_fp32.pth` - PyTorch 原始模型 (FP32)
- `arcface_r34_rgb_fp32.onnx` - ONNX RGB 模型 (FP32)
- `arcface_r34_ir_fp32.onnx` - ONNX IR 模型 (FP32)
- `arcface_r34_rgb_int8.onnx` - ONNX RGB INT8 量化模型
- `arcface_r34_ir_int8.onnx` - ONNX IR INT8 量化模型
- `arcface_r34_rgb_int8.wk` - Hi3516CV610 NNIE RGB 模型
- `arcface_r34_ir_int8.wk` - Hi3516CV610 NNIE IR 模型

**输入**:
- RGB 图像: 1x3x112x112 (CHW, BGR, [0, 255])
- IR 图像: 1x1x112x112 (CHW, Gray, [0, 255])

**输出**:
- 特征向量: 1x512 (L2 归一化)

**来源**: [InsightFace ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

## 模型下载

### 方式 1: 使用下载脚本 (推荐)

```bash
cd models
python download_models.py
```

### 方式 2: 手动下载

**ArcFace-R34**:
```bash
# 下载 MS1MV2 预训练权重
wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r34_ms1mv2.pth
```

**MiniFASNet**:
```bash
# 从 InsightFace 仓库下载
git clone https://github.com/deepinsight/insightface.git
cp insightface/detection/face_anti_spoofing/models/minifasnet.pth ./
```

## 模型转换流程

### 1. PyTorch -> ONNX

```bash
cd ../python/tools

# 导出 ArcFace-R34
python model_export.py \
    --model arcface_r34 \
    --weights ../../models/arcface_r34_ms1mv2.pth \
    --output ../../models/arcface_r34_rgb_fp32.onnx

# 导出 MiniFASNet
python model_export.py \
    --model minifasnet \
    --weights ../../models/minifasnet.pth \
    --output ../../models/minifasnet_dual_fp32.onnx
```

### 2. ONNX FP32 -> ONNX INT8 (量化)

```bash
# 量化 ArcFace-R34
python quantization.py \
    --model ../../models/arcface_r34_rgb_fp32.onnx \
    --calib-data ../../data/calibration \
    --output ../../models/arcface_r34_rgb_int8.onnx

# 量化 MiniFASNet
python quantization.py \
    --model ../../models/minifasnet_dual_fp32.onnx \
    --calib-data ../../data/calibration \
    --output ../../models/minifasnet_dual_int8.onnx
```

### 3. ONNX INT8 -> NNIE .wk (Hi3516CV610)

需要使用海思 NNIE Mapper 工具:

```bash
# 1. ONNX -> Caffe
onnx2caffe ../../models/arcface_r34_rgb_int8.onnx \
    --output ../../models/arcface_r34_rgb_int8.prototxt \
    --output-weights ../../models/arcface_r34_rgb_int8.caffemodel

# 2. Caffe -> NNIE
nnie_mapper \
    --prototxt=../../models/arcface_r34_rgb_int8.prototxt \
    --caffemodel=../../models/arcface_r34_rgb_int8.caffemodel \
    --wk_file=../../models/arcface_r34_rgb_int8.wk \
    --quantize_type=int8
```

详细转换流程请参考 [../docs/model_conversion.md](../docs/model_conversion.md)

## 模型尺寸

| 模型 | FP32 | INT8 | 压缩比 |
|-----|------|------|-------|
| MiniFASNet 双流 | ~12MB | ~8MB | 1.5x |
| ArcFace-R34 RGB | ~88MB | ~22MB | 4x |
| ArcFace-R34 IR | ~88MB | ~22MB | 4x |
| **总计** | ~188MB | ~52MB | 3.6x |

## 模型性能 (Hi3516CV610 NNIE INT8)

| 模型 | 输入尺寸 | 推理延迟 | 备注 |
|-----|---------|---------|------|
| MiniFASNet | 2x112x112 | ~15ms | 双流融合 |
| ArcFace-R34 RGB | 3x112x112 | ~35ms | 单流 |
| ArcFace-R34 IR | 1x112x112 | ~35ms | 单流 |

并行推理 (RGB + IR): ~35ms

## 模型精度

### ArcFace-R34

| 测试集 | FP32 准确率 | INT8 准确率 | 精度损失 |
|-------|-----------|-----------|---------|
| LFW | 99.50% | 99.42% | -0.08% |
| CPLFW | 92.08% | 91.95% | -0.13% |
| AgeDB-30 | 97.77% | 97.68% | -0.09% |

### MiniFASNet

| 指标 | FP32 | INT8 | 精度损失 |
|-----|------|------|---------|
| APCER | 0.5% | 0.6% | +0.1% |
| BPCER | 1.2% | 1.3% | +0.1% |
| ACER | 0.85% | 0.95% | +0.1% |

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
