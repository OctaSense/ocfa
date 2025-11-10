# 人脸识别 SDK 架构说明文档

## 1. 项目概述

### 1.1 项目目标

开发一个适用于嵌入式 Linux 平台的人脸识别算法 SDK，支持 RGB-IR 双模态活体检测、人脸特征提取和人脸比对功能。

**项目定位**：
- 不训练基础模型，直接使用 InsightFace 开源模型
- 仅对开源模型进行 INT8 量化适配
- 针对海思 Hi3516CV610 平台优化

### 1.2 目标运行环境

| 项目 | 规格 |
|------|------|
| **芯片平台** | 海思 Hi3516CV610 |
| **处理器** | ARM A17 双核 800MHz |
| **内存** | 128MB RAM |
| **NPU** | Hi3516CV610 NNIE 3.0，1T ops@INT8 |
| **摄像头** | RGB + IR 双目，720p，FOV H:60° V:45° |
| **操作系统** | 嵌入式 Linux |
| **存储** | 128MB Flash (推荐) |

### 1.3 核心功能

- **RGB-IR 活体检测**: 防止照片/手机/平板攻击
- **人脸特征提取**: 基于 InsightFace ArcFace-R34 模型
- **人脸比对**: 支持 1:1 验证和 1:N 识别模式
- **特征库管理**: 支持用户特征注册、删除、更新

### 1.4 性能指标

- **识别准确率**: >97% (LFW 测试)
- **误识率**: <1%
- **处理延迟**: <100ms
- **帧率**: ≥10fps (推荐 ≥14fps)

## 2. 系统架构

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│        (人脸注册/验证/识别 + 特征库管理接口)             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                    SDK Core Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  图像预处理   │  │  活体检测     │  │  特征提取     │  │
│  │  (同步/增强)  │  │(RGB-IR融合)  │  │ (ArcFace-R34)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  质量评估     │  │  特征融合     │  │  特征比对     │  │
│  │  (模糊/姿态)  │  │  (自适应)    │  │  (1:1/1:N)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                  Inference Engine                       │
│          Hi3516CV610 NNIE + ONNX Runtime                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                   Hardware Layer                        │
│        CPU + NNIE + RGB/IR Camera + Storage             │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
RGB(720p) ──┐                                      ┌──> 1:1 验证
            ├──> 预处理 ──> 活体检测 ──┬──> 特征提取 ──> 特征融合 ──┤
IR(720p) ───┘             (置信值)    │  (ArcFace-R34)  (自适应)    └──> 1:N 识别
                                      │     512维
                                      └──[活体失败]──> 返回拒绝
```

**关键流程说明**:
1. RGB + IR 图像输入，进行同步与预处理
2. 活体检测判断是否为真人（输出置信值）
3. 若活体通过，提取人脸特征（RGB+IR 双流）
4. 特征融合，根据光照条件自适应调整权重
5. 特征比对，1:1 或 1:N 模式，输出结果

## 3. 核心模块设计

### 3.1 图像预处理模块

**功能**: RGB-IR 图像同步、去噪、增强

**输入**: RGB 图像(720p) + IR 图像(720p)

**输出**: 预处理后的 RGB + IR 图像对

#### RGB 预处理流程
```
RGB输入 → 色彩空间转换(BGR2YUV) → 直方图均衡 → 双边滤波 → 白平衡 → 伽马校正
```

#### IR 预处理流程
```
IR输入 → 中值滤波(去噪) → CLAHE(对比度增强) → 温度补偿(可选)
```

**性能**: 8ms (RGB: 5ms, IR: 3ms)

**实现方式**:
- 使用 OpenCV 或自定义 NEON 优化算法
- CPU 实现

### 3.2 活体检测模块

**功能**: RGB-IR 双模态活体检测，防止照片/视频/屏幕攻击

**输入**: 预处理后的 RGB + IR 图像

**输出**: 活体置信值 [0.0, 1.0] + 通过/拒绝标志

#### 技术方案: MiniFASNet 双流融合

**模型架构**: MiniFASNet + IR分支 (双流融合)
- **参数量**: ~2M
- **输入**: RGB + IR 双模态
- **特点**: RGB/IR 独立特征提取后融合
- **性能**: ~15ms@NNIE

**模型来源**:
- InsightFace MiniFASNet: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- 直接使用预训练权重，不需要重新训练
- 基于 RGB 模型扩展 IR 分支

#### 活体检测流程
```
RGB 输入 ──> MiniFASNet-RGB ──> RGB 活体特征
                                       │
                                       ├──> 特征融合 ──> 活体分类 ──> 置信值
                                       │
IR 输入  ──> MiniFASNet-IR  ──> IR 活体特征
```

**检测策略**:
- RGB 分支: 检测纹理、反光等特征
- IR 分支: 检测温度分布、材质特征
- 特征融合: 加权融合 RGB 和 IR 特征
- 最终分类: 通过 softmax 输出置信值

**阈值建议**:
- 严格模式: 0.95 (安全性高，误拒率稍高)
- 平衡模式: 0.90 (推荐)
- 宽松模式: 0.85 (通过率高，安全性稍低)

**性能**: 15ms@NNIE INT8

### 3.3 质量评估模块

**功能**: 评估人脸图像质量，确保特征提取效果

**输入**: 对齐后的人脸图像

**输出**: 质量分数 + 详细评估结果

#### 评估维度

| 维度 | 方法 | 合格标准 | 耗时 |
|------|------|---------|------|
| 清晰度 | 拉普拉斯方差 | >0.5 | 0.5ms |
| 遮挡度 | 关键点可见性 | <0.3 | 0.3ms |
| 姿态角度 | Pitch/Yaw 计算 | ±30° | 0.5ms |
| 光照质量 | 直方图分析 | >0.4 | 0.5ms |
| 人脸大小 | 人脸框面积 | >60px | 0.2ms |

**综合评分**:
```
overall = blur * 0.3 + occlusion * 0.3 + pose * 0.2 + illumination * 0.2
```

**性能**: 2ms

**实现**: CPU 实现

### 3.4 人脸特征提取模块

**功能**: 提取 512 维人脸特征向量

**输入**: 对齐后的 RGB + IR 人脸图像 (112x112)

**输出**: RGB 特征向量 (512维) + IR 特征向量 (512维)

**前置条件**: 活体检测通过 + 质量评估合格

#### 技术方案: ArcFace-R34

**模型架构**: ArcFace-R34 (ResNet-34 骨干网络)
- **参数量**: ~21M
- **FLOPs**: ~3.5G
- **精度**: 99.5% (LFW 测试集)
- **性能**: ~35ms@NNIE INT8

**模型来源**:
- InsightFace 官方预训练模型
- 模型下载: https://github.com/deepinsight/insightface/tree/master/model_zoo
- 使用 MS1MV2 数据集训练的权重
- 直接进行 INT8 量化，不重新训练

#### 双流特征提取
```
RGB (112x112x3) ──> ArcFace-R34-RGB ──> feat_rgb (512)
                                             │
                                             ├──> L2 归一化
                                             │
IR (112x112x1)  ──> ArcFace-R34-IR  ──> feat_ir (512)
```

**权重共享策略**:
- RGB 和 IR 使用相同的骨干网络权重
- 但使用不同的 BatchNorm 层参数
- 减少模型大小，提高效率

**性能**:
- 串行执行: 70ms (RGB: 35ms + IR: 35ms)
- 并行执行: 35ms (RGB 和 IR 同时推理，需要 NNIE 支持)

### 3.5 特征融合模块

**功能**: RGB-IR 特征自适应融合

**输入**: RGB 特征 + IR 特征 + 环境上下文(光照)

**输出**: 融合特征 (512维)

#### 技术方案: 自适应融合

**融合策略**: 基于环境光照的自适应加权融合
- **特点**: 根据光照条件和图像质量动态调整权重
- **性能**: ~1ms (CPU实现)

#### 自适应融合算法
```c
// 自适应融合算法
void ocfa_adaptive_fusion(
    const float* rgb_feat,        // RGB特征 (512)
    const float* ir_feat,         // IR特征 (512)
    float rgb_quality,            // RGB质量 [0,1]
    float ir_quality,             // IR质量 [0,1]
    int illumination_lux,         // 光照强度 (lux)
    float* fused_feat             // 输出融合特征 (512)
) {
    float alpha, beta;

    // 1. 根据光照确定基础权重
    if (illumination_lux > 100) {      // 光照充足
        alpha = 0.8;
        beta = 0.2;
    } else if (illumination_lux > 10) { // 光照一般
        alpha = 0.5;
        beta = 0.5;
    } else {                            // 光照不足
        alpha = 0.2;
        beta = 0.8;
    }

    // 2. 质量加权
    alpha *= rgb_quality;
    beta *= ir_quality;

    // 3. 归一化权重
    float sum = alpha + beta;
    alpha /= sum;
    beta /= sum;

    // 4. 加权融合
    for (int i = 0; i < 512; i++) {
        fused_feat[i] = alpha * rgb_feat[i] + beta * ir_feat[i];
    }

    // 5. L2归一化
    float norm = 0.0f;
    for (int i = 0; i < 512; i++) {
        norm += fused_feat[i] * fused_feat[i];
    }
    norm = sqrtf(norm);

    for (int i = 0; i < 512; i++) {
        fused_feat[i] /= norm;
    }
}
```

**性能**: 1ms

### 3.6 特征比对模块

**功能**: 计算特征向量相似度，支持 1:1 和 1:N 模式

**输入**: 查询特征 + 特征库

**输出**: 相似度（置信值）+ 匹配用户ID

#### 比对算法
```c
// 余弦相似度计算
// 由于特征已L2归一化，简化为点积
float ocfa_cosine_similarity(const float* feat1, const float* feat2) {
    float dot = 0.0f;
    for (int i = 0; i < 512; i++) {
        dot += feat1[i] * feat2[i];
    }
    return dot;
}
```

#### 技术方案: NEON 优化线性搜索

**检索算法**: NEON SIMD 优化的线性搜索
- **特点**: 利用 ARM NEON 指令加速向量点积计算
- **适用规模**: N < 5000 用户
- **性能**: ~2ms (5000人), ~0.5ms (1000人)

**内存占用**:
- 1000 人: 512 × 4B × 1000 = 2MB
- 5000 人: 10MB
- 10000 人: 20MB

**比对阈值建议**:
- **1:1 验证模式**:
  - 严格: 0.70
  - 推荐: 0.65
  - 宽松: 0.60

- **1:N 识别模式**:
  - 严格: 0.75
  - 推荐: 0.70
  - 宽松: 0.65

**性能**: 0.5ms (1000人) ~ 10ms (10000人)

### 3.7 特征库管理模块

**功能**: 用户特征的添加、删除、搜索

**设计原则**:
- SDK 仅提供内存级别的特征库管理（添加、删除、搜索）
- 用户元数据（姓名、权限、注册时间等）由调用者自行管理
- 特征库的持久化（保存/加载）由调用者实现

#### 特征库数据结构（内部实现）
```c
// SDK 内部特征库结构（不对外暴露）
typedef struct {
    uint32_t total_count;          // 总人数
    uint32_t feature_dim;          // 特征维度(512)
    uint32_t max_capacity;         // 最大容量

    // 特征数据
    float* features;               // [N x 512] 矩阵
    uint8_t (*user_ids)[16];      // 用户ID数组 (每个16字节)

    // 索引结构（用于加速检索）
    void* neon_index;              // NEON 优化索引
} ocfa_feature_db_internal_t;
```

**特征库操作**:
- **添加用户**: `ocfa_add_user(user_id, feature)`
- **更新用户**: `ocfa_update_user(user_id, feature)`
- **删除用户**: `ocfa_remove_user(user_id)`
- **查找单个**: `ocfa_search_user(feature, user_id, &similarity)`
- **查找多个**: `ocfa_search_users(feature, threshold, results, max_results)`

**调用者责任**:
- 管理用户元数据（姓名、权限、照片等）
- 实现特征库的持久化存储（文件/数据库）
- 在 SDK 初始化后，通过 `ocfa_add_user` 批量加载用户特征
- 定期备份特征库数据

## 4. 推理引擎方案

### 4.1 Hi3516CV610 NNIE 介绍

**海思 NNIE (Neural Network Inference Engine)**:
- NNIE 3.0 架构
- 1T ops@INT8 算力
- 支持 Caffe 模型转换
- 支持 INT8/INT16 量化
- 专用硬件加速单元

### 4.2 推理引擎配置

#### 推理引擎方案: 纯 NNIE

**推理框架**: 海思 NNIE 3.0
- **特点**: 硬件加速，性能最优
- **支持**: INT8/INT16 量化
- **算力**: 1T ops@INT8

**计算分配**:
- **NNIE**: 活体检测模型、特征提取模型（核心推理）
- **CPU**: 图像预处理、质量评估、特征比对、数据管理

### 4.3 模型转换流程

```
PyTorch (.pth) → ONNX (.onnx) → Caffe (.prototxt + .caffemodel) → NNIE (.wk)
                                                                         ↑
                                                            nnie_mapper (海思工具)
```

**工具链**:
1. **PyTorch → ONNX**: `torch.onnx.export`
2. **ONNX → Caffe**: `onnx2caffe` 或 `MMdeploy`
3. **Caffe → NNIE**: `nnie_mapper` (海思 RuyiStudio)

**详细步骤**:
```bash
# Step 1: 下载 InsightFace 预训练模型
# 从 InsightFace model_zoo 下载 ArcFace-R34

# Step 2: PyTorch → ONNX
python export_onnx.py --model arcface_r34 --input-size 112,112

# Step 3: ONNX → Caffe
python onnx2caffe.py arcface_r34.onnx

# Step 4: 准备量化校准数据集 (1000张人脸图片)
# 用于 INT8 量化统计

# Step 5: Caffe → NNIE (使用 nnie_mapper)
nnie_mapper \
  --prototxt=arcface_r34.prototxt \
  --caffemodel=arcface_r34.caffemodel \
  --calibration_data=calib_data/ \
  --quantize_type=int8 \
  --output=arcface_r34_int8.wk
```

### 4.4 模型量化

#### 量化策略: INT8 PTQ

**量化方案**: INT8 后训练量化 (Post-Training Quantization)
- **精度损失**: 1-2% (可接受范围)
- **实现难度**: 低
- **工具**: NNIE nnie_mapper

**量化工具**:
- NNIE 自带量化工具（nnie_mapper 集成）
- 需要准备校准数据集（~1000张人脸图片）

**量化流程**:
1. 准备校准数据集（从 LFW 或 MS1MV2 中采样 1000 张）
2. 使用 nnie_mapper 进行 INT8 量化
3. 在 NNIE simulator 上验证精度
4. 板端实测精度（与 FP32 对比）

## 5. 数据集准备

### 5.1 数据集用途

**本项目不训练模型，数据集仅用于**:
1. **量化校准**: 用于 INT8 PTQ 量化统计（~1000 张）
2. **精度验证**: 验证量化后模型精度（LFW 标准测试集）
3. **功能测试**: 测试 SDK 各功能模块（~100 张）

### 5.2 校准数据集

**用途**: INT8 量化校准

**规模**: 1000 张人脸图片

**来源**:
- 从 LFW 数据集中随机采样
- 或从 MS1MV2 中随机采样
- 或使用自采集数据

**要求**:
- 多样性: 不同年龄、性别、肤色
- 分辨率: 112x112 (对齐后)
- 格式: JPG/PNG
- 光照: 涵盖不同光照条件

**目录结构**:
```
calibration_data/
├── calib_0001.jpg
├── calib_0002.jpg
├── ...
└── calib_1000.jpg
```

### 5.3 测试数据集

**用途**: 功能测试和精度验证

| 数据集 | 规模 | 用途 | 来源 |
|-------|------|------|------|
| LFW | 13233 张 | 精度验证 | http://vis-www.cs.umass.edu/lfw/ |
| 自采数据 | ~100 张 | 功能测试 | 现场采集 RGB+IR |

## 6. API 接口设计 (Linux 风格)

### 6.1 核心 API

```c
#ifndef OCFA_FACE_SDK_H
#define OCFA_FACE_SDK_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//========== 常量定义 ==========
#define OCFA_FEATURE_DIM 512
#define OCFA_USER_ID_LEN 16
#define OCFA_MAX_NAME_LEN 64
#define OCFA_MAX_ERROR_MSG 128

//========== 错误码 ==========
#define OCFA_SUCCESS                0
#define OCFA_ERROR_INIT_FAILED      -1
#define OCFA_ERROR_INVALID_PARAM    -2
#define OCFA_ERROR_NO_FACE          -3
#define OCFA_ERROR_LIVENESS_FAILED  -4
#define OCFA_ERROR_QUALITY_LOW      -5
#define OCFA_ERROR_FEATURE_EXTRACT  -6
#define OCFA_ERROR_DB_FULL          -7
#define OCFA_ERROR_USER_NOT_FOUND   -8
#define OCFA_ERROR_NO_MATCH         -9
#define OCFA_ERROR_OUT_OF_MEMORY    -10

//========== 数据结构 ==========

// 人脸属性信息（可选，取决于模型是否支持）
typedef struct {
    bool has_age;                   // 是否有年龄信息
    int age;                        // 年龄 (0-100)

    bool has_gender;                // 是否有性别信息
    int gender;                     // 性别: 0-未知, 1-男, 2-女
    float gender_confidence;        // 性别置信度 [0.0, 1.0]

    bool has_mask;                  // 是否有口罩检测
    bool wearing_mask;              // 是否戴口罩
    float mask_confidence;          // 口罩置信度 [0.0, 1.0]

    bool has_expression;            // 是否有表情识别
    int expression;                 // 表情: 0-中性, 1-微笑, 2-惊讶, 3-其他
    float expression_confidence;    // 表情置信度 [0.0, 1.0]
} ocfa_face_attributes_t;

// 识别结果 (仅到特征提取)
typedef struct {
    // 活体检测结果
    bool liveness_passed;           // 活体是否通过
    float liveness_score;           // 活体置信值 [0.0, 1.0]

    // 质量评估结果
    bool quality_passed;            // 质量是否合格
    float quality_score;            // 质量评分 [0.0, 1.0]

    // 特征提取结果
    bool feature_extracted;         // 是否成功提取特征
    float feature[OCFA_FEATURE_DIM]; // 特征向量 (512维)

    // 人脸属性（可选，取决于模型）
    ocfa_face_attributes_t attributes; // 人脸属性信息

    // 处理时间
    uint32_t total_time_ms;         // 总耗时(ms)

    // 错误信息
    int error_code;                 // 错误码
    char error_msg[OCFA_MAX_ERROR_MSG]; // 错误描述
} ocfa_recognition_result_t;

// 搜索结果
typedef struct {
    uint8_t user_id[OCFA_USER_ID_LEN]; // 用户ID (16字节)
    float similarity;                   // 相似度 [0.0, 1.0]
} ocfa_search_result_t;

// 配置参数
typedef struct {
    const char* model_dir;          // 模型目录路径
    const char* config_file;        // 配置文件路径
    float liveness_threshold;       // 活体阈值
    float quality_threshold;        // 质量阈值
    int num_threads;                // 线程数
} ocfa_config_t;

//========== 初始化与释放 ==========

/**
 * @brief 初始化人脸识别 SDK
 * @param config 配置参数
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 */
int ocfa_init(const ocfa_config_t* config);

/**
 * @brief 释放人脸识别 SDK 资源
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 */
int ocfa_release(void);

//========== 人脸识别流程 (到特征提取) ==========

/**
 * @brief 人脸识别流程 (活体检测 -> 质量评估 -> 特征提取)
 * @param rgb_image RGB 图像数据 (720p, BGR格式)
 * @param ir_image IR 图像数据 (720p, 单通道)
 * @param width 图像宽度
 * @param height 图像高度
 * @param result 输出识别结果（包含特征向量）
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 * @note 此接口仅进行到特征提取，不包含特征比对
 */
int ocfa_recognize(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    ocfa_recognition_result_t* result
);

//========== 分步接口 (可选) ==========

/**
 * @brief 活体检测
 * @param rgb_image RGB 图像数据
 * @param ir_image IR 图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param liveness_score 输出活体置信值
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 */
int ocfa_detect_liveness(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    float* liveness_score
);

/**
 * @brief 质量评估
 * @param rgb_image RGB 图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param quality_score 输出质量评分
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 */
int ocfa_assess_quality(
    const uint8_t* rgb_image,
    int width,
    int height,
    float* quality_score
);

/**
 * @brief 提取特征
 * @param rgb_image RGB 图像数据
 * @param ir_image IR 图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param feature 输出特征向量 (512维)
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 */
int ocfa_extract_feature(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    float* feature
);

//========== 特征比对接口 ==========

/**
 * @brief 特征比对 (1:1)
 * @param feature1 特征向量1
 * @param feature2 特征向量2
 * @return 相似度 [0.0, 1.0]
 */
float ocfa_compare_feature(
    const float* feature1,
    const float* feature2
);

//========== 特征库管理接口 ==========

/**
 * @brief 添加用户到特征库
 * @param user_id 用户ID (16字节)
 * @param feature 用户特征向量 (512维)
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 * @note 用户元数据（姓名、权限等）由调用者自行管理
 */
int ocfa_add_user(
    const uint8_t user_id[OCFA_USER_ID_LEN],
    const float* feature
);

/**
 * @brief 更新用户特征
 * @param user_id 用户ID (16字节)
 * @param feature 新的用户特征向量 (512维)
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 * @note 用户必须已存在，否则返回 OCFA_ERROR_USER_NOT_FOUND
 */
int ocfa_update_user(
    const uint8_t user_id[OCFA_USER_ID_LEN],
    const float* feature
);

/**
 * @brief 从特征库删除用户
 * @param user_id 用户ID (16字节)
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 */
int ocfa_remove_user(
    const uint8_t user_id[OCFA_USER_ID_LEN]
);

/**
 * @brief 查找最相似的用户 (1:1 场景)
 * @param query_feature 查询特征向量 (512维)
 * @param user_id 输出最相似用户的ID (16字节)
 * @param similarity 输出相似度 [0.0, 1.0]
 * @return 成功返回 OCFA_SUCCESS，失败返回错误码
 * @note 返回相似度最高的单个用户
 */
int ocfa_search_user(
    const float* query_feature,
    uint8_t user_id[OCFA_USER_ID_LEN],
    float* similarity
);

/**
 * @brief 查找多个相似用户 (1:N 场景)
 * @param query_feature 查询特征向量 (512维)
 * @param threshold 相似度阈值 [0.0, 1.0]，仅返回相似度 >= threshold 的用户
 * @param results 输出结果数组 (需预分配，大小由 max_results 指定)
 * @param max_results 最多返回的结果数量（results 数组大小）
 * @return 实际返回的用户数量（按相似度降序排列），失败返回负数错误码
 * @note 返回的用户按相似度降序排列
 */
int ocfa_search_users(
    const float* query_feature,
    float threshold,
    ocfa_search_result_t* results,
    int max_results
);

//========== 工具函数 ==========

/**
 * @brief 获取 SDK 版本号
 * @return 版本字符串
 */
const char* ocfa_get_version(void);

/**
 * @brief 获取错误描述
 * @param error_code 错误码
 * @return 错误描述字符串
 */
const char* ocfa_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // OCFA_FACE_SDK_H
```

### 6.2 配置文件格式

#### 配置文件示例 (config.json)
```json
{
  "version": "1.0.0",
  "platform": "Hi3516CV610",

  "models": {
    "liveness_model": "models/minifasnet_dual_int8.wk",
    "feature_model_rgb": "models/arcface_r34_rgb_int8.wk",
    "feature_model_ir": "models/arcface_r34_ir_int8.wk"
  },

  "thresholds": {
    "liveness": 0.90,
    "quality": 0.50
  },

  "input": {
    "rgb_width": 1280,
    "rgb_height": 720,
    "ir_width": 1280,
    "ir_height": 720,
    "face_size": 112
  },

  "inference": {
    "device": "nnie",
    "num_threads": 2,
    "batch_size": 1
  },

  "preprocessing": {
    "enable_denoise": true,
    "enable_enhancement": true,
    "enable_color_correction": true
  },

  "fusion": {
    "strategy": "adaptive",
    "rgb_weight_day": 0.8,
    "ir_weight_day": 0.2,
    "rgb_weight_night": 0.2,
    "ir_weight_night": 0.8
  },

  "logging": {
    "level": "info",
    "save_log": true,
    "log_path": "/var/log/ocfa_face.log"
  }
}
```

**说明**:
- 删除了 `database` 配置项，特征库管理由调用者负责
- 删除了 `match_1to1` 和 `match_1toN` 阈值配置，由调用者在搜索时指定
- 保留核心模型和推理配置

## 7. 性能预估与优化

### 7.1 性能分析 (基于固化方案)

**固化方案配置**:
- 活体检测: MiniFASNet 双流融合 (A2)
- 特征提取: ArcFace-R34 (B1)
- 特征融合: 自适应融合 (C2)
- 特征检索: NEON 优化线性搜索 (D2)
- 推理引擎: 纯 NNIE (E1)
- 量化策略: INT8 PTQ (F1)

| 处理阶段 | 耗时 | 运行设备 | 备注 |
|---------|------|---------|------|
| 图像采集与同步 | 5ms | Camera ISP | 硬件同步 |
| 图像预处理 | 8ms | CPU | RGB(5ms) + IR(3ms) |
| 活体检测 | 15ms | NNIE | MiniFASNet双流 INT8 |
| 质量评估 | 2ms | CPU | 多维度评估 |
| 特征提取(RGB) | 35ms | NNIE | ArcFace-R34 INT8 |
| 特征提取(IR) | 35ms | NNIE | ArcFace-R34 INT8 |
| 特征融合 | 1ms | CPU | 自适应融合 |
| 特征比对(1000人) | 0.5ms | CPU | NEON 优化 |
| 决策输出 | 0.5ms | CPU | 阈值判断 |
| **总计 (串行)** | **102ms** | - | **~9.8fps** |

### 7.2 性能优化策略

#### 优化方向

| 优化项 | 方法 | 收益 | 可行性 |
|-------|------|------|--------|
| 特征提取并行 | RGB+IR 并行推理 | 35ms → 35ms (省35ms) | 需 NNIE 支持 |
| Pipeline 优化 | 多帧重叠处理 | 整体提速20% | 高 |
| NEON 加速 | 预处理/比对优化 | 省5ms | 高 |
| 跳帧检测 | 隔帧活体检测 | 省4ms | 高 |
| 降低分辨率 | 720p → 480p | 省2ms | 中 |

#### 优化后性能

| 配置 | 活体模型 | 特征模型 | 优化措施 | 总耗时 | FPS |
|------|---------|---------|---------|-------|-----|
| 基线 | MiniFASNet双流 | ArcFace-R34 | 无 | 102ms | **9.8fps** |
| 优化1 | MiniFASNet双流 | ArcFace-R34 | 并行推理 | 67ms | **14.9fps** ✓ |
| 优化2 | MiniFASNet双流 | ArcFace-R34 | 并行+Pipeline | 55ms | **18fps** |

**推荐配置**: 优化1 (RGB/IR 特征提取并行推理)，满足 ≥10fps 性能要求

### 7.3 内存占用预估

| 项目 | 大小 | 备注 |
|------|------|------|
| 活体检测模型 | ~8MB | MiniFASNet双流 INT8 |
| 特征提取模型(RGB) | ~22MB | ArcFace-R34 INT8 |
| 特征提取模型(IR) | ~22MB | ArcFace-R34 INT8 |
| NNIE 运行时 | ~10MB | 驱动 + 缓冲区 |
| 图像缓冲区 | ~8MB | RGB+IR 多帧缓存 |
| 特征库 (1000人) | ~2MB | 512维×4字节×1000 |
| 特征库 (5000人) | ~10MB | 512维×4字节×5000 |
| SDK 代码 + 依赖 | ~8MB | 核心代码库 |
| 运行时堆栈 | ~5MB | 临时数据 |
| **总计 (1000人)** | **~85MB** | **符合 128MB 限制** ✓ |
| **总计 (5000人)** | **~93MB** | **符合 128MB 限制** ✓ |

**内存优化**:
- RGB 和 IR 特征模型共享权重可节省 22MB
- 动态加载模型可节省约 40MB（但增加启动时间）
- 特征库持久化到 Flash，按需加载热门用户

### 7.4 存储空间需求

| 项目 | 大小 |
|------|------|
| 模型文件 | ~54MB |
| 特征库 (10000人) | ~20MB |
| 配置文件 | <1MB |
| 日志文件 | ~10MB |
| **总计** | **~85MB** |

**推荐 Flash 容量**: 128MB 及以上

## 8. 开发工具链

### 8.1 模型获取

**InsightFace 预训练模型下载**:
```bash
# ArcFace-R34 模型
# 下载地址: https://github.com/deepinsight/insightface/tree/master/model_zoo

# 模型文件:
# - glint360k_r34.pth (PyTorch权重)
# - ms1mv2_r34_fp16_backbone.pth
```

**MiniFASNet 活体检测模型**:
```bash
# 下载地址: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

# 模型文件:
# - 2.7_80x80_MiniFASNetV2.pth
```

### 8.2 模型转换工具链

```bash
# Step 1: PyTorch → ONNX
python tools/export_onnx.py \
  --model arcface_r34 \
  --weights glint360k_r34.pth \
  --input-size 112 112 \
  --output arcface_r34.onnx

# Step 2: ONNX → Caffe
python tools/onnx2caffe.py \
  --onnx arcface_r34.onnx \
  --output arcface_r34

# Step 3: 准备校准数据集
# calibration_data/ 目录包含 1000 张 112x112 的人脸图片

# Step 4: Caffe → NNIE (使用 nnie_mapper)
./nnie_mapper \
  --prototxt=arcface_r34.prototxt \
  --caffemodel=arcface_r34.caffemodel \
  --calibration_data=calibration_data/ \
  --image_list=calib_list.txt \
  --image_type=0 \
  --norm_type=0 \
  --mean_file=mean.txt \
  --quantize_type=int8 \
  --output=arcface_r34_int8.wk

# Step 5: 验证 NNIE 模型
./nnie_simulator \
  --model=arcface_r34_int8.wk \
  --input=test_image.jpg \
  --output=output.bin
```

### 8.3 海思工具链

| 工具 | 功能 | 版本要求 |
|------|------|---------|
| RuyiStudio | 模型转换IDE | 最新版 |
| nnie_mapper | 命令行转换工具 | Hi3516CV610 SDK |
| nnie_simulator | NNIE 模拟器 | Hi3516CV610 SDK |
| Hi3516CV610 SDK | 板端开发SDK | v2.0+ |

### 8.4 开发环境

**交叉编译环境**:
```bash
# 编译器
arm-linux-gnueabihf-gcc 7.5.0

# 交叉编译工具链
export PATH=$PATH:/path/to/hi3516cv610_sdk/toolchain/bin
export CROSS_COMPILE=arm-linux-gnueabihf-
```

**依赖库**:
- OpenCV 3.4+ (交叉编译版本)
- nlohmann/json (配置解析)
- Hi3516CV610 NNIE SDK

## 9. 开发路线图

### Phase 1: 环境搭建与模型准备 (2周)
- [ ] 申请并安装海思 Hi3516CV610 SDK
- [ ] 搭建交叉编译环境
- [ ] 下载 InsightFace ArcFace-R34 预训练模型
- [ ] 下载 MiniFASNet 活体检测模型
- [ ] 准备校准数据集 (从 LFW 采样 1000 张)
- [ ] 准备测试数据集 (LFW + 自采数据)

### Phase 2: 模型转换与量化 (2周)
- [ ] ArcFace-R34 PyTorch → ONNX 转换
- [ ] MiniFASNet PyTorch → ONNX 转换
- [ ] ONNX → Caffe 模型转换
- [ ] Caffe → NNIE (INT8 量化)
- [ ] NNIE simulator 验证
- [ ] 板端精度测试 (LFW 数据集)

### Phase 3: SDK 核心开发 (4-5周)
- [ ] C SDK 框架搭建
- [ ] 图像预处理模块实现
- [ ] 活体检测模块实现 (NNIE 推理)
- [ ] 质量评估模块实现
- [ ] 特征提取模块实现 (NNIE 推理)
- [ ] 特征融合模块实现
- [ ] 特征比对模块实现
- [ ] 特征库管理模块实现
- [ ] 配置文件解析 (JSON)
- [ ] 日志系统

### Phase 4: API 封装与优化 (2周)
- [ ] Linux 风格 API 封装 (ocfa_* 前缀)
- [ ] 错误处理与异常管理
- [ ] Pipeline 并行优化
- [ ] NEON 算子优化 (预处理/比对)
- [ ] 内存优化

### Phase 5: 测试与调优 (2-3周)
- [ ] 单元测试
- [ ] 功能集成测试
- [ ] 性能压力测试
- [ ] LFW 精度测试
- [ ] 真实场景测试 (RGB+IR)
- [ ] 内存泄漏检测
- [ ] 长时间稳定性测试

### Phase 6: 文档与交付 (1周)
- [ ] API 文档编写
- [ ] 用户手册编写
- [ ] 开发者指南编写
- [ ] 示例代码编写
- [ ] 部署手册编写

**总计**: 13-17 周（约 3-4 个月）

## 10. 风险与应对

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| NNIE 工具链不稳定 | 模型转换失败 | 中 | 使用 ONNX Runtime CPU 备用 |
| INT8 量化精度损失大 | 识别率下降 | 中 | 调整量化参数或使用 INT16 |
| 内存超限 (128MB) | 无法运行 | 低 | RGB/IR 共享权重 + 动态加载 |
| IR 图像质量差 | 活体检测失效 | 中 | 降低 IR 权重或仅使用 RGB |
| 海思 SDK 版本兼容性 | 编译失败 | 低 | 使用官方推荐版本 |
| NNIE 不支持并行推理 | 性能不达标 | 中 | Pipeline 优化弥补 |
| 特征库检索速度慢 | 延迟高 | 低 | NEON 优化 |

## 11. 技术选型总结（已固化）

### 最终确定方案

| 项目 | 选择 | 技术细节 | 性能指标 |
|------|------|---------|---------|
| **芯片平台** | Hi3516CV610 | ARM A17 双核 800MHz + NNIE 3.0 | 1T ops@INT8 |
| **活体检测模型** | MiniFASNet 双流融合 (A2) | RGB + IR 双分支 | 15ms@NNIE |
| **特征提取模型** | ArcFace-R34 (B1) | ResNet-34 骨干网络 | 35ms@NNIE |
| **特征融合策略** | 自适应融合 (C2) | 基于光照和质量的动态加权 | 1ms@CPU |
| **特征库检索** | NEON 优化线性搜索 (D2) | ARM SIMD 加速 | 2ms@5000人 |
| **推理引擎** | 纯 NNIE (E1) | 硬件加速 | - |
| **量化策略** | INT8 PTQ (F1) | 后训练量化 | 1-2% 精度损失 |
| **API 风格** | Linux 风格 (ocfa_*) | 符合嵌入式开发习惯 | - |
| **配置格式** | JSON | 易读易写，广泛支持 | - |

### 性能指标总结

**基线性能** (串行执行):
- **总延迟**: 102ms
- **帧率**: 9.8fps
- **内存占用**: 85MB (1000人) / 93MB (5000人)
- **存储空间**: 85MB

**优化性能** (并行执行):
- **总延迟**: 67ms
- **帧率**: 14.9fps ✓ (满足 ≥10fps 要求)
- **内存占用**: 不变
- **优化手段**: RGB/IR 特征提取并行推理

### 方案特点

**优势**:
1. RGB-IR 双模态活体检测，安全性高
2. ArcFace-R34 精度达 99.5% (LFW)
3. 自适应融合适应不同光照环境
4. NEON 优化支持 5000 人规模
5. 纯 NNIE 推理，性能最优
6. 内存占用 < 100MB，符合限制

**适用场景**:
- 门禁考勤系统
- 访客管理系统
- 安防监控系统
- 智能楼宇系统

## 12. 参考资料

### 核心算法
- InsightFace: https://github.com/deepinsight/insightface
- ArcFace Paper: https://arxiv.org/abs/1801.07698
- MiniFASNet: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- ArcFace Model Zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo

### 数据集
- LFW: http://vis-www.cs.umass.edu/lfw/
- MS1MV2: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
- CASIA-SURF (活体): http://www.cbsr.ia.ac.cn/users/jjyan/CASIA-SURF/

### 海思平台
- Hi3516CV610 产品页: https://www.hisilicon.com/
- Hi3516CV610 SDK: 向海思申请
- NNIE 开发文档: SDK 自带文档

### 工具库
- OpenCV: https://opencv.org/
- nlohmann/json: https://github.com/nlohmann/json
- onnx2caffe: https://github.com/MTlab/onnx2caffe

---

**文档版本**: v5.0 (技术方案固化版)
**创建日期**: 2025-11-10
**更新日期**: 2025-11-10
**芯片平台**: 海思 Hi3516CV610
**API 风格**: Linux 风格 (ocfa_* 前缀)
**模型来源**: InsightFace 开源预训练模型 (不重新训练)

**固化方案**:
- **[A2]** 活体检测: MiniFASNet 双流融合
- **[B1]** 特征提取: ArcFace-R34
- **[C2]** 特征融合: 自适应融合
- **[D2]** 特征检索: NEON 优化线性搜索
- **[E1]** 推理引擎: 纯 NNIE
- **[F1]** 量化策略: INT8 PTQ

**变更说明** (v4.0 → v5.0):
1. 固化技术方案选型，删除待决策选项
2. 更新性能分析为基于固化方案的实际指标
3. 更新内存占用和存储空间预估
4. 简化技术选型总结，突出最终方案

**性能指标**:
- 基线性能: 102ms, 9.8fps
- 优化性能: 67ms, 14.9fps ✓
- 内存占用: 85MB (1000人), 93MB (5000人)

**下一步**:
1. 搭建海思 Hi3516CV610 开发环境
2. 下载 InsightFace 预训练模型 (ArcFace-R34 + MiniFASNet)
3. 准备量化校准数据集 (1000 张)
4. 开始 Phase 1: 环境搭建与模型准备

## 附录: API 使用示例

### A.1 基本使用流程

```c
#include "ocfa_face_sdk.h"

int main() {
    // 1. 初始化 SDK
    ocfa_config_t config = {
        .model_dir = "models",
        .config_file = "config.json",
        .liveness_threshold = 0.90,
        .quality_threshold = 0.50,
        .num_threads = 2
    };

    if (ocfa_init(&config) != OCFA_SUCCESS) {
        printf("SDK 初始化失败\n");
        return -1;
    }

    // 2. 人脸识别（到特征提取）
    uint8_t* rgb_image = ...; // 720p RGB 图像
    uint8_t* ir_image = ...;  // 720p IR 图像

    ocfa_recognition_result_t result;
    int ret = ocfa_recognize(rgb_image, ir_image, 1280, 720, &result);

    if (ret != OCFA_SUCCESS) {
        printf("识别失败: %s\n", result.error_msg);
        return -1;
    }

    // 3. 检查活体和质量
    if (!result.liveness_passed) {
        printf("活体检测未通过: %.2f\n", result.liveness_score);
        return -1;
    }

    if (!result.quality_passed) {
        printf("质量评估未通过: %.2f\n", result.quality_score);
        return -1;
    }

    // 4. 获取特征向量
    float* feature = result.feature; // 512 维特征
    printf("特征提取成功，耗时: %u ms\n", result.total_time_ms);

    // 5. 特征比对由调用者实现
    // ... (见下面的具体示例)

    // 6. 释放 SDK
    ocfa_release();

    return 0;
}
```

### A.2 添加用户到特征库

```c
// 添加用户
int register_user(const uint8_t user_id[16], const float* feature) {
    // 添加到 SDK 内部特征库
    int ret = ocfa_add_user(user_id, feature);
    if (ret != OCFA_SUCCESS) {
        printf("添加用户失败: %d\n", ret);
        return ret;
    }

    // 调用者自行保存用户元数据（姓名、照片等）
    // save_user_metadata(user_id, name, photo, ...);

    // 调用者自行持久化特征数据
    // save_feature_to_file(user_id, feature);

    return OCFA_SUCCESS;
}

// 更新用户特征
int update_user_feature(const uint8_t user_id[16], const float* new_feature) {
    // 更新 SDK 内部特征库
    int ret = ocfa_update_user(user_id, new_feature);
    if (ret != OCFA_SUCCESS) {
        if (ret == OCFA_ERROR_USER_NOT_FOUND) {
            printf("用户不存在: ");
            for (int i = 0; i < 16; i++) printf("%02x", user_id[i]);
            printf("\n");
        } else {
            printf("更新用户失败: %d\n", ret);
        }
        return ret;
    }

    // 调用者自行更新持久化存储
    // update_feature_in_file(user_id, new_feature);

    return OCFA_SUCCESS;
}
```

### A.3 1:1 验证（查找最相似用户）

```c
// 验证用户身份
int verify_user(const float* query_feature) {
    uint8_t matched_id[16];
    float similarity;

    // 查找最相似的用户
    int ret = ocfa_search_user(query_feature, matched_id, &similarity);
    if (ret != OCFA_SUCCESS) {
        printf("搜索失败\n");
        return ret;
    }

    // 判断相似度
    float threshold = 0.70; // 1:1 验证阈值
    if (similarity >= threshold) {
        printf("验证通过: 用户ID=");
        for (int i = 0; i < 16; i++) printf("%02x", matched_id[i]);
        printf(", 相似度=%.3f\n", similarity);

        // 调用者获取用户元数据
        // UserInfo* info = load_user_metadata(matched_id);
        return OCFA_SUCCESS;
    } else {
        printf("验证失败: 相似度过低 %.3f < %.3f\n", similarity, threshold);
        return OCFA_ERROR_NO_MATCH;
    }
}
```

### A.4 1:N 识别（查找多个相似用户）

```c
// 识别用户（1:N）
int identify_user(const float* query_feature) {
    float threshold = 0.70;  // 相似度阈值
    int max_results = 5;     // 最多返回 5 个用户

    // 预分配结果数组
    ocfa_search_result_t results[5];

    // 搜索相似用户
    int count = ocfa_search_users(query_feature, threshold, results, max_results);
    if (count < 0) {
        printf("搜索失败\n");
        return count;
    }

    if (count == 0) {
        printf("未找到匹配用户（相似度 < %.2f）\n", threshold);
        return OCFA_ERROR_NO_MATCH;
    }

    // 输出结果（按相似度降序排列）
    printf("找到 %d 个匹配用户:\n", count);
    for (int i = 0; i < count; i++) {
        printf("  %d. 用户ID=", i+1);
        for (int j = 0; j < 16; j++) printf("%02x", results[i].user_id[j]);
        printf(", 相似度=%.3f\n", results[i].similarity);

        // 调用者获取用户元数据
        // UserInfo* info = load_user_metadata(results[i].user_id);
        // printf("     姓名: %s\n", info->name);
    }

    return OCFA_SUCCESS;
}
```

### A.5 删除用户

```c
// 删除用户
int delete_user(const uint8_t user_id[16]) {
    // 从 SDK 内部特征库删除
    int ret = ocfa_remove_user(user_id);
    if (ret != OCFA_SUCCESS) {
        printf("删除用户失败: %d\n", ret);
        return ret;
    }

    // 调用者自行删除用户元数据
    // delete_user_metadata(user_id);

    // 调用者自行从持久化存储删除
    // delete_feature_from_file(user_id);

    printf("用户 ");
    for (int i = 0; i < 16; i++) printf("%02x", user_id[i]);
    printf(" 已删除\n");
    return OCFA_SUCCESS;
}
```

### A.6 完整示例：门禁系统

```c
// 门禁系统示例
void access_control_system() {
    // 初始化 SDK
    ocfa_config_t config = {...};
    ocfa_init(&config);

    // 调用者自行加载特征库
    // load_feature_database("face_db.bin");

    while (1) {
        // 采集 RGB + IR 图像
        uint8_t* rgb = capture_rgb_image();
        uint8_t* ir = capture_ir_image();

        // 人脸识别（到特征提取）
        ocfa_recognition_result_t result;
        if (ocfa_recognize(rgb, ir, 1280, 720, &result) != OCFA_SUCCESS) {
            continue;
        }

        // 检查活体和质量
        if (!result.liveness_passed || !result.quality_passed) {
            printf("活体或质量检测未通过\n");
            continue;
        }

        // 1:N 识别
        ocfa_search_result_t results[5];
        int count = ocfa_search_users(result.feature, 0.70, results, 5);

        if (count > 0) {
            uint8_t* matched_user_id = results[0].user_id;  // 16字节用户ID
            float similarity = results[0].similarity;

            // 获取用户信息（传入16字节ID）
            UserInfo* info = load_user_metadata(matched_user_id);

            // 检查权限
            if (check_permission(info)) {
                printf("欢迎 %s! (相似度: %.3f)\n", info->name, similarity);
                open_door();
            } else {
                printf("无权限访问\n");
            }
        } else {
            printf("未识别用户\n");
        }

        usleep(100000); // 100ms
    }

    // 释放 SDK
    ocfa_release();
}
```
