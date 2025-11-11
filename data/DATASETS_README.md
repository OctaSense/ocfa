# OCFA Face SDK - Public Datasets Guide

本文档说明如何获取和使用公开的RGB-IR人脸数据集用于开发测试。

---

## 可用数据集概览

### 1. UNCC ThermalFace Dataset ⭐ 推荐

**描述:** 约10,000张热成像人脸图像，10个受试者

**特点:**
- 热成像图像（16-bit radiometric）
- 手动标注72或43个面部关键点
- 包含环境参数（温度、湿度、距离等）
- 4个温度范围（20.6°C - 26.6°C）
- 10种距离变化（1m - 6.6m）
- 25种头部姿态

**大小:** ~5-10 GB

**下载方法:**

```bash
# 方法1: 使用我们的脚本（推荐）
source venv/bin/activate
python data/download_public_datasets.py --dataset uncc_thermal

# 方法2: 直接从Google Drive下载
# 访问: https://drive.google.com/drive/folders/1UE4BsRsJBSEyLtcAkMQ_jYIBKwKjwyQc
# 手动下载到 data/datasets/uncc_thermal/
```

**引用:**
```bibtex
@article{khatri2022novel,
  title={A Novel Fully Annotated Thermal Infrared Face Dataset},
  author={Khatri, Ravi and others},
  journal={IEEE Access},
  year={2022}
}
```

**GitHub:** https://github.com/TeCSAR-UNCC/UNCC-ThermalFace

---

### 2. OTCBVS Benchmark Dataset

**描述:** 4,228对RGB-热成像图像，30个受试者

**特点:**
- 同时采集的可见光和热成像图像
- 多种光照条件
- 表情和姿态变化
- 经典的RGB-IR基准数据集

**大小:** ~500 MB

**下载方法:**

```bash
# 需要注册后手动下载
# 1. 访问: https://vcipl-okstate.org/pbvs/bench/
# 2. 注册账号
# 3. 下载数据集
# 4. 解压到 data/datasets/otcbvs/
```

**引用:**
```bibtex
@inproceedings{chen2005otcbvs,
  title={The OTCBVS benchmark dataset collection},
  author={Chen, Xiaoyang and Flynn, Patrick J and Bowyer, Kevin W},
  booktitle={CVPR Workshop},
  year={2005}
}
```

---

### 3. TFW: Thermal Faces in the Wild

**描述:** 野外环境下的热成像人脸数据集

**特点:**
- 真实场景采集
- 多样化的环境条件
- 标注的人脸区域

**大小:** 可变

**下载方法:**

```bash
# 从GitHub克隆
git clone https://github.com/IS2AI/TFW.git data/datasets/tfw/

# 或使用我们的脚本
python data/download_public_datasets.py --dataset tfw
```

**GitHub:** https://github.com/IS2AI/TFW

---

### 4. Tufts Face Database (Kaggle)

**描述:** 多模态人脸数据库，~100K图像，112人

**特点:**
- 包含可见光、近红外、热成像等多种模态
- 大规模数据集
- 适合深度学习训练

**大小:** ~10+ GB

**下载方法:**

```bash
# 需要Kaggle账号和API密钥

# 1. 安装Kaggle CLI
pip install kaggle

# 2. 配置API密钥 (从 kaggle.com/settings 获取)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 3. 下载数据集
kaggle datasets download -d kpvisionlab/tufts-face-database
unzip tufts-face-database.zip -d data/datasets/tufts/
```

**Kaggle链接:** https://www.kaggle.com/datasets/kpvisionlab/tufts-face-database

---

### 5. SYSU-MM01 Dataset

**描述:** 22K可见光 + 11K近红外图像，491个身份

**特点:**
- 大规模跨模态数据集
- 4个RGB相机 + 2个NIR相机
- 适合跨模态人脸识别研究

**大小:** ~2-3 GB

**下载方法:**

```bash
# 需要联系作者获取下载权限
# GitHub: https://github.com/wuancong/SYSU-MM01
# 发送邮件申请访问
```

---

### 6. LFW (Labeled Faces in the Wild)

**描述:** 标准人脸识别基准数据集（仅RGB）

**特点:**
- 13,000+ 图像，5,000+ 人
- 标准的人脸识别评测基准
- 广泛使用的测试集

**大小:** ~180 MB

**下载方法:**

```bash
# 直接下载
python data/download_public_datasets.py --dataset lfw

# 或手动下载
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz -C data/datasets/
```

**官网:** http://vis-www.cs.umass.edu/lfw/

---

## 快速开始

### 1. 下载推荐的测试数据集

```bash
# 激活虚拟环境
source venv/bin/activate

# 列出所有可用数据集
python data/download_public_datasets.py --list

# 下载UNCC ThermalFace数据集（热成像）
python data/download_public_datasets.py --dataset uncc_thermal

# 下载LFW数据集（RGB基准）
python data/download_public_datasets.py --dataset lfw
```

### 2. 数据集目录结构

下载后的数据集会组织成如下结构：

```
data/
├── datasets/                    # 公开数据集
│   ├── uncc_thermal/           # UNCC ThermalFace
│   │   ├── Subject_01/
│   │   ├── Subject_02/
│   │   └── ...
│   ├── lfw/                    # LFW (RGB)
│   │   ├── Person_A/
│   │   ├── Person_B/
│   │   └── ...
│   ├── tfw/                    # TFW
│   └── datasets_info.json      # 数据集信息
│
├── samples/                     # 测试样本（小）
│   ├── rgb/
│   └── ir/
│
├── calibration/                 # 校准数据
└── calibration_80x80/          # MiniFASNet校准数据
```

### 3. 在SDK中使用数据集

```python
from ocfa import OCFAFaceSDK
import cv2
import glob

# 初始化SDK
sdk = OCFAFaceSDK(config_path='configs/default_config.json')

# 加载UNCC ThermalFace数据集
dataset_dir = 'data/datasets/uncc_thermal/Subject_01'

# 遍历热成像图像
for thermal_path in glob.glob(f'{dataset_dir}/*.png'):
    # 加载热成像图像
    thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)

    # 创建伪RGB图像（用于测试）
    rgb_image = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2RGB)

    # 运行识别
    result = sdk.recognize(rgb_image, thermal_image)

    if result.success:
        print(f"Image: {thermal_path}")
        print(f"  Liveness: {result.liveness_score:.3f}")
        print(f"  Quality: {result.quality_score:.3f}")
```

### 4. 批量处理数据集

```bash
# 使用SDK工具批量评估
python python/tools/evaluate.py \
    --data data/datasets/uncc_thermal \
    --task liveness \
    --output evaluation_results.json

# 性能基准测试
python python/tools/benchmark.py \
    --data data/datasets/uncc_thermal \
    --samples 100 \
    --output benchmark_results.json
```

---

## 数据集对比

| 数据集 | 大小 | 模态 | 人数 | 图像数 | 标注 | 推荐用途 |
|--------|------|------|------|--------|------|----------|
| UNCC ThermalFace | ~5-10 GB | Thermal | 10 | 10K | 关键点 | ⭐ 热成像测试 |
| OTCBVS | ~500 MB | RGB+Thermal | 30 | 4.2K | 无 | RGB-IR配对 |
| TFW | 可变 | Thermal | 多 | 可变 | 边界框 | 野外场景 |
| Tufts | ~10+ GB | 多模态 | 112 | 100K | 多种 | 大规模训练 |
| SYSU-MM01 | ~2-3 GB | RGB+NIR | 491 | 33K | 身份 | 跨模态识别 |
| LFW | ~180 MB | RGB | 5K+ | 13K | 身份 | 基准评测 |

---

## 数据预处理建议

### 热成像图像预处理

```python
import cv2
import numpy as np

def preprocess_thermal(thermal_image):
    """预处理热成像图像"""
    # 归一化到0-255
    normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    # 直方图均衡化
    equalized = cv2.equalizeHist(normalized)

    # 调整大小
    resized = cv2.resize(equalized, (112, 112))

    return resized
```

### RGB-IR配对验证

```python
def verify_rgb_ir_pair(rgb_path, ir_path):
    """验证RGB和IR图像配对"""
    import os

    # 检查文件存在
    if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
        return False

    # 检查图像可读
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None or ir is None:
        return False

    # 检查尺寸匹配
    if rgb.shape[:2] != ir.shape[:2]:
        print(f"Warning: Size mismatch - RGB: {rgb.shape}, IR: {ir.shape}")
        return False

    return True
```

---

## 常见问题

### Q1: 数据集太大，如何下载部分数据？

对于Google Drive数据集（如UNCC ThermalFace），可以：
1. 访问Google Drive链接
2. 只选择部分文件夹下载
3. 或使用gdown命令指定具体文件

```bash
# 下载特定文件
gdown --id <file_id> -O data/datasets/
```

### Q2: 没有真实的IR相机怎么办？

可以使用以下替代方案：
1. 使用灰度图像模拟IR图像（用于算法测试）
2. 使用NIR（近红外）图像
3. 使用深度图（Depth map）
4. 下载公开的RGB-IR数据集进行测试

### Q3: 如何创建自己的RGB-IR数据集？

```bash
# 1. 采集图像（需要RGB和IR相机）
# 2. 组织目录结构
mkdir -p data/custom_dataset/{rgb,ir}

# 3. 确保文件命名对应
# RGB: data/custom_dataset/rgb/person_001_001.jpg
# IR:  data/custom_dataset/ir/person_001_001.jpg

# 4. 生成校准数据
python python/tools/create_calibration_data.py \
    --dual-stream \
    --num-samples 100 \
    --output data/custom_dataset/calibration
```

### Q4: 数据集许可证问题？

所有列出的数据集均为研究用途，请遵守各自的许可证：
- 学术研究：通常免费使用
- 商业应用：需要联系原作者获取授权
- 引用：使用数据集请在论文中引用

---

## 数据集贡献

如果您有公开的RGB-IR人脸数据集希望添加到本列表：

1. Fork本项目
2. 在`data/download_public_datasets.py`中添加数据集配置
3. 更新本README文档
4. 提交Pull Request

---

## 参考资源

- **UNCC ThermalFace:** https://github.com/TeCSAR-UNCC/UNCC-ThermalFace
- **OTCBVS:** https://vcipl-okstate.org/pbvs/bench/
- **TFW:** https://github.com/IS2AI/TFW
- **Tufts:** https://www.kaggle.com/datasets/kpvisionlab/tufts-face-database
- **SYSU-MM01:** https://github.com/wuancong/SYSU-MM01
- **LFW:** http://vis-www.cs.umass.edu/lfw/

---

**更新日期:** 2025-01-11
**维护者:** OCFA Face SDK Team
