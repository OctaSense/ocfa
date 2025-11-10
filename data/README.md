# 测试数据集目录

本目录存放用于测试和验证的人脸图像数据。

## 目录结构

```
data/
├── calibration/           # 量化校准数据集 (~1000张)
├── test/                  # 测试数据集
│   ├── rgb/              # RGB 测试图像
│   ├── ir/               # IR 测试图像
│   └── pairs.txt         # 测试图像对列表
└── samples/              # 示例图像
    ├── live/             # 真实人脸样本
    └── fake/             # 伪造攻击样本
```

## 数据集说明

### 1. 校准数据集 (calibration/)

**用途**: 用于 INT8 量化的校准，生成量化参数

**规模**: ~1000 张图像

**要求**:
- RGB + IR 图像对
- 包含不同人种、年龄、性别
- 包含不同光照条件 (白天/夜晚)
- 包含不同表情和姿态
- 图像质量良好，无模糊遮挡

**格式**:
```
calibration/
├── 0001_rgb.jpg
├── 0001_ir.jpg
├── 0002_rgb.jpg
├── 0002_ir.jpg
...
```

**采集建议**:
- 每人采集 2-3 张不同角度的图像
- 涵盖目标应用场景的光照条件
- 确保 RGB 和 IR 图像时间同步

### 2. 测试数据集 (test/)

**用途**: 用于模型精度评估和性能测试

**规模**: 根据需要准备，建议 > 5000 对

**内容**:
- RGB 图像: `test/rgb/*.jpg`
- IR 图像: `test/ir/*.jpg`
- 测试对列表: `test/pairs.txt`

**pairs.txt 格式**:
```
# 格式: rgb_path ir_path label user_id
test/rgb/0001.jpg test/ir/0001.jpg 1 user_0001
test/rgb/0002.jpg test/ir/0002.jpg 1 user_0002
test/rgb/0003.jpg test/ir/0003.jpg 0 attack_photo
test/rgb/0004.jpg test/ir/0004.jpg 0 attack_screen
...
```

- `label`: 1=真实人脸, 0=伪造攻击
- `user_id`: 用户标识符 (可选)

### 3. 示例图像 (samples/)

**用途**: 快速测试和演示

**内容**:
- `samples/live/`: 真实人脸样本 (5-10 对)
- `samples/fake/`: 伪造攻击样本 (5-10 对)

**攻击类型**:
- 照片攻击 (photo attack)
- 视频回放攻击 (video replay)
- 屏幕攻击 (screen attack)
- 面具攻击 (mask attack, 可选)

## 数据准备

### 方式 1: 使用公开数据集

#### LFW (Labeled Faces in the Wild)
- 用途: 人脸识别准确率测试
- 规模: 13,000+ 张图像
- 下载: http://vis-www.cs.umass.edu/lfw/

#### CASIA-SURF
- 用途: RGB-IR 活体检测测试
- 规模: 1,000+ 视频
- 下载: https://sites.google.com/view/face-anti-spoofing-challenge

#### NUAA Imposter Database
- 用途: 活体检测测试
- 规模: 15 个受试者
- 下载: http://parnec.nuaa.edu.cn/xtan/data/nuaa-imposter.html

### 方式 2: 自行采集

**设备要求**:
- RGB 摄像头: 720p, 60° FOV
- IR 摄像头: 720p, 60° FOV, 850nm
- 同步触发器 (可选)

**采集流程**:

```bash
# 使用示例程序采集数据
cd cpp/examples
./demo_camera --mode capture --output ../../data/calibration --count 1000
```

**采集规范**:
1. 确保 RGB 和 IR 图像时间同步 (< 30ms)
2. 人脸区域占比 > 30% 画面
3. 光照均匀，无强烈阴影
4. 避免过曝或欠曝
5. 人脸姿态: yaw < 30°, pitch < 30°

## 数据格式要求

### 图像格式
- RGB: JPEG/PNG, 24位彩色
- IR: JPEG/PNG, 8位灰度
- 分辨率: 720p (1280x720) 或更高

### 命名规范
```
{user_id}_{capture_id}_rgb.jpg
{user_id}_{capture_id}_ir.jpg
```

示例:
```
user_0001_001_rgb.jpg
user_0001_001_ir.jpg
user_0001_002_rgb.jpg
user_0001_002_ir.jpg
```

## 数据增强 (可选)

对于校准数据集，可以进行适当的数据增强:

```python
# 使用工具脚本进行数据增强
cd python/tools
python data_augmentation.py \
    --input ../../data/calibration \
    --output ../../data/calibration_aug \
    --methods flip,rotate,brightness
```

**增强方法**:
- 水平翻转 (flip)
- 轻微旋转 (rotate, ±10°)
- 亮度调整 (brightness, ±20%)
- 对比度调整 (contrast, ±10%)

注意: IR 图像不建议进行色彩增强

## 数据集统计

完成数据准备后，可以使用统计工具检查数据集:

```bash
python python/tools/dataset_stats.py --data-dir data/calibration
```

输出示例:
```
Dataset Statistics:
- Total pairs: 1000
- RGB images: 1000 (1280x720)
- IR images: 1000 (1280x720)
- Average brightness: 128.5
- Face size range: 180-450 pixels
- Pose angle range: -25° to +28°
```

## 数据安全

注意: 人脸数据属于敏感个人信息，请遵守以下规范:

1. **采集**: 获得明确授权
2. **存储**: 加密存储，限制访问权限
3. **使用**: 仅用于开发和测试
4. **销毁**: 测试完成后及时删除
5. **合规**: 遵守 GDPR、个人信息保护法等法规

## .gitignore

为了避免误提交图像数据，已在 `.gitignore` 中排除:

```
data/calibration/*.jpg
data/calibration/*.png
data/test/**/*.jpg
data/test/**/*.png
data/samples/**/*.jpg
data/samples/**/*.png
```

仅提交目录结构和说明文档。

## 参考资料

- [LFW 数据集](http://vis-www.cs.umass.edu/lfw/)
- [CASIA-SURF 数据集](https://sites.google.com/view/face-anti-spoofing-challenge)
- [人脸活体检测综述](https://arxiv.org/abs/2101.00488)
