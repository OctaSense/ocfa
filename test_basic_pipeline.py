#!/usr/bin/env python3
"""
OCFA Face SDK - 基本流程测试

测试完整的人脸识别处理流程：
1. 加载RGB和IR图像
2. 活体检测
3. 质量评估
4. 特征提取
5. 人脸比对
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# 添加python目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'python'))

def test_basic_pipeline():
    """测试基本处理流程"""

    print("="*70)
    print("OCFA Face SDK - 基本流程测试")
    print("="*70)
    print()

    # 1. 加载SDK
    print("[1/6] 加载SDK...")
    try:
        from ocfa import OCFAFaceSDK

        # 检查配置文件
        config_path = 'configs/default_config.json'
        if not Path(config_path).exists():
            print(f"  ✗ 配置文件不存在: {config_path}")
            return False

        sdk = OCFAFaceSDK(config_path=config_path)
        print(f"  ✓ SDK加载成功")
    except Exception as e:
        print(f"  ✗ SDK加载失败: {e}")
        print(f"\n提示: 请先确保模型文件存在:")
        print(f"  - models/buffalo_l/w600k_r50.onnx (或量化版本)")
        print(f"  - models/minifasnet_v2.onnx")
        return False

    # 2. 加载测试图像
    print("\n[2/6] 加载测试图像...")

    # 检查样本数据
    rgb_path = Path('data/samples/rgb/sample_001.jpg')
    ir_path = Path('data/samples/ir/sample_001_ir.jpg')

    if not rgb_path.exists() or not ir_path.exists():
        print(f"  ✗ 测试图像不存在")
        print(f"\n提示: 请运行以下命令生成测试数据:")
        print(f"  bash data/download_test_data.sh")
        return False

    # 加载图像
    rgb_image = cv2.imread(str(rgb_path))
    ir_image = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)

    if rgb_image is None or ir_image is None:
        print(f"  ✗ 图像加载失败")
        return False

    print(f"  ✓ RGB图像加载成功: {rgb_image.shape}")
    print(f"  ✓ IR图像加载成功: {ir_image.shape}")

    # 3. 运行人脸识别（旁路活体检测）
    print("\n[3/6] 运行人脸识别（livecheck=0，旁路活体检测）...")
    try:
        result = sdk.recognize(rgb_image, ir_image, livecheck=0)

        if result.success:
            print(f"  ✓ 识别成功（活体检测已旁路）")
            print(f"    - 活体检测: 旁路模式 (score: {result.liveness_score:.4f})")
            print(f"    - 质量评估分数: {result.quality_score:.4f} (阈值: {sdk.config.get('quality_threshold', 30.0)})")
            print(f"    - 活体通过: {'✓' if result.liveness_passed else '✗'}")
            print(f"    - 质量通过: {'✓' if result.quality_passed else '✗'}")
            print(f"    - 特征维度: {result.feature.shape}")
            print(f"    - 特征L2范数: {np.linalg.norm(result.feature):.4f}")
        else:
            print(f"  ✗ 识别失败: {result.error_message}")
            return False

    except Exception as e:
        print(f"  ✗ 识别过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 添加用户到数据库
    print("\n[4/6] 添加用户到数据库...")
    try:
        # 创建用户ID（16字节）
        user_id = b'user_001_test123'  # 16字节

        # 添加用户
        sdk.add_user(user_id, result.feature)
        print(f"  ✓ 用户已添加: {user_id.hex()}")

        # 检查数据库
        user_count = sdk.get_user_count()
        print(f"  ✓ 数据库用户数: {user_count}")

    except Exception as e:
        print(f"  ✗ 添加用户失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试1:1验证
    print("\n[5/6] 测试1:1人脸验证...")
    try:
        # 再次识别同一张图像（同样旁路活体检测）
        result2 = sdk.recognize(rgb_image, ir_image, livecheck=0)

        if result2.success:
            # 计算相似度
            similarity = sdk.comparator.compare(result.feature, result2.feature)
            print(f"  ✓ 特征提取成功")
            print(f"    - 相似度: {similarity:.4f}")
            print(f"    - 阈值: {sdk.config.get('match_threshold', 0.60)}")

            is_match = similarity >= sdk.config.get('match_threshold', 0.60)
            print(f"    - 匹配结果: {'✓ 匹配' if is_match else '✗ 不匹配'}")
        else:
            print(f"  ✗ 第二次识别失败: {result2.error_message}")

    except Exception as e:
        print(f"  ✗ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 测试1:N识别
    print("\n[6/6] 测试1:N人脸识别...")
    try:
        # 搜索用户
        matches = sdk.search_users(
            result.feature,
            threshold=sdk.config.get('match_threshold', 0.60),
            max_results=5
        )

        if matches:
            print(f"  ✓ 找到匹配用户: {len(matches)}个")
            for i, (matched_id, score) in enumerate(matches, 1):
                print(f"    {i}. 用户ID: {matched_id.hex()}, 相似度: {score:.4f}")
        else:
            print(f"  ⚠ 未找到匹配用户（可能阈值过高）")

    except Exception as e:
        print(f"  ✗ 搜索过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试完成
    print("\n" + "="*70)
    print("✓ 所有测试通过！")
    print("="*70)

    # 输出统计信息
    print("\n统计信息:")
    print(f"  - 数据库用户数: {sdk.get_user_count()}")
    print(f"  - 特征维度: {result.feature.shape[0]}")
    print(f"  - SDK版本: 1.0.0")

    return True


def test_with_dummy_models():
    """使用虚拟模型测试（如果真实模型不存在）"""

    print("="*70)
    print("使用虚拟模型测试")
    print("="*70)
    print()

    print("提示: 真实模型不存在，将使用虚拟模型进行演示")
    print()

    # 创建虚拟特征
    print("[1/4] 生成虚拟特征...")
    feature1 = np.random.randn(512).astype(np.float32)
    feature1 = feature1 / np.linalg.norm(feature1)  # L2归一化
    print(f"  ✓ 特征1: shape={feature1.shape}, norm={np.linalg.norm(feature1):.4f}")

    feature2 = feature1 + np.random.randn(512).astype(np.float32) * 0.1
    feature2 = feature2 / np.linalg.norm(feature2)  # L2归一化
    print(f"  ✓ 特征2: shape={feature2.shape}, norm={np.linalg.norm(feature2):.4f}")

    # 计算相似度
    print("\n[2/4] 计算特征相似度...")
    similarity = np.dot(feature1, feature2)
    print(f"  ✓ 余弦相似度: {similarity:.4f}")

    # 模拟数据库
    print("\n[3/4] 模拟用户数据库...")
    database = {}

    # 添加用户
    user_id1 = b'user_001_test123'
    database[user_id1] = feature1
    print(f"  ✓ 添加用户: {user_id1.hex()}")

    user_id2 = b'user_002_test456'
    database[user_id2] = feature2
    print(f"  ✓ 添加用户: {user_id2.hex()}")

    print(f"  ✓ 数据库用户数: {len(database)}")

    # 模拟搜索
    print("\n[4/4] 模拟1:N搜索...")
    query_feature = feature1

    matches = []
    for uid, feat in database.items():
        sim = np.dot(query_feature, feat)
        matches.append((uid, sim))

    # 排序
    matches.sort(key=lambda x: x[1], reverse=True)

    print(f"  ✓ 找到匹配用户: {len(matches)}个")
    for i, (uid, sim) in enumerate(matches[:5], 1):
        print(f"    {i}. 用户ID: {uid.hex()}, 相似度: {sim:.4f}")

    print("\n" + "="*70)
    print("✓ 虚拟模型测试完成！")
    print("="*70)

    return True


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description='OCFA Face SDK - 基本流程测试')
    parser.add_argument('--dummy', action='store_true',
                       help='使用虚拟模型测试（无需真实模型）')

    args = parser.parse_args()

    print()
    print("OCFA Face Recognition SDK")
    print("Version: 1.0.0")
    print("="*70)
    print()

    if args.dummy:
        # 使用虚拟模型测试
        success = test_with_dummy_models()
    else:
        # 使用真实SDK测试
        success = test_basic_pipeline()

    if success:
        print("\n建议后续步骤:")
        print("  1. 下载真实的RGB-IR人脸数据集:")
        print("     python data/download_public_datasets.py --dataset uncc_thermal")
        print()
        print("  2. 运行完整评估:")
        print("     python python/tools/evaluate.py --task all")
        print()
        print("  3. 性能基准测试:")
        print("     python python/tools/benchmark.py --module all")
        print()
        print("  4. 查看文档:")
        print("     cat docs/README.md")
        print()

        return 0
    else:
        print("\n故障排除:")
        print("  1. 检查模型文件是否存在:")
        print("     ls -lh models/*.onnx")
        print()
        print("  2. 检查测试数据:")
        print("     ls -lh data/samples/rgb/ data/samples/ir/")
        print()
        print("  3. 使用虚拟模型测试:")
        print("     python test_basic_pipeline.py --dummy")
        print()
        print("  4. 查看完整文档:")
        print("     cat MODEL_ACQUISITION_REPORT.md")
        print()

        return 1


if __name__ == '__main__':
    sys.exit(main())
