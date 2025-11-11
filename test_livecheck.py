#!/usr/bin/env python3
"""
OCFA Face SDK - Livecheck Parameter Demo

演示活体检测旁路功能的使用场景：
1. livecheck=1: 默认模式，执行RGB+IR活体检测
2. livecheck=0: 旁路模式，跳过活体检测（适用于受控环境）
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# 添加python目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'python'))


def test_livecheck_modes():
    """测试不同的livecheck模式"""

    print("=" * 70)
    print("OCFA Face SDK - Livecheck 参数演示")
    print("=" * 70)
    print()

    # 1. 加载SDK
    print("[1/4] 加载SDK...")
    try:
        from ocfa import OCFAFaceSDK

        config_path = 'configs/default_config.json'
        if not Path(config_path).exists():
            print(f"  ✗ 配置文件不存在: {config_path}")
            return False

        sdk = OCFAFaceSDK(config_path=config_path)
        print(f"  ✓ SDK加载成功")
    except Exception as e:
        print(f"  ✗ SDK加载失败: {e}")
        return False

    # 2. 加载测试图像
    print("\n[2/4] 加载测试图像...")
    rgb_path = Path('data/samples/rgb/sample_001.jpg')
    ir_path = Path('data/samples/ir/sample_001_ir.jpg')

    if not rgb_path.exists() or not ir_path.exists():
        print(f"  ✗ 测试图像不存在")
        return False

    rgb_image = cv2.imread(str(rgb_path))
    ir_image = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)

    print(f"  ✓ RGB图像: {rgb_image.shape}")
    print(f"  ✓ IR图像: {ir_image.shape}")

    # 3. 测试模式1: livecheck=1（默认，启用活体检测）
    print("\n[3/4] 模式1: livecheck=1 (启用RGB+IR活体检测)")
    print("  使用场景: 安全要求高的场景（门禁、支付等）")
    try:
        result1 = sdk.recognize(rgb_image, ir_image, livecheck=1)

        print(f"  识别结果:")
        print(f"    - 成功: {result1.success}")
        print(f"    - 活体检测分数: {result1.liveness_score:.4f}")
        print(f"    - 活体检测通过: {result1.liveness_passed}")
        print(f"    - 质量评估分数: {result1.quality_score:.4f}")
        print(f"    - 质量评估通过: {result1.quality_passed}")

        if result1.success:
            print(f"    - 特征提取: ✓ ({result1.feature.shape})")
            print(f"    - 处理时间: {result1.total_time_ms:.2f} ms")
        else:
            print(f"    - 失败原因: {result1.error_message}")

    except Exception as e:
        print(f"  ✗ 模式1测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 测试模式2: livecheck=0（旁路活体检测）
    print("\n[4/4] 模式2: livecheck=0 (旁路活体检测)")
    print("  使用场景: 受控环境（员工考勤、内部系统等）")
    try:
        result2 = sdk.recognize(rgb_image, ir_image, livecheck=0)

        print(f"  识别结果:")
        print(f"    - 成功: {result2.success}")
        print(f"    - 活体检测: 旁路模式 (score={result2.liveness_score:.4f})")
        print(f"    - 活体检测通过: {result2.liveness_passed} (自动通过)")
        print(f"    - 质量评估分数: {result2.quality_score:.4f}")
        print(f"    - 质量评估通过: {result2.quality_passed}")

        if result2.success:
            print(f"    - 特征提取: ✓ ({result2.feature.shape})")
            print(f"    - 处理时间: {result2.total_time_ms:.2f} ms")

            # 计算性能提升
            if 'result1' in locals() and result1.total_time_ms > 0:
                speedup = (result1.total_time_ms - result2.total_time_ms) / result1.total_time_ms * 100
                if speedup > 0:
                    print(f"    - 性能提升: ~{speedup:.1f}% (跳过活体检测)")
        else:
            print(f"    - 失败原因: {result2.error_message}")

    except Exception as e:
        print(f"  ✗ 模式2测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print()
    print("livecheck 参数说明:")
    print("  • livecheck=1 (默认)")
    print("    - 执行RGB+IR双模态活体检测")
    print("    - 防护等级: 高（防照片、视频、面具攻击）")
    print("    - 适用场景: 门禁、支付、身份认证")
    print()
    print("  • livecheck=0 (旁路)")
    print("    - 跳过活体检测，仅质量评估和特征提取")
    print("    - 防护等级: 基础（质量检查）")
    print("    - 适用场景: 员工考勤、内部系统、受控环境")
    print("    - 优势: 性能更快，降低误拒率")
    print()

    return True


def main():
    """主函数"""

    print()
    print("OCFA Face Recognition SDK")
    print("Livecheck Parameter Demo")
    print("Version: 1.0.0")
    print("=" * 70)
    print()

    success = test_livecheck_modes()

    if success:
        print("\n✓ 演示完成！")
        print()
        print("建议:")
        print("  1. 根据实际应用场景选择合适的livecheck模式")
        print("  2. 高安全场景使用 livecheck=1")
        print("  3. 受控环境可使用 livecheck=0 提升用户体验")
        print()
        return 0
    else:
        print("\n✗ 演示失败")
        print()
        print("故障排除:")
        print("  1. 确保模型文件存在")
        print("  2. 确保测试图像存在")
        print("  3. 使用 --dummy 模式测试: python test_basic_pipeline.py --dummy")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
