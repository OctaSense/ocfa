# OCFA Face SDK - C++ 构建指南

**版本**: v1.0.0
**日期**: 2025-01-11
**状态**: ✅ 构建成功

---

## 快速开始

### 基础构建 (macOS/Linux)

```bash
cd cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 查看构建输出

```bash
# 库文件
ls -lh libocfa_face.*

# 可执行文件
ls -lh examples/
```

---

## 构建配置选项

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_EXAMPLES` | ON | 编译示例程序 |
| `BUILD_TESTS` | ON | 编译单元测试 |
| `USE_ONNXRUNTIME` | ON | 使用 ONNX Runtime |
| `USE_NNIE` | OFF | 使用 Hi3516CV610 NNIE |
| `USE_OPENCV` | OFF | 使用 OpenCV |
| `CMAKE_BUILD_TYPE` | Release | Debug / Release |

### 常用构建命令

#### 1. 仅构建库，不构建示例

```bash
cmake .. -DBUILD_EXAMPLES=OFF
make -j$(nproc)
```

#### 2. 调试模式构建

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

#### 3. 使用 NNIE (Hi3516CV610)

```bash
cmake .. \
  -DUSE_NNIE=ON \
  -DNNIE_SDK_PATH=/path/to/nnie/sdk
make
```

#### 4. 完整构建（包含测试）

```bash
cmake .. \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=ON \
  -DUSE_ONNXRUNTIME=ON
make
```

---

## 已知问题和解决方案

### 问题 1: ONNX Runtime 不被找到

**错误消息**:
```
CMake Error: ONNX Runtime not found
```

**原因**: C++ 开发文件（头文件和库）未找到

**解决方案**:

方案 A: 使用 Homebrew (macOS)
```bash
brew install onnxruntime
cmake .. -DUSE_ONNXRUNTIME=ON
```

方案 B: 手动指定路径
```bash
cmake .. \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnx/include \
  -DONNXRUNTIME_LIB=/path/to/onnx/lib/libonnxruntime.so
```

方案 C: 禁用 ONNX Runtime (仅构建基础库)
```bash
cmake .. -DUSE_ONNXRUNTIME=OFF
```

### 问题 2: ARM NEON 编译错误

**错误消息**:
```
error: unsupported option '-mfpu=' for target 'arm64-apple-darwin'
```

**原因**: macOS ARM64 不支持 Linux ARM 编译标志

**解决方案**: 已在 CMakeLists.txt 中修复
- Linux ARM: 使用 `-mfpu=neon -mfloat-abi=hard`
- macOS ARM64: 隐式支持，无需特殊标志

### 问题 3: 缺少 cmath 头文件

**错误消息**:
```
error: use of undeclared identifier 'sqrt'
```

**原因**: 某些源文件未包含 `<cmath>`

**解决方案**: 已修复 demo_basic.cpp 和其他相关文件

---

## 编译输出文件

### 生成的库文件

```
cpp/build/
├── libocfa_face.a          (静态库 ~49 KB)
└── libocfa_face.dylib      (动态库 ~56 KB)
```

### 生成的可执行文件

```
cpp/build/examples/
├── demo_basic              (基础功能演示)
├── demo_recognition        (人脸识别演示)
└── benchmark_neon          (NEON 性能基准)
```

---

## 运行示例

### 运行基础演示

```bash
./cpp/build/examples/demo_basic
```

**预期输出**:
```
OCFA Face SDK - Basic Example
Version: 1.0.0

Initializing SDK...
Failed to initialize SDK: Unsupported device
```

注: 失败是预期的，因为演示环境中没有推理引擎

### 运行人脸识别演示

```bash
./cpp/build/examples/demo_recognition
```

### 运行性能基准测试

```bash
./cpp/build/examples/benchmark_neon
```

---

## 库的使用

### 链接静态库

```bash
g++ -std=c++17 myapp.cpp -o myapp \
  -I/path/to/ocfa/cpp/include \
  -L/path/to/ocfa/cpp/build \
  -locfa_face -lm -lpthread
```

### 链接动态库

```bash
g++ -std=c++17 myapp.cpp -o myapp \
  -I/path/to/ocfa/cpp/include \
  -L/path/to/ocfa/cpp/build \
  -locfa_face -lm -lpthread

# 运行时指定库路径
export LD_LIBRARY_PATH=/path/to/ocfa/cpp/build:$LD_LIBRARY_PATH
./myapp
```

### CMakeLists.txt 集成

```cmake
# 包含 OCFA SDK
add_subdirectory(path/to/ocfa/cpp)

# 你的应用
add_executable(myapp myapp.cpp)
target_link_libraries(myapp ocfa_face_sdk)
```

---

## 跨平台编译

### Linux ARM (Hi3516CV610)

```bash
mkdir build_arm
cd build_arm

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/arm-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_NNIE=ON \
  -DNNIE_SDK_PATH=/path/to/nnie/sdk

make -j4
```

### x86 Linux

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### macOS Intel

```bash
cmake .. \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_BUILD_TYPE=Release
make
```

### macOS ARM64 (Apple Silicon)

```bash
cmake .. \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_BUILD_TYPE=Release
make
```

---

## 性能优化

### 启用优化编译

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native"
make
```

### ARM NEON 优化验证

运行基准测试确认 NEON 优化已启用：

```bash
./cpp/build/examples/benchmark_neon
```

预期性能改进: 3-4x (vs 标准实现)

---

## 故障排除

### 清除构建缓存

```bash
rm -rf cpp/build
mkdir -p cpp/build
cd cpp/build
cmake ..
make
```

### 查看编译详情

```bash
# 显示完整的编译命令
make VERBOSE=1

# 或者使用 CMake 的详细模式
cmake --build . --verbose
```

### 检查 CMake 配置

```bash
cd cpp/build
cmake -LAH ..
```

### 生成编译数据库 (IDE 使用)

```bash
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

生成的 `compile_commands.json` 可被 IDE (CLion, VSCode) 使用

---

## 文档和参考

### 构建系统文档

- `cpp/CMakeLists.txt` - CMake 构建配置
- `cpp/README.md` - C++ SDK 概览

### 源代码

- `cpp/include/ocfa_face_sdk.h` - 公开 API
- `cpp/src/core/` - 核心实现
- `cpp/src/utils/` - 工具函数
- `cpp/examples/` - 使用示例

### 编译选项参考

```bash
cmake .. -h     # 显示所有选项
cmake .. -LH    # 显示 CMake 变量
```

---

## 后续工作

### 短期

- [ ] 集成 ONNX Runtime C++ SDK
- [ ] 完成推理引擎封装
- [ ] 添加单元测试

### 中期

- [ ] Hi3516CV610 交叉编译配置
- [ ] NNIE 推理引擎集成
- [ ] 性能优化和基准测试

### 长期

- [ ] 完整的文档和 API 参考
- [ ] Python 绑定生成
- [ ] Package 管理集成 (Conan, vcpkg)

---

## 获取帮助

### CMake 常见问题

```bash
# 查看所有变量
cmake .. -LAH

# 调试 CMake
cmake .. --debug-output
```

### 编译错误诊断

1. 检查编译器版本: `c++ --version`
2. 检查依赖: `brew list` (macOS)
3. 查看完整的错误信息: `make VERBOSE=1`

### 相关资源

- CMake 官方文档: https://cmake.org/cmake/help/latest/
- ONNX Runtime: https://github.com/microsoft/onnxruntime

---

**最后更新**: 2025-01-11
**版本**: v1.0.0
**状态**: ✅ 构建成功并已验证
