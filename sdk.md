
目标运行环境：
- 嵌入式linux平台，ARM或RISC-V，双核心800MHz, 128MB RAM，NPU算力1T ops@INT8；
- RGB + IR双目摄像头，IR主要用于活体检测功能，720p传感器，FOV H：60度，FOV V：45度；
- 不自己训练基础模型，使用InsightFace的arcface模型, 根据指标要求不同使用r34, r50, r100, r152等不同尺寸模型
- 未来可能支持adaface模型，尺寸同arcface

基本要求：
- RGB + IRG活体检测，可以对抗手机/平板/照片攻击；
- 人脸特征提取，人脸比对等功能；
- 具备人脸比对功能，支持1:1和1: N比对模式；

