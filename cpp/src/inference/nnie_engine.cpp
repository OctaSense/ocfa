/**
 * @file nnie_engine.cpp
 * @brief HiSilicon NNIE inference engine implementation
 */

#include "nnie_engine.h"
#include <cstring>
#include <cmath>
#include <fstream>

namespace ocfa {
namespace inference {

#ifdef USE_NNIE

NNIEEngine::NNIEEngine() : loaded_(false), nnie_handle_(0) {
    std::memset(&model_, 0, sizeof(model_));
    std::memset(&model_buf_, 0, sizeof(model_buf_));
    std::memset(&nnie_cfg_, 0, sizeof(nnie_cfg_));
}

NNIEEngine::~NNIEEngine() {
    ReleaseMemory();
}

bool NNIEEngine::LoadModel(const std::string& model_path) {
    // Read model file (.wk format for NNIE)
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate model buffer
    HI_S32 ret = HI_MPI_SYS_MmzAlloc(&model_buf_.u64PhyAddr,
                                      (HI_VOID**)&model_buf_.u64VirAddr,
                                      NULL, HI_NULL, size);
    if (ret != HI_SUCCESS) {
        return false;
    }
    model_buf_.u32Size = size;

    // Read model data
    if (!file.read(reinterpret_cast<char*>(model_buf_.u64VirAddr), size)) {
        HI_MPI_SYS_MmzFree(model_buf_.u64PhyAddr, (HI_VOID*)model_buf_.u64VirAddr);
        return false;
    }
    file.close();

    // Load NNIE model
    ret = HI_MPI_SVP_NNIE_LoadModel(&model_buf_, &model_);
    if (ret != HI_SUCCESS) {
        HI_MPI_SYS_MmzFree(model_buf_.u64PhyAddr, (HI_VOID*)model_buf_.u64VirAddr);
        return false;
    }

    // Get input/output information from model
    // Assume segment 0 for simplicity (single-stage model)
    HI_U32 seg_idx = 0;

    // Input shapes
    input_shapes_.clear();
    for (HI_U32 i = 0; i < model_.astSeg[seg_idx].u16SrcNum; ++i) {
        SVP_BLOB_S* blob = &model_.astSeg[seg_idx].astSrc[i];
        std::vector<int64_t> shape = {
            blob->unShape.stWhc.u32Chn,
            blob->unShape.stWhc.u32Height,
            blob->unShape.stWhc.u32Width
        };
        input_shapes_.push_back(shape);
    }

    // Output shapes
    output_shapes_.clear();
    for (HI_U32 i = 0; i < model_.astSeg[seg_idx].u16DstNum; ++i) {
        SVP_BLOB_S* blob = &model_.astSeg[seg_idx].astDst[i];
        std::vector<int64_t> shape = {
            blob->unShape.stWhc.u32Chn,
            blob->unShape.stWhc.u32Height,
            blob->unShape.stWhc.u32Width
        };
        output_shapes_.push_back(shape);
    }

    // Allocate memory for blobs
    if (!AllocateMemory()) {
        HI_MPI_SVP_NNIE_UnloadModel(&model_);
        HI_MPI_SYS_MmzFree(model_buf_.u64PhyAddr, (HI_VOID*)model_buf_.u64VirAddr);
        return false;
    }

    loaded_ = true;
    return true;
}

bool NNIEEngine::AllocateMemory() {
    HI_U32 seg_idx = 0;

    // Allocate input blobs
    input_blobs_.resize(model_.astSeg[seg_idx].u16SrcNum);
    input_data_buffers_.resize(input_blobs_.size());

    for (size_t i = 0; i < input_blobs_.size(); ++i) {
        SVP_BLOB_S* src = &model_.astSeg[seg_idx].astSrc[i];
        HI_U32 size = src->u32Stride * src->unShape.stWhc.u32Height;

        HI_S32 ret = HI_MPI_SYS_MmzAlloc(
            &input_blobs_[i].u64PhyAddr,
            (HI_VOID**)&input_blobs_[i].u64VirAddr,
            NULL, HI_NULL, size);

        if (ret != HI_SUCCESS) {
            return false;
        }

        input_blobs_[i].u32Num = 1;
        input_blobs_[i].unShape = src->unShape;
        input_blobs_[i].u32Stride = src->u32Stride;
        input_blobs_[i].enType = src->enType;

        input_data_buffers_[i] = reinterpret_cast<HI_U8*>(input_blobs_[i].u64VirAddr);
    }

    // Allocate output blobs
    output_blobs_.resize(model_.astSeg[seg_idx].u16DstNum);
    output_data_buffers_.resize(output_blobs_.size());

    for (size_t i = 0; i < output_blobs_.size(); ++i) {
        SVP_BLOB_S* dst = &model_.astSeg[seg_idx].astDst[i];
        HI_U32 size = dst->u32Stride * dst->unShape.stWhc.u32Height;

        HI_S32 ret = HI_MPI_SYS_MmzAlloc(
            &output_blobs_[i].u64PhyAddr,
            (HI_VOID**)&output_blobs_[i].u64VirAddr,
            NULL, HI_NULL, size);

        if (ret != HI_SUCCESS) {
            return false;
        }

        output_blobs_[i].u32Num = 1;
        output_blobs_[i].unShape = dst->unShape;
        output_blobs_[i].u32Stride = dst->u32Stride;
        output_blobs_[i].enType = dst->enType;

        output_data_buffers_[i] = reinterpret_cast<HI_U8*>(output_blobs_[i].u64VirAddr);
    }

    return true;
}

void NNIEEngine::ReleaseMemory() {
    if (!loaded_) return;

    // Free input blobs
    for (auto& blob : input_blobs_) {
        if (blob.u64PhyAddr != 0) {
            HI_MPI_SYS_MmzFree(blob.u64PhyAddr, (HI_VOID*)blob.u64VirAddr);
        }
    }
    input_blobs_.clear();
    input_data_buffers_.clear();

    // Free output blobs
    for (auto& blob : output_blobs_) {
        if (blob.u64PhyAddr != 0) {
            HI_MPI_SYS_MmzFree(blob.u64PhyAddr, (HI_VOID*)blob.u64VirAddr);
        }
    }
    output_blobs_.clear();
    output_data_buffers_.clear();

    // Unload model
    if (model_buf_.u64PhyAddr != 0) {
        HI_MPI_SVP_NNIE_UnloadModel(&model_);
        HI_MPI_SYS_MmzFree(model_buf_.u64PhyAddr, (HI_VOID*)model_buf_.u64VirAddr);
    }

    loaded_ = false;
}

bool NNIEEngine::CopyInputData(const std::vector<Tensor>& inputs) {
    if (inputs.size() != input_blobs_.size()) {
        return false;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor& tensor = inputs[i];
        SVP_BLOB_S* src = &model_.astSeg[0].astSrc[i];

        // NNIE expects INT8 quantized input
        // Assuming input tensors are float32, we need to quantize
        // Quantization: int8_value = (float_value - zero_point) * scale
        // For simplicity, assume scale=1.0, zero_point=0
        // Real implementation should load quantization params from model

        HI_U32 chn = src->unShape.stWhc.u32Chn;
        HI_U32 height = src->unShape.stWhc.u32Height;
        HI_U32 width = src->unShape.stWhc.u32Width;
        HI_U32 stride = src->u32Stride;

        // Convert float to INT8 (simple conversion for demonstration)
        for (HI_U32 c = 0; c < chn; ++c) {
            for (HI_U32 h = 0; h < height; ++h) {
                for (HI_U32 w = 0; w < width; ++w) {
                    HI_U32 idx = c * height * width + h * width + w;
                    float val = tensor.data[idx];

                    // Clamp to [-128, 127] and convert to INT8
                    val = std::max(-128.0f, std::min(127.0f, val));
                    HI_S8 int8_val = static_cast<HI_S8>(std::round(val));

                    // Write to blob with stride
                    input_data_buffers_[i][c * stride * height + h * stride + w] =
                        static_cast<HI_U8>(int8_val);
                }
            }
        }
    }

    return true;
}

bool NNIEEngine::CopyOutputData(std::vector<Tensor>& outputs) {
    outputs.resize(output_blobs_.size());

    for (size_t i = 0; i < output_blobs_.size(); ++i) {
        SVP_BLOB_S* dst = &model_.astSeg[0].astDst[i];

        HI_U32 chn = dst->unShape.stWhc.u32Chn;
        HI_U32 height = dst->unShape.stWhc.u32Height;
        HI_U32 width = dst->unShape.stWhc.u32Width;
        HI_U32 stride = dst->u32Stride;

        outputs[i].shape = output_shapes_[i];
        outputs[i].data.resize(chn * height * width);

        // Dequantize INT8 to float
        // Real implementation should use actual quantization params
        for (HI_U32 c = 0; c < chn; ++c) {
            for (HI_U32 h = 0; h < height; ++h) {
                for (HI_U32 w = 0; w < width; ++w) {
                    HI_U32 idx = c * height * width + h * width + w;
                    HI_S8 int8_val = static_cast<HI_S8>(
                        output_data_buffers_[i][c * stride * height + h * stride + w]);

                    // Dequantize: float_value = int8_value / scale + zero_point
                    outputs[i].data[idx] = static_cast<float>(int8_val);
                }
            }
        }
    }

    return true;
}

bool NNIEEngine::Run(const Tensor& input, Tensor& output) {
    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> outputs(1);

    bool success = Run(inputs, outputs);
    if (success) {
        output = outputs[0];
    }
    return success;
}

bool NNIEEngine::Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (!loaded_) {
        return false;
    }

    // Copy input data
    if (!CopyInputData(inputs)) {
        return false;
    }

    // Configure NNIE runtime
    nnie_cfg_.astSeg[0].u16SrcNum = input_blobs_.size();
    nnie_cfg_.astSeg[0].u16DstNum = output_blobs_.size();

    for (size_t i = 0; i < input_blobs_.size(); ++i) {
        nnie_cfg_.astSeg[0].astSrc[i] = input_blobs_[i];
    }

    for (size_t i = 0; i < output_blobs_.size(); ++i) {
        nnie_cfg_.astSeg[0].astDst[i] = output_blobs_[i];
    }

    // Run inference (segment 0)
    HI_BOOL bInstant = HI_TRUE;
    HI_S32 ret = HI_MPI_SVP_NNIE_Forward(&nnie_handle_, &input_blobs_[0],
                                          &model_, &nnie_cfg_, bInstant);
    if (ret != HI_SUCCESS) {
        return false;
    }

    // Wait for completion
    HI_BOOL bFinish = HI_FALSE;
    HI_BOOL bBlock = HI_TRUE;
    ret = HI_MPI_SVP_NNIE_Query(nnie_handle_, &model_, bBlock, &bFinish);
    if (ret != HI_SUCCESS || !bFinish) {
        return false;
    }

    // Copy output data
    if (!CopyOutputData(outputs)) {
        return false;
    }

    return true;
}

std::vector<int64_t> NNIEEngine::GetInputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(input_shapes_.size())) {
        return input_shapes_[index];
    }
    return {};
}

std::vector<int64_t> NNIEEngine::GetOutputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(output_shapes_.size())) {
        return output_shapes_[index];
    }
    return {};
}

int NNIEEngine::GetInputCount() const {
    return static_cast<int>(input_shapes_.size());
}

int NNIEEngine::GetOutputCount() const {
    return static_cast<int>(output_shapes_.size());
}

#endif // USE_NNIE

} // namespace inference
} // namespace ocfa
