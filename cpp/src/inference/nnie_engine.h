/**
 * @file nnie_engine.h
 * @brief HiSilicon NNIE inference engine
 */

#ifndef OCFA_NNIE_ENGINE_H
#define OCFA_NNIE_ENGINE_H

#include "inference_engine.h"

#ifdef USE_NNIE
// NNIE SDK headers (Hi3516CV610)
#include "hi_comm_svp.h"
#include "hi_nnie.h"
#include "mpi_nnie.h"
#include "mpi_svp.h"
#endif

namespace ocfa {
namespace inference {

#ifdef USE_NNIE

/**
 * @brief NNIE implementation of inference engine
 */
class NNIEEngine : public InferenceEngine {
public:
    NNIEEngine();
    ~NNIEEngine() override;

    bool LoadModel(const std::string& model_path) override;
    bool Run(const Tensor& input, Tensor& output) override;
    bool Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

    std::vector<int64_t> GetInputShape(int index = 0) const override;
    std::vector<int64_t> GetOutputShape(int index = 0) const override;
    int GetInputCount() const override;
    int GetOutputCount() const override;
    bool IsLoaded() const override { return loaded_; }

private:
    bool AllocateMemory();
    void ReleaseMemory();
    bool CopyInputData(const std::vector<Tensor>& inputs);
    bool CopyOutputData(std::vector<Tensor>& outputs);

    bool loaded_;

    // NNIE model and runtime structures
    SVP_NNIE_MODEL_S model_;
    SVP_MEM_INFO_S model_buf_;

    SVP_NNIE_CFG_S nnie_cfg_;
    SVP_NNIE_HANDLE nnie_handle_;

    std::vector<SVP_SRC_BLOB_S> input_blobs_;
    std::vector<SVP_DST_BLOB_S> output_blobs_;

    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Temporary buffer for INT8 quantization
    std::vector<HI_U8*> input_data_buffers_;
    std::vector<HI_U8*> output_data_buffers_;
};

#endif // USE_NNIE

} // namespace inference
} // namespace ocfa

#endif // OCFA_NNIE_ENGINE_H
