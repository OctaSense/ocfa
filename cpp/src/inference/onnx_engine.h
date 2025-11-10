/**
 * @file onnx_engine.h
 * @brief ONNX Runtime inference engine
 */

#ifndef OCFA_ONNX_ENGINE_H
#define OCFA_ONNX_ENGINE_H

#include "inference_engine.h"

#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace ocfa {
namespace inference {

#ifdef USE_ONNX_RUNTIME

/**
 * @brief ONNX Runtime implementation of inference engine
 */
class ONNXEngine : public InferenceEngine {
public:
    ONNXEngine(int num_threads = 1);
    ~ONNXEngine() override;

    bool LoadModel(const std::string& model_path) override;
    bool Run(const Tensor& input, Tensor& output) override;
    bool Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

    std::vector<int64_t> GetInputShape(int index = 0) const override;
    std::vector<int64_t> GetOutputShape(int index = 0) const override;
    int GetInputCount() const override;
    int GetOutputCount() const override;
    bool IsLoaded() const override { return loaded_; }

private:
    bool loaded_;
    int num_threads_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
};

#endif // USE_ONNX_RUNTIME

} // namespace inference
} // namespace ocfa

#endif // OCFA_ONNX_ENGINE_H
