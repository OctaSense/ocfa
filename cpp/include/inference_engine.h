/**
 * @file inference_engine.h
 * @brief Inference engine interface for face recognition models
 */

#ifndef OCFA_INFERENCE_ENGINE_H
#define OCFA_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <memory>

namespace ocfa {
namespace inference {

/**
 * @brief Tensor data structure
 */
struct Tensor {
    std::vector<float> data;
    std::vector<int64_t> shape;

    int64_t size() const {
        int64_t s = 1;
        for (auto dim : shape) s *= dim;
        return s;
    }
};

/**
 * @brief Inference engine interface (abstract base class)
 */
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    /**
     * @brief Load model from file
     * @param model_path Path to model file
     * @return true on success
     */
    virtual bool LoadModel(const std::string& model_path) = 0;

    /**
     * @brief Run inference with single input
     * @param input Input tensor
     * @param output Output tensor
     * @return true on success
     */
    virtual bool Run(const Tensor& input, Tensor& output) = 0;

    /**
     * @brief Run inference with multiple inputs
     * @param inputs Vector of input tensors
     * @param outputs Vector of output tensors
     * @return true on success
     */
    virtual bool Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) = 0;

    /**
     * @brief Get input tensor shape
     * @param index Input index (default 0)
     * @return Shape vector
     */
    virtual std::vector<int64_t> GetInputShape(int index = 0) const = 0;

    /**
     * @brief Get output tensor shape
     * @param index Output index (default 0)
     * @return Shape vector
     */
    virtual std::vector<int64_t> GetOutputShape(int index = 0) const = 0;

    /**
     * @brief Get number of input tensors
     */
    virtual int GetInputCount() const = 0;

    /**
     * @brief Get number of output tensors
     */
    virtual int GetOutputCount() const = 0;

    /**
     * @brief Check if model is loaded
     */
    virtual bool IsLoaded() const = 0;
};

/**
 * @brief Create inference engine based on type
 * @param type Engine type ("onnx" or "nnie")
 * @param num_threads Number of CPU threads (for ONNX Runtime)
 * @return Unique pointer to engine instance
 */
std::unique_ptr<InferenceEngine> CreateEngine(const std::string& type, int num_threads = 1);

} // namespace inference
} // namespace ocfa

#endif // OCFA_INFERENCE_ENGINE_H
