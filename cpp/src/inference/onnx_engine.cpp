/**
 * @file onnx_engine.cpp
 * @brief ONNX Runtime inference engine implementation
 */

#include "onnx_engine.h"
#include <cstring>
#include <stdexcept>

namespace ocfa {
namespace inference {

#ifdef USE_ONNX_RUNTIME

ONNXEngine::ONNXEngine(int num_threads)
    : loaded_(false), num_threads_(num_threads) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OCFAFaceSDK");
    session_options_ = std::make_unique<Ort::SessionOptions>();

    session_options_->SetIntraOpNumThreads(num_threads_);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

ONNXEngine::~ONNXEngine() {
    session_.reset();
    session_options_.reset();
    env_.reset();
}

bool ONNXEngine::LoadModel(const std::string& model_path) {
    try {
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        // Get input information
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        input_names_.clear();
        input_shapes_.clear();

        for (size_t i = 0; i < num_inputs; ++i) {
            // Get input name
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(name_ptr.get()));

            // Get input shape
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            input_shapes_.push_back(shape);
        }

        // Get output information
        size_t num_outputs = session_->GetOutputCount();
        output_names_.clear();
        output_shapes_.clear();

        for (size_t i = 0; i < num_outputs; ++i) {
            // Get output name
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(name_ptr.get()));

            // Get output shape
            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            output_shapes_.push_back(shape);
        }

        loaded_ = true;
        return true;

    } catch (const Ort::Exception& e) {
        loaded_ = false;
        return false;
    }
}

bool ONNXEngine::Run(const Tensor& input, Tensor& output) {
    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> outputs(1);

    bool success = Run(inputs, outputs);
    if (success) {
        output = outputs[0];
    }
    return success;
}

bool ONNXEngine::Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (!loaded_) {
        return false;
    }

    if (inputs.size() != input_names_.size()) {
        return false;
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names_char;

        for (size_t i = 0; i < inputs.size(); ++i) {
            input_names_char.push_back(input_names_[i].c_str());

            // Create input tensor
            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(inputs[i].data.data()),
                inputs[i].data.size(),
                inputs[i].shape.data(),
                inputs[i].shape.size()
            );
            input_tensors.push_back(std::move(tensor));
        }

        // Prepare output names
        std::vector<const char*> output_names_char;
        for (const auto& name : output_names_) {
            output_names_char.push_back(name.c_str());
        }

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_char.data(),
            input_tensors.data(),
            inputs.size(),
            output_names_char.data(),
            output_names_.size()
        );

        // Extract outputs
        outputs.resize(output_tensors.size());
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto* float_data = output_tensors[i].GetTensorMutableData<float>();
            auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();

            int64_t size = 1;
            for (auto dim : shape) size *= dim;

            outputs[i].shape = shape;
            outputs[i].data.assign(float_data, float_data + size);
        }

        return true;

    } catch (const Ort::Exception& e) {
        return false;
    }
}

std::vector<int64_t> ONNXEngine::GetInputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(input_shapes_.size())) {
        return input_shapes_[index];
    }
    return {};
}

std::vector<int64_t> ONNXEngine::GetOutputShape(int index) const {
    if (index >= 0 && index < static_cast<int>(output_shapes_.size())) {
        return output_shapes_[index];
    }
    return {};
}

int ONNXEngine::GetInputCount() const {
    return static_cast<int>(input_names_.size());
}

int ONNXEngine::GetOutputCount() const {
    return static_cast<int>(output_names_.size());
}

#endif // USE_ONNX_RUNTIME

} // namespace inference
} // namespace ocfa
