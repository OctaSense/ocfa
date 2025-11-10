/**
 * @file engine_factory.cpp
 * @brief Inference engine factory implementation
 */

#include "inference_engine.h"
#include "onnx_engine.h"
#include "nnie_engine.h"
#include <algorithm>
#include <cctype>

namespace ocfa {
namespace inference {

std::unique_ptr<InferenceEngine> CreateEngine(const std::string& type, int num_threads) {
    // Convert to lowercase for case-insensitive comparison
    std::string type_lower = type;
    std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

#ifdef USE_ONNX_RUNTIME
    if (type_lower == "onnx" || type_lower == "onnxruntime") {
        return std::make_unique<ONNXEngine>(num_threads);
    }
#endif

#ifdef USE_NNIE
    if (type_lower == "nnie" || type_lower == "hi3516cv610") {
        return std::make_unique<NNIEEngine>();
    }
#endif

    // Return nullptr if engine type not supported
    return nullptr;
}

} // namespace inference
} // namespace ocfa
