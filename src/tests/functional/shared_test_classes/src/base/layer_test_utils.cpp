// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include <thread>

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/core_config.hpp"
#include "ie_icore.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsUtils {

LayerTestsCommon::LayerTestsCommon() : threshold(1e-2f), abs_threshold(-1.f) {
    core = PluginCache::get().ie(targetDevice);
}

void LayerTestsCommon::Run() {
    bool isCurrentTestDisabled = ov::test::utils::current_test_is_disabled();

    ov::test::utils::PassRate::Statuses status = isCurrentTestDisabled ?
         ov::test::utils::PassRate::Statuses::SKIPPED :
         ov::test::utils::PassRate::Statuses::CRASHED;

    auto &s = ov::test::utils::OpSummary::getInstance();
    s.setDeviceName(targetDevice);
    s.updateOPsStats(function, status);

    if (isCurrentTestDisabled)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

    if (functionRefs == nullptr) {
        functionRefs = function->clone();
        functionRefs->set_friendly_name("refFunction");
    }

    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        try {
            LoadNetwork();
            GenerateInputs();
            Infer();
            Validate();
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::PASSED);
        }
        catch (const std::runtime_error &re) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
            GTEST_FATAL_FAILURE_(re.what());
        } catch (const std::exception &ex) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
            GTEST_FATAL_FAILURE_(ex.what());
        } catch (...) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
            GTEST_FATAL_FAILURE_("Unknown failure occurred.");
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        IE_THROW() << "Crash happens";
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::HANGED);
        IE_THROW() << "Crash happens";
    }
}

InferenceEngine::Blob::Ptr LayerTestsCommon::GenerateInput(const InferenceEngine::InputInfo& info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

void LayerTestsCommon::Compare(
    const std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
    const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs,
    float threshold,
    float abs_threshold) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto &expected = expectedOutputs[outputIndex];
        const auto &actual = actualOutputs[outputIndex];
        Compare(expected, actual, threshold, abs_threshold);
    }
}

template <typename T_IE>
inline void callCompare(const std::pair<ov::element::Type, std::vector<std::uint8_t>>& expected,
                        const T_IE* actualBuffer,
                        size_t size,
                        float threshold,
                        float abs_threshold) {
    auto expectedBuffer = expected.second.data();
    switch (expected.first) {
    case ov::element::Type_t::boolean:
    case ov::element::Type_t::u8:
        LayerTestsCommon::Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t*>(expectedBuffer),
                                                 actualBuffer,
                                                 size,
                                                 threshold,
                                                 abs_threshold);
        break;
    case ov::element::Type_t::i8:
        LayerTestsCommon::Compare<T_IE, int8_t>(reinterpret_cast<const int8_t*>(expectedBuffer),
                                                actualBuffer,
                                                size,
                                                threshold,
                                                abs_threshold);
        break;
    case ov::element::Type_t::u16:
        LayerTestsCommon::Compare<T_IE, uint16_t>(reinterpret_cast<const uint16_t*>(expectedBuffer),
                                                  actualBuffer,
                                                  size,
                                                  threshold,
                                                  abs_threshold);
        break;
    case ov::element::Type_t::i16:
        LayerTestsCommon::Compare<T_IE, int16_t>(reinterpret_cast<const int16_t*>(expectedBuffer),
                                                 actualBuffer,
                                                 size,
                                                 threshold,
                                                 abs_threshold);
        break;
    case ov::element::Type_t::u32:
        LayerTestsCommon::Compare<T_IE, uint32_t>(reinterpret_cast<const uint32_t*>(expectedBuffer),
                                                  actualBuffer,
                                                  size,
                                                  threshold,
                                                  abs_threshold);
        break;
    case ov::element::Type_t::i32:
        LayerTestsCommon::Compare<T_IE, int32_t>(reinterpret_cast<const int32_t*>(expectedBuffer),
                                                 actualBuffer,
                                                 size,
                                                 threshold,
                                                 abs_threshold);
        break;
    case ov::element::Type_t::u64:
        LayerTestsCommon::Compare<T_IE, uint64_t>(reinterpret_cast<const uint64_t*>(expectedBuffer),
                                                  actualBuffer,
                                                  size,
                                                  threshold,
                                                  abs_threshold);
        break;
    case ov::element::Type_t::i64:
        LayerTestsCommon::Compare<T_IE, int64_t>(reinterpret_cast<const int64_t*>(expectedBuffer),
                                                 actualBuffer,
                                                 size,
                                                 threshold,
                                                 abs_threshold);
        break;
    case ov::element::Type_t::bf16:
        LayerTestsCommon::Compare<T_IE, ov::bfloat16>(reinterpret_cast<const ov::bfloat16*>(expectedBuffer),
                                                      actualBuffer,
                                                      size,
                                                      threshold,
                                                      abs_threshold);
        break;
    case ov::element::Type_t::f16:
        LayerTestsCommon::Compare<T_IE, ov::float16>(reinterpret_cast<const ov::float16*>(expectedBuffer),
                                                     actualBuffer,
                                                     size,
                                                     threshold,
                                                     abs_threshold);
        break;
    case ov::element::Type_t::f32:
        LayerTestsCommon::Compare<T_IE, float>(reinterpret_cast<const float*>(expectedBuffer),
                                               actualBuffer,
                                               size,
                                               threshold,
                                               abs_threshold);
        break;
    case ov::element::Type_t::f64:
        LayerTestsCommon::Compare<T_IE, double>(reinterpret_cast<const double*>(expectedBuffer),
                                                actualBuffer,
                                                size,
                                                threshold,
                                                abs_threshold);
        break;
    case ov::element::Type_t::i4: {
        auto expectedOut =
            ngraph::helpers::convertOutputPrecision(expected.second, expected.first, ov::element::Type_t::i8, size);
        LayerTestsCommon::Compare<T_IE, int8_t>(reinterpret_cast<const int8_t*>(expectedOut.data()),
                                                actualBuffer,
                                                size,
                                                threshold,
                                                abs_threshold);
        break;
    }
    case ov::element::Type_t::u4: {
        auto expectedOut =
            ngraph::helpers::convertOutputPrecision(expected.second, expected.first, ov::element::Type_t::u8, size);
        LayerTestsCommon::Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t*>(expectedOut.data()),
                                                 actualBuffer,
                                                 size,
                                                 threshold,
                                                 abs_threshold);
        break;
    }
    case ov::element::Type_t::dynamic:
    case ov::element::Type_t::undefined:
        LayerTestsCommon::Compare<T_IE, T_IE>(reinterpret_cast<const T_IE*>(expectedBuffer),
                                              actualBuffer,
                                              size,
                                              threshold,
                                              abs_threshold);
        break;
    default:
        FAIL() << "Comparator for " << expected.first << " precision isn't supported";
    }
    return;
}

void LayerTestsCommon::Compare(const std::pair<ov::element::Type, std::vector<std::uint8_t>>& expected,
                               const InferenceEngine::Blob::Ptr& actual,
                               float threshold,
                               float abs_threshold) {
    const auto &precision = actual->getTensorDesc().getPrecision();
    auto k =  static_cast<float>(expected.first.size()) / precision.size();
    // W/A for int4, uint4
    if (expected.first == ov::element::Type_t::u4 || expected.first == ov::element::Type_t::i4) {
        k /= 2;
    } else if (expected.first == ov::element::Type_t::undefined || expected.first == ov::element::Type_t::dynamic) {
        k = 1;
    }
    ASSERT_EQ(expected.second.size(), actual->byteSize() * k);

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::U8:
            callCompare<uint8_t>(expected, reinterpret_cast<const uint8_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I8:
            callCompare<int8_t>(expected, reinterpret_cast<const int8_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::U16:
            callCompare<uint16_t>(expected, reinterpret_cast<const uint16_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I16:
            callCompare<int16_t>(expected, reinterpret_cast<const int16_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::U32:
            callCompare<uint32_t>(expected, reinterpret_cast<const uint32_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I32:
            callCompare<int32_t>(expected, reinterpret_cast<const int32_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::U64:
            callCompare<uint64_t>(expected, reinterpret_cast<const uint64_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I64:
            callCompare<int64_t>(expected, reinterpret_cast<const int64_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::BF16:
            callCompare<ov::bfloat16>(expected,
                                      reinterpret_cast<const ov::bfloat16*>(actualBuffer),
                                      size,
                                      threshold,
                                      abs_threshold);
            break;
        case InferenceEngine::Precision::FP16:
            callCompare<ov::float16>(expected,
                                     reinterpret_cast<const ov::float16*>(actualBuffer),
                                     size,
                                     threshold,
                                     abs_threshold);
            break;
        case InferenceEngine::Precision::FP32:
            callCompare<float>(expected, reinterpret_cast<const float *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::FP64:
            callCompare<double>(expected, reinterpret_cast<const double *>(actualBuffer), size, threshold, abs_threshold);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void LayerTestsCommon::Compare(const std::pair<ov::element::Type, std::vector<std::uint8_t>>& expected,
                               const InferenceEngine::Blob::Ptr& actual) {
    Compare(expected, actual, threshold);
}

void LayerTestsCommon::Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual) {
    auto get_raw_buffer = [](const InferenceEngine::Blob::Ptr &blob) {
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        return lockedMemory.as<const std::uint8_t *>();
    };
    const auto expectedBuffer = get_raw_buffer(expected);
    const auto actualBuffer = get_raw_buffer(actual);

    const auto &precision = actual->getTensorDesc().getPrecision();
    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            Compare(reinterpret_cast<const float *>(expectedBuffer), reinterpret_cast<const float *>(actualBuffer),
                    size, threshold);
            break;
        case InferenceEngine::Precision::I32:
            Compare(reinterpret_cast<const std::int32_t *>(expectedBuffer),
                    reinterpret_cast<const std::int32_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::I16:
            Compare(reinterpret_cast<const std::int16_t *>(expectedBuffer),
                    reinterpret_cast<const std::int16_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::U8:
            Compare(reinterpret_cast<const std::uint8_t *>(expectedBuffer),
                    reinterpret_cast<const std::uint8_t *>(actualBuffer), size, 0);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void LayerTestsCommon::Compare(const InferenceEngine::TensorDesc &actualDesc, const InferenceEngine::TensorDesc &expectedDesc) {
    auto expectedDims = actualDesc.getDims();
    auto actualDims = expectedDesc.getDims();
    ASSERT_EQ(actualDims.size(), expectedDims.size());
    for (size_t j = 0; j < actualDims.size(); ++j) {
        ASSERT_EQ(actualDims.at(j), expectedDims.at(j));
    }
    ASSERT_EQ(actualDesc.getLayout(), expectedDesc.getLayout());
    ASSERT_EQ(actualDesc.getPrecision(), expectedDesc.getPrecision());
}

void LayerTestsCommon::ConfigureNetwork() {
    for (const auto &in : cnnNetwork.getInputsInfo()) {
        if (inLayout != InferenceEngine::Layout::ANY) {
            in.second->setLayout(inLayout);
        }
        if (inPrc != InferenceEngine::Precision::UNSPECIFIED) {
            in.second->setPrecision(inPrc);
        }
    }

    for (const auto &out : cnnNetwork.getOutputsInfo()) {
        if (outLayout != InferenceEngine::Layout::ANY) {
            out.second->setLayout(outLayout);
        }
        if (outPrc != InferenceEngine::Precision::UNSPECIFIED) {
            out.second->setPrecision(outPrc);
        }
    }
}

void LayerTestsCommon::LoadNetwork() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    CoreConfiguration(this);
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
}

void LayerTestsCommon::ExpectLoadNetworkToThrow(const std::string& msg) {
    std::string what;
    try {
        LoadNetwork();
    } catch (const std::exception& e) {
        what.assign(e.what());
    }
    EXPECT_STR_CONTAINS(what.c_str(), msg.c_str());
}

void LayerTestsCommon::GenerateInputs() {
    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
        inputs.push_back(blob);
    }
}

void LayerTestsCommon::ConfigureInferRequest() {
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;
        auto blob = inputs[i];
        inferRequest.SetBlob(info->name(), blob);
    }
}

void LayerTestsCommon::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();

    ConfigureInferRequest();

    inferRequest.Infer();
}

std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> LayerTestsCommon::CalculateRefs() {
    return {};
}

std::vector<InferenceEngine::Blob::Ptr> LayerTestsCommon::GetOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &name = output.first;
        outputs.push_back(inferRequest.GetBlob(name));
    }
    return outputs;
}

void LayerTestsCommon::Compare(
    const std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
    const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    Compare(expectedOutputs, actualOutputs, threshold);
}

void LayerTestsCommon::Validate() {}

std::shared_ptr<ov::Model> LayerTestsCommon::GetFunction() {
    return function;
}

std::map<std::string, std::string> &LayerTestsCommon::GetConfiguration() {
    return configuration;
}

}  // namespace LayerTestsUtils
