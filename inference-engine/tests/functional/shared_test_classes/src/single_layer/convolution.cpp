// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "shared_test_classes/single_layer/convolution.hpp"
#include "functional_test_utils/partial_shape_utils.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
    convSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<std::pair<size_t, size_t>> inputShape;
    std::vector<InferenceEngine::SizeVector> targetShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShapes, targetDevice) =
        obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ConvolutionLayerTest::SetUp() {
    convSpecificParams convParams;
    std::vector<std::pair<size_t, size_t>> inputShape;
    std::vector<InferenceEngine::SizeVector> targetShapes;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShapes, targetDevice) =
        this->GetParam();
    for (auto&& targetShape : targetShapes) {
        targetStaticShapes.emplace_back(targetShape);
    }
    inputDynamicShape = FuncTestUtils::PartialShapeUtils::vec2partialshape(inputShape, targetStaticShapes[0]);
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    setTargetStaticShape(targetStaticShapes[0]);
    function = makeConvolution("convolution");
    functionRefs = makeConvolution("convolutionRefs");
}

std::shared_ptr<ngraph::Function> ConvolutionLayerTest::makeConvolution(const std::string& name) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> filter_weights;
    if (targetDevice == CommonTestUtils::DEVICE_GNA) {
        auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
        filter_weights = CommonTestUtils::generate_float_numbers(convOutChannels * targetStaticShape[1] * filter_size,
                                                                 -0.5f, 0.5f);
    }
    auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(
            ngraph::builder::makeConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, false, filter_weights));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(conv)};
    return std::make_shared<ngraph::Function>(results, params, name);
}

void ConvolutionLayerTest::Run() {
    auto crashHandler = [](int errCode) {
        auto &s = LayerTestsUtils::Summary::getInstance();
        s.saveReport();
        std::cout << "Unexpected application crash!" << std::endl;
        std::abort();
    };
    signal(SIGSEGV, crashHandler);

    auto &s = LayerTestsUtils::Summary::getInstance();
    s.setDeviceName(targetDevice);

    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        s.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::SKIPPED);
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
    } else {
        s.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::CRASHED);
    }

    try {
        LoadNetwork();
        for (auto&& tss : targetStaticShapes) {
            setTargetStaticShape(tss);
            GenerateInputs();
            Infer();
            Validate();
            s.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::PASSED);
        }
    }
    catch (const std::runtime_error &re) {
        s.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_(re.what());
    } catch (const std::exception &ex) {
        s.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_(ex.what());
    } catch (...) {
        s.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_("Unknown failure occurred.");
    }
}

}  // namespace LayerTestsDefinitions
