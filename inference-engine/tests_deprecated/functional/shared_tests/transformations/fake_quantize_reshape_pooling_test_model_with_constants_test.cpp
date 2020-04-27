// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void FakeQuantizeReshapePoolingTestModelWithConstants::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "inputLow"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "inputHigh"), 127.f / 4.f, "custom");
    fillData(getLayer(network, "outputLow"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "outputHigh"), 127.f / 4.f, "custom");

    fillDataMy(getLayer(network, "reshapeConst1"), { 0, 1280, 7, 1 }, "custom");
    fillDataMy(getLayer(network, "reshapeConst2"), { 0, 1280 }, "custom");
}

std::string FakeQuantizeReshapePoolingTestModelWithConstants::getName() const {
    return "FakeQuantizeReshapePoolingTestModelWithConstants";
}

bool FakeQuantizeReshapePoolingTestModelWithConstants::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

std::string FakeQuantizeReshapePoolingTestModelWithConstants::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::conv_common_params conv =
            { {1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 32, false, false };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::vector<size_t> weightsConstInputDims = { 32lu, 32lu, 3lu, 3lu };
    std::vector<size_t> biasesConvolutionConstDims = { conv.out_c };
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fakeQuantizeParams = {{ "levels", "256" }};
    std::map<std::string, std::string> power_params = {{"power", "1"}, {"scale", "1"}, {"shift", "0"}};
    std::map<std::string, std::string> poolingParams = { {"kernel", "7,1"}, { "pool-method", "avg" }, { "strides", "1,1" } };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, // input => inputPower
        {"1,2", "6,7"}, // inputPower => fakeQuantize
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers => fakeQuantize
        {"6,12", "8,14"}, // fakeQuantize => reshape1
        {"7,13", "8,15"}, // reshapeConst1 => reshape1
        {"8,16", "9,17"}, // reshape1 => pooling
        {"9,18", "11,20"}, // pooling => reshape2
        {"10,19", "11,21"}, // reshapeConst2 => reshape2
        {"11,22", "12,23"}, // reshape2 => outputPower
    };

    auto network = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
        "QuantizationOnWeights", p.inputDimensions[0], p._network_precision)
        // inputPower: id=1
        .addLayer("Power", p._network_precision, &power_params, { {p.inputDimensions[0]}, {p.inputDimensions[0]} }, "inputPower")
        // inputLow: id=2
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "inputLow")
        // inputHigh: id=3
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "inputHigh")
        // outputLow: id=4
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "outputLow")
        // outputHigh: id=5
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "outputHigh")
        // fakeQuantize: id=6
        .addLayer("FakeQuantize", p._network_precision, &fakeQuantizeParams, { {p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}} }, "fakeQuantize")
        // reshapeConst1: id=7
        .addLayer("Const", "I32", {}, { {}, {{4}} }, 4 * 4, "reshapeConst1")
        // reshape1: id=8
        .addLayer("Reshape", p._network_precision, {}, { {{ 1, 1280, 7 }, {4}}, {{1, 1280, 7, 1}} }, "reshape1")
        // pooling: id=9
        .addLayer("Pooling", p._network_precision, &poolingParams, { {{ 1, 1280, 7, 1 }}, {{1, 1280, 1, 1}} }, "pooling")
        // reshapeConst2: id=10
        .addLayer("Const", "I32", {}, { {}, {{2}} }, 2 * 4, "reshapeConst2")
        // reshape2: id=11
        .addLayer("Reshape", p._network_precision, {}, { {{ 1, 1280, 1, 1 }, {2}}, {{1, 1280 }} }, "reshape2")
        // outputPower: id=12
        .addLayer("Power", p._network_precision, &power_params, { {{ 1, 1280 }}, {{1, 1280}} }, "outputPower")
        .finish(&edges);
    return network;
}
