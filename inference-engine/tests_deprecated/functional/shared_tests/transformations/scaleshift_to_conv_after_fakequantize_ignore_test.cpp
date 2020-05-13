// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 0.f, "custom");
    fillData(getLayer(network, "Const3"), 255.f / 8.f, "custom");
    fillData(getLayer(network, "Const4"), 0.f, "custom");
    fillData(getLayer(network, "Const5"), 255.f / 8.f, "custom");

    fillData(getLayer(network, "ScaleShift7"), 3.f, "weights");
    fillData(getLayer(network, "ScaleShift7"), 0.f, "biases");
}

std::string ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel::getName() const {
    return "ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel";
}

bool ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);

    CNNLayerPtr scaleShift = CNNNetworkHelper::getLayer(network, "ScaleShift7");
    if (scaleShift != nullptr) {
        THROW_IE_EXCEPTION << "unexpected layer " << scaleShift->type << " '" << scaleShift->name << "'";
    }

    return true;
}

std::string ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size;
    if (p._network_precision == "FP16") {
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);
    } else if (p._network_precision == "FP32") {
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    } else {
        THROW_IE_EXCEPTION << "unexpected network precision " << p._network_precision;
    }

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = { {"levels", "256"} };
    std::map<std::string, std::string> power_params = { {"power", "2"}, {"scale", "1"}, {"shift", "0"} };
    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, // Input -> Power
        {"1,2", "6,7"}, // Power -> FakeQuantize
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"}, // FakeQuantize -> ScaleShift
        {"7,14", "8,15"}, // FakeQuantize -> ScaleShift
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel", p.inputDimensions[0], p._network_precision)
        // 1
        .addLayer("Power", p._network_precision, &power_params, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        // 2
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 3
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 4
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 5
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 6
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, { {p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}} })
        // 7
        .addLayer("ScaleShift", p._network_precision, {}, { {p.inputDimensions[0]}, {p.inputDimensions[0]} }, p.inputDimensions[0][1] * type_size, p.outputDimensions[0][1] * type_size)
        // 8
        .addLayer("Power", p._network_precision, &power_params, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        .finish(&edges);
}
