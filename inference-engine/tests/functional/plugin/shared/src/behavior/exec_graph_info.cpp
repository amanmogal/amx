// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include <details/ie_cnn_network_tools.h>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "exec_graph_info.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/exec_graph_info.hpp"


namespace LayerTestsDefinitions {
    std::string ExecGraphTests::getTestCaseName(testing::TestParamInfo<ExecGraphParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            result << "configItem=" << configuration.begin()->first << "_" << configuration.begin()->second;
        }
        return result.str();
    }

    void ExecGraphTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void ExecGraphTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

    inline std::vector<std::string> separateStrToVec(std::string str, const char sep) {
        std::vector<std::string> result;

        std::istringstream stream(str);
        std::string strVal;

        while (getline(stream, strVal, sep)) {
            result.push_back(strVal);
        }
        return result;
    }

TEST_P(ExecGraphTests, CheckExecGraphInfoBeforeExecution) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::CNNNetwork execGraph;
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    if (targetDevice == CommonTestUtils::DEVICE_CPU || targetDevice == CommonTestUtils::DEVICE_GPU) {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        // Store all the original layers from the network
        const auto originalLayers = function->get_ops();
        std::map<std::string, int> originalLayersMap;
        for (const auto &layer : originalLayers) {
            if (layer->description() == "Result")
                continue;
            originalLayersMap[layer->get_friendly_name()] = 0;
        }
        int IteratorForLayersConstant = 0;
        // Store all the layers from the executable graph information represented as CNNNetwork
        const std::vector<InferenceEngine::CNNLayerPtr> execGraphLayers =
                InferenceEngine::details::CNNNetSortTopologically(execGraph);
        for (const auto &execLayer : execGraphLayers) {
            IE_SUPPRESS_DEPRECATED_START
            // Each layer from the execGraphInfo network must have PM data option set
            ASSERT_EQ("not_executed", execLayer->params[ExecGraphInfoSerialization::PERF_COUNTER]);
            // Parse origin layer names (fused/merged layers) from the executable graph
            // and compare with layers from the original model
            auto origFromExecLayer = execLayer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES];
            if (origFromExecLayer == "")
                IteratorForLayersConstant++;
            std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            });
        }
        // All layers from the original IR must be present with in ExecGraphInfo
        for (auto &layer : originalLayersMap) {
            if ((layer.second == 0) && (IteratorForLayersConstant > 0)) {
                IteratorForLayersConstant--;
                continue;
            }
            ASSERT_GE(layer.second, 0);
        }
    } else {
        ASSERT_THROW(ie->LoadNetwork(cnnNet, targetDevice).GetExecGraphInfo(),
                         InferenceEngine::details::InferenceEngineException);
    }
    IE_SUPPRESS_DEPRECATED_END
    function.reset();
}

TEST_P(ExecGraphTests, CheckExecGraphInfoAfterExecution) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::CNNNetwork execGraph;
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    if (targetDevice == CommonTestUtils::DEVICE_CPU || targetDevice == CommonTestUtils::DEVICE_GPU) {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        // Store all the original layers from the network
        const auto originalLayers = function->get_ops();
        std::map<std::string, int> originalLayersMap;
        for (const auto &layer : originalLayers) {
            originalLayersMap[layer->get_friendly_name()] = 0;
        }
        int IteratorForLayersConstant = 0;
        // Store all the layers from the executable graph information represented as CNNNetwork
        const std::vector<InferenceEngine::CNNLayerPtr> execGraphLayers =
                InferenceEngine::details::CNNNetSortTopologically(execGraph);
        bool has_layer_with_valid_time = false;
        for (const auto &execLayer : execGraphLayers) {
            IE_SUPPRESS_DEPRECATED_START
            // At least one layer in the topology should be executed and have valid perf counter value
            try {
                float x = static_cast<float>(std::atof(
                        execLayer->params[ExecGraphInfoSerialization::PERF_COUNTER].c_str()));
                ASSERT_GE(x, 0.0f);
                has_layer_with_valid_time = true;
            } catch (std::exception &) {}

            // Parse origin layer names (fused/merged layers) from the executable graph
            // and compare with layers from the original model
            auto origFromExecLayer = execLayer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES];
            std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');
            if (origFromExecLayer == "")
                IteratorForLayersConstant++;
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            });
        }
        ASSERT_TRUE(has_layer_with_valid_time);

        // All layers from the original IR must be present within ExecGraphInfo
        for (auto &layer : originalLayersMap) {
            if ((layer.second == 0) && (IteratorForLayersConstant > 0)) {
                IteratorForLayersConstant--;
                continue;
            }
            ASSERT_GE(layer.second, 0);
        }
    } else {
        ASSERT_THROW(ie->LoadNetwork(cnnNet, targetDevice).GetExecGraphInfo(),
                InferenceEngine::details::InferenceEngineException);
    }
        IE_SUPPRESS_DEPRECATED_END
    function.reset();
}

TEST_P(ExecGraphTests, CheckExecGraphInfoSerialization) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::CNNNetwork execGraph;
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    if (targetDevice == CommonTestUtils::DEVICE_CPU || targetDevice == CommonTestUtils::DEVICE_GPU) {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        execGraph.serialize("exeNetwork.xml", "exeNetwork.bin");
        ASSERT_EQ(0, std::remove("exeNetwork.xml"));
    } else {
        ASSERT_THROW(ie->LoadNetwork(cnnNet, targetDevice).GetExecGraphInfo(),
                InferenceEngine::details::InferenceEngineException);
    }
    function.reset();
}
}  // namespace LayerTestsDefinitions