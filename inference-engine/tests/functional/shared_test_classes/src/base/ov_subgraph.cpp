// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <signal.h>
#include <transformations/utils/utils.hpp>
#include <shared_test_classes/base/utils/generate_inputs.hpp>
#include <shared_test_classes/base/utils/compare_results.hpp>

#ifdef _WIN32
#include <process.h>
#endif

#include "openvino/pass/serialize.hpp"

#include "graph_comparator.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

void SubgraphBaseTest::run() {
    auto crashHandler = [](int errCode) {
        auto &s = LayerTestsUtils::Summary::getInstance();
        s.saveReport();
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
        std::abort();
    };
    signal(SIGSEGV, crashHandler);

    LayerTestsUtils::PassRate::Statuses status =
            FuncTestUtils::SkipTestsConfig::currentTestIsDisabled() ?
            LayerTestsUtils::PassRate::Statuses::SKIPPED : LayerTestsUtils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    ASSERT_FALSE(targetStaticShapes.empty()) << "Target Static Shape is empty!!!";
    std::string errorMessage;
//    try {
        compile_model();
        for (const auto& targetStaticShapeVec : targetStaticShapes) {
//            try {
                if (!inputDynamicShapes.empty()) {
                    // resize ngraph function according new target shape
                    ngraph::helpers::resize_function(functionRefs, targetStaticShapeVec);
                }
                generate_inputs(targetStaticShapeVec);
                infer();
                validate();
//            } catch (const std::exception &ex) {
//                throw std::runtime_error("Incorrect target static shape: " + CommonTestUtils::vec2str(targetStaticShapeVec) + " " + ex.what());
//            }
        }
        status = LayerTestsUtils::PassRate::Statuses::PASSED;
//    } catch (const std::exception &ex) {
//        status = LayerTestsUtils::PassRate::Statuses::FAILED;
//        errorMessage = ex.what();
//    } catch (...) {
//        status = LayerTestsUtils::PassRate::Statuses::FAILED;
//        errorMessage = "Unknown failure occurred.";
//    }
//    summary.updateOPsStats(function, status);
    if (status != LayerTestsUtils::PassRate::Statuses::PASSED) {
        GTEST_FATAL_FAILURE_(errorMessage.c_str());
    }
}

void SubgraphBaseTest::serialize() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::string output_name = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + GetTimestamp();

    std::string out_xml_path = output_name + ".xml";
    std::string out_bin_path = output_name + ".bin";

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(out_xml_path, out_bin_path);
    manager.run_passes(function);
    function->validate_nodes_and_infer_types();

    auto result = core->read_model(out_xml_path, out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
            compare_functions(result, function, false, false, false,
                              true,     // precision
                              true);    // attributes

    EXPECT_TRUE(success) << message;

    CommonTestUtils::removeIRFiles(out_xml_path, out_bin_path);
}

void SubgraphBaseTest::query_model() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto queryNetworkResult = core->query_model(function, targetDevice);
    std::set<std::string> expected;
    for (auto&& node : function->get_ops()) {
        expected.insert(node->get_friendly_name());
    }

    std::set<std::string> actual;
    for (auto&& res : queryNetworkResult) {
        actual.insert(res.first);
    }
    ASSERT_EQ(expected, actual);
}

void SubgraphBaseTest::compare(const std::vector<ov::runtime::Tensor> &expected,
                               const std::vector<ov::runtime::Tensor> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_EQ(expected.size(), function->get_results().size());
    auto compareMap = utils::getCompareMap();
    const auto& results = function->get_results();
    for (size_t j = 0; j < results.size(); j++) {
        const auto result = results[j];
        for (size_t i = 0; i < result->get_input_size(); ++i) {
            std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
            auto it = compareMap.find(inputNode->get_type_info());
            it->second(inputNode, i, expected[j], actual[j], abs_threshold, rel_threshold);
        }
    }
}

void SubgraphBaseTest::configure_model() {
    // configure input precision
    {
        auto params = function->get_parameters();
        for (auto& param : params) {
            if (inType != ov::element::Type_t::undefined) {
                param->get_output_tensor(0).set_element_type(inType);
            }
        }
    }

    // configure output precision
    {
        auto results = function->get_results();
        for (auto& result : results) {
            if (outType != ov::element::Type_t::undefined) {
                result->get_output_tensor(0).set_element_type(outType);
            }
        }
    }
}

void SubgraphBaseTest::compile_model() {
    configure_model();
    if (functionRefs == nullptr) {
        functionRefs = ov::clone_function(*function);
    }
    executableNetwork = core->compile_model(function, targetDevice, configuration);
}

void SubgraphBaseTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto inputMap = utils::getInputMap();
    for (const auto &param : function->get_parameters()) {
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto &node : param->get_output_target_inputs(i)) {
                const auto nodePtr = node.get_node()->shared_from_this();
                auto it = inputMap.find(nodePtr->get_type_info());
                auto itTargetShape = targetInputStaticShapes.begin();
                for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                    if (nodePtr->get_input_node_ptr(port)->shared_from_this() == param->shared_from_this()) {
                        inputs.insert({param, it->second(nodePtr, port, param->get_element_type(), *itTargetShape++)});
                        break;
                    }
                }
            }
        }
    }
}

void SubgraphBaseTest::infer() {
    inferRequest = executableNetwork.create_infer_request();
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }
    inferRequest.infer();
}

std::vector<ov::runtime::Tensor> SubgraphBaseTest::calculate_refs() {
    functionRefs->validate_nodes_and_infer_types();
    return ngraph::helpers::interpretFunction(functionRefs, inputs);
}

std::vector<ov::runtime::Tensor> SubgraphBaseTest::get_plugin_outputs() {
    auto outputs = std::vector<ov::runtime::Tensor>{};
    for (const auto& output : function->outputs()) {
        outputs.push_back(inferRequest.get_tensor(output));
    }
    return outputs;
}

void SubgraphBaseTest::validate() {
    auto expectedOutputs = calculate_refs();
    const auto& actualOutputs = get_plugin_outputs();

    if (expectedOutputs.empty()) {
        return;
    }

    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size()) << "nGraph interpreter has "
        << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    compare(expectedOutputs, actualOutputs);
}

void SubgraphBaseTest::init_input_shapes(const std::vector<InputShape>& shapes) {
    size_t targetStaticShapeSize = shapes.front().second.size();
    targetStaticShapes.resize(targetStaticShapeSize);
    for (const auto& shape : shapes) {
        auto dynShape = shape.first;
        if (dynShape.rank() == 0) {
            ASSERT_EQ(targetStaticShapeSize, 1) << "Incorrect number of static shapes for static case";
            dynShape = shape.second.front();
        }
        inputDynamicShapes.push_back(dynShape);
        ASSERT_EQ(shape.second.size(), targetStaticShapeSize) << "Target static count shapes should be the same for all inputs";
        for (size_t i = 0; i < shape.second.size(); ++i) {
            targetStaticShapes[i].push_back(shape.second.at(i));
        }
    }
}
}  // namespace test
}  // namespace ov