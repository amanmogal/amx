// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include <pugixml.hpp>

#include "shared_test_classes/base/utils/ranges.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "functional_test_utils/summary/op_info.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "input_info.hpp"
#include "conformance.hpp"
#include "read_ir_test/read_ir.hpp"

#include <setjmp.h>

#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
namespace test {
namespace conformance {
// It is used while files lookup
std::list<std::string> dirList;
}
namespace subgraph {

ShapeMode shapeMode = ShapeMode::BOTH;

std::string ReadIRTest::getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj) {
    using namespace CommonTestUtils;
    std::pair<std::string, std::string> model_pair;
    std::string path_to_model, path_to_cache, deviceName;
    ov::AnyMap config;
    std::tie(model_pair, deviceName, config) = obj.param;
    std::tie(path_to_model, path_to_cache) = model_pair;

    std::ostringstream result;
    auto splittedFilename = CommonTestUtils::splitStringByDelimiter(path_to_model, CommonTestUtils::FileSeparator);
    std::reverse(splittedFilename.begin(), splittedFilename.end());
    bool is_valid_path_format = true;

    // Check that op is valid
    if (splittedFilename.size() > 2) {
        auto pos = splittedFilename[2].find('-');
        std::string op_name = splittedFilename[2], op_version = "";
        if (pos != std::string::npos) {
            op_name = splittedFilename[2].substr(0, pos);
            op_version = splittedFilename[2].substr(pos + 1);
        }
        if (std::find(ov::test::conformance::unique_ops[op_name].begin(),
                      ov::test::conformance::unique_ops[op_name].end(), op_version) != ov::test::conformance::unique_ops[op_name].end() &&
            ov::test::conformance::unique_ops.find(op_name) != ov::test::conformance::unique_ops.end()) {
            std::string message = "Op=" + op_name;
            if (op_version != "") {
                message += "." + op_version;
            }
            message += "_";
            result << message;
        } else {
            is_valid_path_format = false;
        }
    }
    // Check the element_type
    if (splittedFilename.size() > 1) {
        if (std::find(ov::test::conformance::element_type_names.begin(),
                      ov::test::conformance::element_type_names.end(),
                      splittedFilename[1]) != ov::test::conformance::element_type_names.end()) {
            result << "Type=" << splittedFilename[1] << "_";
        } else {
            is_valid_path_format = false;
        }
    }
    result << "IR=" << (is_valid_path_format ? CommonTestUtils::replaceExt(splittedFilename[0], "") : path_to_model) << "_";
    result << "Device=" << deviceName << "_";
    result << "Config=(";
    auto configItem = config.begin();
    while (configItem != config.end()) {
        result << configItem->first << "=";
        configItem->second.print(result);
        if (++configItem != config.end()) {
            result << "_";
        }
    }
    result << ")";
    return result.str();
}

void ReadIRTest::query_model() {
    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<CommonTestUtils::CrashHandler>(new CommonTestUtils::CrashHandler());
    auto &s = ov::test::utils::OpSummary::getInstance();

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(CommonTestUtils::env);
#else
    jmpRes = sigsetjmp(CommonTestUtils::env, 1);
#endif
    if (jmpRes == CommonTestUtils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        if (functionRefs == nullptr) {
            functionRefs = ngraph::clone_function(*function);
            functionRefs->set_friendly_name("refFunction");
        }
        s.setDeviceName(targetDevice);

        if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::SKIPPED, rel_influence_coef);
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
        } else {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::CRASHED, rel_influence_coef);
        }
        try {
            SubgraphBaseTest::query_model();
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::PASSED, rel_influence_coef);
        } catch (std::exception& err) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED, rel_influence_coef);
            GTEST_FAIL() << err.what();
        } catch (...) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED, rel_influence_coef);
            GTEST_FAIL() << "Something is wrong in Query model! Please check";
        }
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::alarmErr) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
        IE_THROW() << "Crash happens";
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::anyError) {
        IE_THROW() << "Crash happens";
    }
}

uint64_t clip(uint64_t n, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(n, upper));
}

void ReadIRTest::SetUp() {
    std::pair<std::string, std::string> model_pair;
    std::tie(model_pair, targetDevice, configuration) = this->GetParam();
    std::tie(path_to_model, path_to_cache) = model_pair;
    function = core->read_model(path_to_model);
    const auto metaFile = CommonTestUtils::replaceExt(path_to_model, "meta");
    if (CommonTestUtils::fileExists(metaFile)) {
        pugi::xml_document doc;
        doc.load_file(metaFile.c_str());
        rel_influence_coef = doc.child("meta_info").child("graph_priority").attribute("value").as_double();
        // TODO: remove after cache update w/a
        if (rel_influence_coef == 0) {
            rel_influence_coef = 1.f;
        }
        auto input_info_xml = doc.child("meta_info").child("input_info");
        std::map<std::string, ov::tools::subgraph_dumper::InputInfo> input_info;
        for (const auto &input : input_info_xml.children()) {
            auto in_name = std::string(input.attribute("id").value());
            ov::tools::subgraph_dumper::InputInfo in_info;
            in_info.is_const = input.attribute("convert_to_const").as_bool();
            if (std::string(input.attribute("min").value()) != "undefined") {
                in_info.ranges.min = input.attribute("min").as_double();
            }
            if (std::string(input.attribute("max").value()) != "undefined") {
                in_info.ranges.max = input.attribute("max").as_double();
            }
            input_info.insert({in_name, in_info});
        }
        auto inputMap = utils::getInputMap();
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> parameter_to_remove;
        for (const auto& param : function->get_parameters()) {
            auto in_info = input_info.find(param->get_friendly_name())->second;
            if (!in_info.is_const) {
                continue;
            }
            utils::ConstRanges::set(in_info.ranges.min, in_info.ranges.max);
            // auto next_node = param->get_default_output().get_node_shared_ptr();
            auto next_node = param->get_default_output().get_target_inputs().begin()->get_node()->shared_from_this();
            auto it = inputMap.find(next_node->get_type_info());
            auto tensor = it->second(next_node, function->get_parameter_index(param), param->get_element_type(), param->get_shape());
            auto const_node = std::make_shared<ov::op::v0::Constant>(tensor);
            ov::replace_node(param, const_node);
            parameter_to_remove.push_back(param);
            utils::ConstRanges::reset();
        }
        for (const auto& param : parameter_to_remove) {
            function->remove_parameter(param);
        }
    }

    bool hasDynamic = false;
    for (const auto& param : function->get_parameters()) {
        if (param->get_partial_shape().is_dynamic()) {
            hasDynamic = true;
            break;
        }
    }
    if (!hasDynamic) {
        for (const auto& result : function->get_results()) {
            if (result->get_output_partial_shape(0).is_dynamic()) {
                hasDynamic = true;
                break;
            }
        }
    }
    if (hasDynamic && ov::test::subgraph::shapeMode == ov::test::subgraph::ShapeMode::STATIC) {
        GTEST_SKIP() << "Dynamic cases are skipped according `shape_mode`";
    } else if (!hasDynamic && ov::test::subgraph::shapeMode == ov::test::subgraph::ShapeMode::DYNAMIC) {
        GTEST_SKIP() << "Static cases are skipped according `shape_mode`";
    }

    std::vector<InputShape> inputShapes;
    for (const auto& param : function -> get_parameters()) {
        if (param->get_partial_shape().is_static()) {
            inputShapes.push_back(InputShape{{}, {param->get_shape()}});
        } else {
            std::vector<ov::Shape> staticShapes = { param->get_partial_shape().get_min_shape(),
                                                    param->get_partial_shape().get_min_shape(),
                                                    param->get_partial_shape().get_max_shape() };
            ov::Shape midShape;
            for (const auto s : param->get_partial_shape()) {
                int dimValue = 1;
                if (s.is_dynamic()) {
                    size_t range = s.get_max_length() - s.get_min_length();
                    if (range > std::numeric_limits<char>::max()) {
                        CommonTestUtils::fill_data_random(&range, 1, std::numeric_limits<char>::max(), s.get_min_length(), 1);
                    }
                    CommonTestUtils::fill_data_random(&dimValue, 1, range, s.get_min_length(), 1);
                } else {
                    dimValue = s.get_length();
                }
                midShape.push_back(dimValue);
            }
            staticShapes[1] = midShape;

            // Shape validation to avoid large values
            uint64_t dimMin = 1;
            uint64_t dimMax = std::numeric_limits<char>::max();
            for (int i = 0; i < staticShapes[0].size(); ++i) {
                auto& dim0 = staticShapes[0][i];
                auto& dim2 = staticShapes[2][i];
                if (dim0 != dim2) {
                    dim0 = clip(dim0, dimMin, dimMax);
                    dim2 = clip(dim2, dimMin, dimMax);
                }
            }
            inputShapes.push_back(InputShape{param->get_partial_shape(), staticShapes});
        }
    }
    if (inputShapes.empty()) {
        GTEST_SKIP() << "The graph is constant. The case is not applicable for Operation conformance scenario";
    }
    std::cout << "[ CONFORMANCE ] Influence coefficient: " << rel_influence_coef << std::endl;
    init_input_shapes(inputShapes);
    is_report_stages = true;
}

std::vector<ov::Tensor> ReadIRTest::calculate_refs() {
    auto start_time = std::chrono::system_clock::now();
    if (is_report_stages) {
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
    }
    ov::TensorVector output_tensors;
    if (!CommonTestUtils::fileExists(path_to_cache)) {
        std::cout << "[ REFERENCE   ] Calculate reference in runtime" << std::endl;
        output_tensors = SubgraphBaseTest::calculate_refs();
        if (path_to_cache != "") {
            std::ofstream ofstream_tensor(path_to_cache, std::ios::out | std::ios::binary);
            for (const auto& out_tensor : output_tensors) {
                ofstream_tensor.write(reinterpret_cast<const char*>(out_tensor.data()), out_tensor.get_byte_size());
            }
            ofstream_tensor.close();
        }
    } else {
        std::cout << "[ REFERENCE   ] Read reference from file: " << path_to_cache << std::endl;
        // Because of functionRefs is a static function
        std::ifstream ref_data_ifstream(path_to_cache, std::ifstream::binary);
        ref_data_ifstream.open(path_to_cache, std::ios::binary);
        if (!ref_data_ifstream.is_open())
            IE_THROW() << "Weights file " << path_to_cache << " cannot be opened!";

        size_t buf_size = 0;
        for (const auto& output : functionRefs->outputs()) {
            buf_size += (sizeof output.get_element_type() * ov::shape_size(output.get_partial_shape().get_shape()));
        }
        char* ref_buffer = nullptr;
        ref_data_ifstream.read(ref_buffer, buf_size);

        size_t pos = 0;
        for (const auto& output : functionRefs->outputs()) {
            auto out_tensor = ov::runtime::Tensor(output.get_element_type(), output.get_shape(), &ref_buffer[pos]);
            pos += out_tensor.get_byte_size();
        }
    }
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
    return output_tensors;
}

} // namespace subgraph
} // namespace test
} // namespace ov


