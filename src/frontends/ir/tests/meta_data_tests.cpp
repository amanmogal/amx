// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "file_utils.h"
#include "openvino/core/any.hpp"
#include "openvino/openvino.hpp"

class MetaData : public ::testing::Test {
public:
    ov::Core core;

    std::string ir_with_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
    <meta_data>
        <MO_version value="TestVersion"/>
        <Runtime_version value="TestVersion"/>
        <cli_parameters>
            <input_shape value="[1, 3, 22, 22]"/>
            <transform value=""/>
            <use_new_frontend value="False"/>
        </cli_parameters>
    </meta_data>
</net>
)V0G0N";

    std::string ir_without_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    void SetUp() override {}
};

TEST_F(MetaData, get_meta_data_from_model_without_info) {
    auto model = core.read_model(ir_without_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    ASSERT_EQ(rt_info.find("meta_data"), rt_info.end());
}

TEST_F(MetaData, get_meta_data_as_map_from_model_without_info) {
    auto model = core.read_model(ir_without_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    auto it = rt_info.find("meta_data");
    EXPECT_EQ(it, rt_info.end());
    ov::AnyMap meta;
    ASSERT_THROW(meta = model->get_rt_info<ov::AnyMap>("meta_data"), ov::Exception);
    ASSERT_TRUE(meta.empty());
}

TEST_F(MetaData, get_meta_data) {
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    ASSERT_NE(rt_info.find("MO_version"), rt_info.end());
    ASSERT_NE(rt_info.find("Runtime_version"), rt_info.end());
    ASSERT_NE(rt_info.find("conversion_parameters"), rt_info.end());
}

TEST_F(MetaData, get_meta_data_as_map) {
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    ASSERT_TRUE(!rt_info.empty());
    auto it = rt_info.find("MO_version");
    ASSERT_NE(it, rt_info.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

    it = rt_info.find("Runtime_version");
    ASSERT_NE(it, rt_info.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

    ov::AnyMap cli_map;
    EXPECT_NO_THROW(cli_map = model->get_rt_info<ov::AnyMap>("conversion_parameters"));

    it = cli_map.find("input_shape");
    ASSERT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "[1, 3, 22, 22]");

    it = cli_map.find("transform");
    ASSERT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "");

    it = cli_map.find("use_new_frontend");
    ASSERT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "False");
}

TEST_F(MetaData, get_meta_data_from_removed_file) {
    std::string file_path =
        InferenceEngine::getIELibraryPath() + ov::util::FileTraits<char>::file_separator + "test_model.xml";
    // Create file
    {
        std::ofstream ir(file_path);
        ir << ir_with_meta;
    }
    auto model = core.read_model(file_path);

    // Remove file (meta section wasn't read)
    std::remove(file_path.c_str());

    auto& rt_info = model->get_rt_info();
    ASSERT_TRUE(!rt_info.empty());
    auto it = rt_info.find("MO_version");
    ASSERT_NE(it, rt_info.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

    it = rt_info.find("Runtime_version");
    ASSERT_NE(it, rt_info.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "TestVersion");

    ov::AnyMap cli_map;
    EXPECT_NO_THROW(cli_map = model->get_rt_info<ov::AnyMap>("conversion_parameters"));

    it = cli_map.find("input_shape");
    ASSERT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "[1, 3, 22, 22]");

    it = cli_map.find("transform");
    ASSERT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "");

    it = cli_map.find("use_new_frontend");
    ASSERT_NE(it, cli_map.end());
    EXPECT_TRUE(it->second.is<std::string>());
    EXPECT_EQ(it->second.as<std::string>(), "False");
}
