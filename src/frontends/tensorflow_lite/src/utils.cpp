// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset10.hpp>
#include "schema_generated.h"
#include "tensor_lite_place.hpp"
#include "decoder_flatbuffer.h"
#include "utils.hpp"

using namespace ov;

ov::frontend::tensorflow_lite::Quantization ov::frontend::tensorflow_lite::get_quantization(const tflite::QuantizationParameters* tf_quantization) {
    if (tf_quantization == NULL)
        return {};
    ov::frontend::tensorflow_lite::Quantization quantization;
    auto tf_zp = tf_quantization->zero_point();
    auto tf_scale = tf_quantization->scale();
    if (tf_zp != NULL)
        quantization.zero_point = {(*tf_zp).begin(), (*tf_zp).end()};
    if (tf_scale != NULL)
        quantization.scale = {(*tf_scale).begin(), (*tf_scale).end()};
    if (quantization.zero_point.empty() && quantization.scale.empty())
        return {};
    quantization.axis = tf_quantization->quantized_dimension();
    quantization.no_quantization = false;
    return quantization;
}

namespace {
    const std::map<tflite::TensorType, ov::element::Type> &TYPE_MAP() {
        static const std::map<tflite::TensorType, ov::element::Type> type_map{
                {tflite::TensorType_FLOAT32,        element::f32},
                {tflite::TensorType_FLOAT16,        element::f16},
                {tflite::TensorType_INT32,          element::i32},
                {tflite::TensorType_UINT8,          element::u8},
                {tflite::TensorType_INT64,          element::i64},
                {tflite::TensorType_BOOL,           element::boolean},
                {tflite::TensorType_INT16,          element::i16},
                {tflite::TensorType_INT8,           element::i8},
                {tflite::TensorType_FLOAT64,        element::f64},
                {tflite::TensorType_UINT64,         element::u64},
                {tflite::TensorType_UINT32,         element::u32},
                {tflite::TensorType_UINT16,         element::u16},
                {tflite::TensorType_INT4,           element::i4},
// TODO: support the following types
//          {TensorType_STRING,         element::string},
//          {TensorType_COMPLEX64,      element::complex64},
//          {TensorType_COMPLEX128,     element::complex128},
//          {TensorType_RESOURCE,       element::resource},
//          {TensorType_VARIANT,        element::variant},
        };
        return type_map;
    }
}

ov::element::Type ov::frontend::tensorflow_lite::get_ov_type(const tflite::TensorType& tf_type) {
    const auto& mapping = TYPE_MAP();
    if (mapping.find(tf_type) == mapping.end()) {
        FRONT_END_THROW("Unexpected type");
    }
    return mapping.at(tf_type);
}

ov::PartialShape ov::frontend::tensorflow_lite::get_ov_shape(const flatbuffers::Vector<int32_t>* tf_shape) {
    return ov::Shape{tf_shape->begin(), tf_shape->end()};
}

ov::Output<ov::Node> ov::frontend::tensorflow_lite::apply_quantization(
        ov::Output<ov::Node> output, const std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace>& tensor, bool is_input) {
    auto quantization = tensor->get_quantization();
    if (quantization.no_quantization)
        return output;

    auto input_type = output.get_element_type();
    ov::Output<ov::Node> input_low, input_high, output_low, output_high;

    auto zp = quantization.zero_point;
    auto scale = quantization.scale;
    auto zp_node = ov::opset10::Constant::create(element::f32, (zp.size() == 1 ? ov::Shape{} : ov::Shape{zp.size()}), zp);
    auto scale_node = ov::opset10::Constant::create(element::f32, (scale.size() == 1 ? ov::Shape{} : ov::Shape{scale.size()}), scale);

    if (ov::is_type<ov::opset10::Constant>(output.get_node_shared_ptr())) {
        output = std::make_shared<ov::opset10::Convert>(output, element::f32);
        if (std::any_of(zp.begin(), zp.end(), [](const int64_t& i){ return i != 0; }))
            output = std::make_shared<ov::opset10::Subtract>(output, zp_node);
        output = std::make_shared<ov::opset10::Multiply>(output, scale_node);
        return output;
    }

    auto levels = 256;
    if (is_input) {
        FRONT_END_GENERAL_CHECK(input_type == element::u8, "Inputs of type other than u8 is not yet supported");
        output = std::make_shared<ov::opset10::Convert>(output, element::f32);
        input_low = ov::opset10::Constant::create(element::f32, {}, {0});
        input_high = ov::opset10::Constant::create(element::f32, {}, {levels - 1});
    }
   if (std::all_of(zp.begin(), zp.end(), [](const int64_t& i){ return i == 0; })) {
        output_low = ov::opset10::Constant::create(element::f32, {}, {0});
    } else {
        output_low = std::make_shared<opset10::Multiply>(std::make_shared<opset10::Negative>(scale_node), zp_node);
    }
    output_high = std::make_shared<opset10::Multiply>(scale_node, std::make_shared<opset10::Subtract>(ov::opset10::Constant::create(element::f32, {}, {levels - 1}), zp_node));
    if (!is_input) {
        input_low = output_low;
        input_high = output_high;
    }
    auto fq = std::make_shared<opset10::FakeQuantize>(output, input_low, input_high, output_low, output_high, levels);
    tensor->disable_quantization(); // we applied parameters -- disable them so that they won't apply twice
    return fq;
}