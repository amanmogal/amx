// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/read_value.hpp>

namespace ov {
namespace op {

template <class T1, class T2>
void read_value_shape_infer(T1* op, const std::vector<T2>& input_shapes, std::vector<T2>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    output_shapes[0] = input_shape;
}

namespace v3 {
template <class T>
void shape_infer(ReadValue* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    read_value_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v3

namespace v6 {
template <class T>
void shape_infer(ReadValue* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    read_value_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v6
}  // namespace op
}  // namespace ov