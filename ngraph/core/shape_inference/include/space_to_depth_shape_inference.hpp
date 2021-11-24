// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/space_to_depth.hpp>
#include <openvino/opsets/opset1.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const ov::op::v0::SpaceToDepth* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const ov::Rank data_rank = data_shape.rank();
    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              !(data_shape.size() < 3),
                              "The input tensor with rank lower than 3 is not supported (input rank: ",
                              data_shape.size(),
                              ")");

        const auto block_size = op->get_block_size();
        NODE_VALIDATION_CHECK(op, block_size > 0, "The block size must begreater then 0 ", block_size);
        const int64_t multiplier = std::pow(block_size, data_shape.size() - 2);

        auto& out_shape = output_shapes[0];
        out_shape.resize(data_shape.size());

        out_shape[0] = data_shape[0];
        out_shape[1] = DimType{multiplier} * data_shape[1];
        for (size_t i = 2; i < out_shape.size(); i++) {
            if (data_shape[i].is_static()) {
                NODE_VALIDATION_CHECK(op,
                                      !(data_shape[i].get_length() % block_size),
                                      "The dimension on position: ",
                                      i,
                                      " equal to: ",
                                      data_shape[i],
                                      " must be a multiple of m_blocksize: ",
                                      block_size);

                out_shape[i] = data_shape[i].get_length() / block_size;
            } else {
                if (data_shape[i] == ov::Dimension::dynamic())
                    out_shape[i] = ov::Dimension::dynamic();
                else
                    out_shape[i] = ov::Dimension{data_shape[i].get_min_length() / static_cast<int64_t>(block_size) , 
                                                data_shape[i].get_max_length() / static_cast<int64_t>(block_size)};
            }
        }
    } else {
        // For PartialShape, Set the output to be dynamic;
        // For StaticShape, will throw error caused by implicitly constructing StaticShape with PartialShape argument;
        output_shapes[0] = ov::PartialShape::dynamic(data_rank);
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov