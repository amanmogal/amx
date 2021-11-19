// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/experimental_detectron_prior_grid_generator.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(const ExperimentalDetectronPriorGridGenerator* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    const auto& priors_shape = input_shapes[0];
    const auto& featmap_shape = input_shapes[1];
    const auto& im_data_shape = input_shapes[2];

    auto& output_shape = output_shapes[0];
    size_t output_size = op->m_attrs.flatten ? 2 : 4;

    output_shape.resize(output_size);
    output_shape[output_size - 1] = 4;

    if (priors_shape.rank().is_dynamic() || featmap_shape.rank().is_dynamic()) {
        return;
    }

    NODE_VALIDATION_CHECK(op, priors_shape.size() == 2, "Priors rank must be equal to 2.");

    NODE_VALIDATION_CHECK(op,
                          priors_shape[1].compatible(4),
                          "The last dimension of the 'priors' input must be equal to 4. Got: ",
                          priors_shape[1]);

    NODE_VALIDATION_CHECK(op, featmap_shape.size() == 4, "Feature_map rank must be equal to 4.");

    if (im_data_shape.rank().is_dynamic()) {
        return;
    }

    NODE_VALIDATION_CHECK(op, im_data_shape.size() == 4, "Im_data rank must be equal to 4.");

    const auto num_batches_featmap = featmap_shape[0];
    const auto num_batches_im_data = im_data_shape[0];

    NODE_VALIDATION_CHECK(op,
                          num_batches_featmap.compatible(num_batches_im_data),
                          "The first dimension of both 'feature_map' and 'im_data' must match. "
                          "Feature_map: ",
                          num_batches_featmap,
                          "; Im_data: ",
                          num_batches_im_data);

    auto num_priors = priors_shape[0];
    auto featmap_height = featmap_shape[2];
    auto featmap_width = featmap_shape[3];

    if (op->m_attrs.flatten) {
        output_shape[0] = featmap_height * featmap_width * num_priors;
    } else {
        output_shape[0] = featmap_height;
        output_shape[1] = featmap_width;
        output_shape[2] = num_priors;
    }
}

}  // namespace v6
}  // namespace op
}  // namespace ov
