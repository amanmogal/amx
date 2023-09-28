// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_pixel_shuffle(const NodeContext& context) {
    // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto upscale_factor = context.get_input(1);
    auto neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto neg_3 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-3}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto zero_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto one_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true);
    // 1. Reshape input to [*, -1, r, r, H, W], where r is upscale factor
    auto indices = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {-3, -2, -1}));
    auto dims = context.mark_node(std::make_shared<v8::Gather>(shape, indices, zero_s));
    auto dims_splitted = context.mark_node(std::make_shared<v1::Split>(dims, zero_s, 3));
    auto c = dims_splitted->output(0);
    auto h = dims_splitted->output(1);
    auto w = dims_splitted->output(2);
    auto dims_before = context.mark_node(std::make_shared<v8::Slice>(shape, zero, neg_3, one));
    auto upscale_factor_1d = context.mark_node(std::make_shared<v1::Reshape>(upscale_factor, neg_1, false));
    auto intermediate_shape = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{dims_before, neg_1, upscale_factor_1d, upscale_factor_1d, h, w}, 0));
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(x, intermediate_shape, false));
    // 2. Transpose tensor to [*, C, r, H, r, W]
    auto dims_before_len = context.mark_node(std::make_shared<v3::ShapeOf>(dims_before, element::i32));
    auto dims_before_len_s = context.mark_node(std::make_shared<v0::Squeeze>(dims_before_len, zero));
    auto order_begin = context.mark_node(std::make_shared<v4::Range>(zero_s, dims_before_len_s, one_s, element::i32));
    auto order_end_neg = context.mark_node(
        v0::Constant::create(element::i32, Shape{5}, {-3, 0, -2, 1, -1}));  // +2 because rank is expanded
    auto order_end = context.mark_node(std::make_shared<v1::Add>(order_end_neg, rank));
    auto order = context.mark_node(std::make_shared<v0::Concat>(OutputVector{order_begin, order_end}, 0));
    auto transpose = context.mark_node(std::make_shared<v1::Transpose>(reshape, order));
    // 3. Reshape to [*, -1, r * H, r * W]
    auto new_h = context.mark_node(std::make_shared<v1::Multiply>(h, upscale_factor));
    auto new_w = context.mark_node(std::make_shared<v1::Multiply>(w, upscale_factor));
    auto shape_after =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{dims_before, neg_1, new_h, new_w}, 0));
    return {context.mark_node(std::make_shared<v1::Reshape>(transpose, shape_after, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov