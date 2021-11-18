// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reverse_sequence.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, reverse_sequence_op) {
    NodeBuilder::get_ops().register_factory<op::v0::ReverseSequence>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 2});
    auto seq_indices = make_shared<op::Parameter>(element::i32, Shape{4});

    auto batch_axis = 2;
    auto seq_axis = 1;

    auto reverse_sequence = make_shared<op::v0::ReverseSequence>(data, seq_indices, batch_axis, seq_axis);

    NodeBuilder builder(reverse_sequence);
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto g_reverse_sequence = ov::as_type_ptr<op::v0::ReverseSequence>(builder.create());

    EXPECT_EQ(g_reverse_sequence->get_origin_batch_axis(), reverse_sequence->get_origin_batch_axis());
    EXPECT_EQ(g_reverse_sequence->get_origin_sequence_axis(), reverse_sequence->get_origin_sequence_axis());
}
