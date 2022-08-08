// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "openvino/op/augru_sequence.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::AUGRUSequence;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
