// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/search_sorted.hpp"

namespace ov {
namespace intel_gpu {

static void CreateSearchSortedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::SearchSorted>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    auto roi_align_prim = cldnn::search_sorted(layer_type_name_ID(op), inputs[0], inputs[1], op->get_right_mode());
    p.add_primitive(*op, roi_align_prim);
}

REGISTER_FACTORY_IMPL(v15, SearchSorted);

}  // namespace intel_gpu
}  // namespace ov
