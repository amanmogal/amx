// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GatherSinkingMatmulForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingMatmulForward", "0");
    GatherSinkingMatmulForward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
