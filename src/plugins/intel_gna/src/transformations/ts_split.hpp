// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief This transformation is a part of Transpose/Gather sinking group of transformations.
 * This transformation moves Transpose through Split layer. It changes Split axis.
 * Currently GNA plugin has restrictions working with Split layers.
 * It doesn't support all types of Split layers. This transformation allows to remove Transpose layer
 * on the Split output with adding Gather layer on the input with Split axis being supported.
 * Substitute graph from
 *          Any#1
 *           |
 *         Split
 *    |      |        |
 *    |   Transpose   |
 *    |      |        |
 * Any#2 .. Any#K .. Any#N
 * to
 *          Any#1
 *           |
 *         Reshape
 *           |
 *         Gather
 *           |
 *         Split
 *    |      |        |
 * Reshape Reshape   Reshape
 *    |      |        |
 * Any#2 .. Any#K .. Any#N
 */
class TSSplitBackward : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TSSplitBackward", "0");
    TSSplitBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
