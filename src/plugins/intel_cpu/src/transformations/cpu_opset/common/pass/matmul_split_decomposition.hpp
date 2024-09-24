// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

/**
 * @brief MatmulGatherDecomposition transformation matches following graph:
 *
 *         +----------+
 *         |  input   |
 *         +----------+
 *              |
 *              v
 *         +----------+
 *         |  MatMul  |
 *         +----------+
 *              |
 *              v
 *         +------------+
 *         | Some nodes |
 *         +------------+
 *              |
 *              v
 *         +-----------------------+
 *         |       Transpose       |
 *         +-----------------------+
 *          |          |          |
 *          v          v          v
 *     +-------+   +-------+   +-------+
 *     |Gather |   |Gather |   |Gather |
 *     +-------+   +-------+   +-------+
 * and replaces with:
 *
 *         +-----------------------+
 *         |       input           |
 *         +-----------------------+
 *          |          |          |
 *          v          v          v
 *     +-------+   +-------+   +-------+
 *     |MatMul |   |MatMul |   |MatMul |
 *     +-------+   +-------+   +-------+
 *          |          |          |
 *          v          v          v
 *     +-------+   +-------+   +-------+
 *     |Nodes  |   |Nodes  |   |Nodes  |
 *     +-------+   +-------+   +-------+
 *          |          |          |
 *          v          v          v
 *   +---------+  +---------+  +---------+
 *   |Transpose|  |Transpose|  |Transpose|
 *   +---------+  +---------+  +---------+
 */

class MatmulGatherDecomposition : public pass::MatcherPass {
public:
    OPENVINO_RTTI("MatmulGatherDecomposition", "0");
    MatmulGatherDecomposition();
    bool split_weights(const Output<Node>& weights,
                       OutputVector& new_weights,
                       Output<Node>* bias,
                       OutputVector& new_bias,
                       const bool transpose_b);

private:
    const size_t decompose_num = 3;
};

}  // namespace intel_cpu
}  // namespace ov