// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_acc_values(float** outs, float* weights, void** vs, size_t vec_num, size_t vec_len, ov::element::Type input_precision);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine