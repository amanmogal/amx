// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_system_conf.h>
#include <stdint.h>

namespace ov {
namespace intel_gna {

inline bool isAvx2Supported() {
#ifdef HAVE_AVX2
    return InferenceEngine::with_cpu_x86_avx2();
#else
    return false;
#endif  // HAVE_AVX2
}

namespace pre_post_processing {

void ConvertMatrixFp32ToInt16(int16_t* ptr_dst,
                              const float* ptr_src,
                              const uint32_t num_rows,
                              const uint32_t num_columns,
                              const float scale_factor,
                              bool transpose);

void ConvertMatrixFp32ToInt8(int8_t* ptr_dst,
                             const float* ptr_src,
                             const uint32_t num_rows,
                             const uint32_t num_columns,
                             const float scale_factor,
                             bool transpose);

void ConvertMatrixInt32ToFp32Avx(float* ptr_dst,
                                 const int32_t* ptr_src,
                                 uint32_t num_rows,
                                 uint32_t num_columns,
                                 float scale_factor,
                                 bool scale,
                                 bool transpose);

void ConvertMatrixInt16ToFp32Avx(float* ptr_dst,
                                 const int16_t* ptr_src,
                                 uint32_t num_rows,
                                 uint32_t num_columns,
                                 float scale_factor,
                                 bool scale,
                                 bool transpose);

void ConvertMatrixInt8ToFp32Avx(float* ptr_dst,
                                const int8_t* ptr_src,
                                uint32_t num_rows,
                                uint32_t num_columns,
                                float scale_factor,
                                bool scale,
                                bool transpose);
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov