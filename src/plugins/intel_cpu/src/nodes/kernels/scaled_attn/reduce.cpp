// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
    #include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "common.hpp"
#include "reduce.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template<typename T>
void attn_reduce_inner(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i+= vec_len_f32_avx512) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            //auto* temp = &m_temp.at({ithr, b, pq, h, 0});
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm256_set1_ps(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = mm256_uni_loadu_ps(src);
            result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        mm256_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#endif
    for (; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}

void attn_reduce(void* dst, float* temp, size_t M, size_t S, size_t temp_stride, ov::element::Type input_precision) {
    if (input_precision == ov::element::f32) {
        auto dst_ptr = static_cast<float*>(dst);
        attn_reduce_inner(dst_ptr, temp, M, S, temp_stride);
    } else {
        assert(input_precision == ov::element::bf16);
        auto dst_ptr = static_cast<ov::bfloat16*>(dst);
        attn_reduce_inner(dst_ptr, temp, M, S, temp_stride);
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine