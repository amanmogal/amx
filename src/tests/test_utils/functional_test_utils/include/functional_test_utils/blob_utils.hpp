﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include "blob_factory.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/runtime/common.hpp"

namespace FuncTestUtils {
namespace Bf16TestUtils {
inline short reducePrecisionBitwiseS(const float in);
}  // namespace Bf16TestUtils

enum CompareType {
    ABS,
    REL,
    ABS_AND_REL  //  if absolute and relative differences are too high, an exception is thrown
};
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Pointer to considered blob
 * @param ref Pointer to reference blob
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparison
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template <typename dType>
inline void compareRawBuffers(const dType* res,
                              const dType* ref,
                              size_t resSize,
                              size_t refSize,
                              CompareType compareType,
                              float thr1 = 0.01,
                              float thr2 = 0.01,
                              bool printData = false) {
    if (printData) {
        std::cout << "Reference results: " << std::endl;
        for (size_t i = 0; i < refSize; i++) {
            std::cout << ref[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Test results: " << std::endl;
        for (size_t i = 0; i < resSize; i++) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
    }

    switch (compareType) {
    case CompareType::ABS:
        for (size_t i = 0; i < refSize; i++) {
            float absDiff = std::abs(res[i] - ref[i]);
            ASSERT_LE(absDiff, thr1) << "Relative comparison of values ref: " << ref[i] << " and res: " << res[i]
                                     << " , index in blobs: " << i << " failed!";
        }
        break;
    case CompareType::REL:
        for (size_t i = 0; i < refSize; i++) {
            float absDiff = std::abs(res[i] - ref[i]);
            float relDiff = absDiff / std::max(res[i], ref[i]);
            ASSERT_LE(relDiff, thr2) << "Relative comparison of values ref: " << ref[i] << " and res: " << res[i]
                                     << " , index in blobs: " << i << " failed!";
        }
        break;
    case CompareType::ABS_AND_REL:
        for (size_t i = 0; i < refSize; i++) {
            float absDiff = std::abs(res[i] - ref[i]);
            if (absDiff > thr1) {
                float relDiff = absDiff / std::max(res[i], ref[i]);
                ASSERT_LE(relDiff, thr2) << "Comparison of values ref: " << ref[i] << " and res: " << res[i]
                                         << " , index in blobs: " << i << " failed!";
            }
        }
        break;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Pointer to considered blob
 * @param ref Pointer to reference blob
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData Flag if data printing is demanded
 */
template <typename dType>
inline void compareRawBuffers(const dType* res,
                              const dType* ref,
                              size_t resSize,
                              size_t refSize,
                              float thr = 0.01,
                              bool printData = false) {
    compareRawBuffers(res, ref, resSize, refSize, CompareType::ABS_AND_REL, thr, thr, printData);
}
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparision
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template <typename dType>
inline void compareRawBuffers(const std::vector<dType*> res,
                              const std::vector<dType*> ref,
                              const std::vector<size_t>& resSizes,
                              const std::vector<size_t>& refSizes,
                              CompareType compareType,
                              float thr1 = 0.01,
                              float thr2 = 0.01,
                              bool printData = false) {
    ASSERT_TRUE(res.size() == ref.size()) << "Reference and Results vector have to be same length";
    ASSERT_TRUE(res.size() == resSizes.size()) << "Results vector and elements count vector have to be same length";
    ASSERT_TRUE(ref.size() == refSizes.size()) << "Reference vector and elements count vector have to be same length";
    for (size_t i = 0; i < res.size(); i++) {
        if (printData)
            std::cout << "BEGIN CHECK BUFFER [" << i << "]" << std::endl;
        compareRawBuffers(res[i], ref[i], resSizes[i], refSizes[i], compareType, thr1, thr2, printData);
        if (printData)
            std::cout << "END CHECK BUFFER [" << i << "]" << std::endl;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData A flag if data printing is demanded
 */
template <typename dType>
inline void compareRawBuffers(const std::vector<dType*> res,
                              const std::vector<dType*> ref,
                              const std::vector<size_t>& resSizes,
                              const std::vector<size_t>& refSizes,
                              float thr = 0.01,
                              bool printData = false) {
    compareRawBuffers(res, ref, resSizes, refSizes, CompareType::ABS_AND_REL, thr, thr, printData);
}
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparision
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template <typename dType>
inline void compareRawBuffers(const std::vector<dType*> res,
                              const std::vector<std::shared_ptr<dType*>> ref,
                              const std::vector<size_t>& resSizes,
                              const std::vector<size_t>& refSizes,
                              CompareType compareType,
                              float thr1 = 0.01,
                              float thr2 = 0.01,
                              bool printData = false) {
    ASSERT_TRUE(res.size() == ref.size()) << "Reference and Results vector have to be same length";
    ASSERT_TRUE(res.size() == resSizes.size()) << "Results vector and elements count vector have to be same length";
    ASSERT_TRUE(ref.size() == refSizes.size()) << "Reference vector and elements count vector have to be same length";
    for (size_t i = 0; i < res.size(); i++) {
        if (printData)
            std::cout << "BEGIN CHECK BUFFER [" << i << "]" << std::endl;
        compareRawBuffers(res[i], *ref[i], resSizes[i], refSizes[i], compareType, thr1, thr2, printData);
        if (printData)
            std::cout << "END CHECK BUFFER [" << i << "]" << std::endl;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData A flag if data printing is demanded
 */
template <typename dType>
inline void compareRawBuffers(const std::vector<dType*> res,
                              const std::vector<std::shared_ptr<dType*>> ref,
                              const std::vector<size_t>& resSizes,
                              const std::vector<size_t>& refSizes,
                              float thr = 0.01,
                              bool printData = false) {
    compareRawBuffers(res, ref, resSizes, refSizes, CompareType::ABS_AND_REL, thr, thr, printData);
}

inline void GetComparisonThreshold(InferenceEngine::Precision prc, float& absoluteThreshold, float& relativeThreshold) {
    switch (prc) {
    case InferenceEngine::Precision::FP32:
        absoluteThreshold = relativeThreshold = 1e-4f;
        break;
    case InferenceEngine::Precision::FP16:
        absoluteThreshold = relativeThreshold = 1e-2f;
        break;
    case InferenceEngine::Precision::I16:
    case InferenceEngine::Precision::I8:
    case InferenceEngine::Precision::U8:
        absoluteThreshold = relativeThreshold = 1;
        break;
    default:
        IE_THROW() << "Unhandled precision " << prc << " passed to the GetComparisonThreshold()";
    }
}

inline float GetComparisonThreshold(InferenceEngine::Precision prc) {
    float res;
    GetComparisonThreshold(prc, res, res);
    return res;
}

// Copy from net_pass.h
template <InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
inline void convertArrayPrecision(typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type* dst,
                                  const typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type* src,
                                  size_t nelem) {
    using dst_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = static_cast<dst_type>(src[i]);
    }
}

template <>
inline void convertArrayPrecision<InferenceEngine::Precision::BF16, InferenceEngine::Precision::FP32>(float* dst,
                                                                                                      const short* src,
                                                                                                      size_t nelem) {
    auto srcBf16 = reinterpret_cast<const ov::bfloat16*>(src);
    for (size_t i = 0; i < nelem; i++) {
        dst[i] = static_cast<float>(srcBf16[i]);
    }
}

template <InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
inline InferenceEngine::Blob::Ptr convertBlobPrecision(const InferenceEngine::Blob::Ptr& blob) {
    using from_d_type = typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type;
    using to_d_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    auto tensor_desc = blob->getTensorDesc();
    InferenceEngine::Blob::Ptr new_blob = InferenceEngine::make_shared_blob<to_d_type>(
        InferenceEngine::TensorDesc{PREC_TO, tensor_desc.getDims(), tensor_desc.getLayout()});
    new_blob->allocate();
    auto target = new_blob->buffer().as<to_d_type*>();
    auto source = blob->buffer().as<from_d_type*>();
    convertArrayPrecision<PREC_FROM, PREC_TO>(target, source, blob->size());
    return new_blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobFloatNormalDistribution(const InferenceEngine::TensorDesc& td,
                                                                           const float mean,
                                                                           const float stddev,
                                                                           const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X)                                                                      \
    case X:                                                                          \
        ov::test::utils::fill_data_normal_random_float<X>(blob, mean, stddev, seed); \
        break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobFloat(const InferenceEngine::TensorDesc& td,
                                                         const uint32_t range = 10,
                                                         const int32_t start_from = 0,
                                                         const int32_t resolution = 1,
                                                         const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);

    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X)                                                                                \
    case X:                                                                                    \
        ov::test::utils::fill_data_random_float<X>(blob, range, start_from, resolution, seed); \
        break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

template <typename T>
inline InferenceEngine::Blob::Ptr createAndFillBlobWithFloatArray(const InferenceEngine::TensorDesc& td,
                                                                  const T values[],
                                                                  const int size) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X)                                                           \
    case X:                                                               \
        ov::test::utils::fill_data_float_array<X, T>(blob, values, size); \
        break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlob(const InferenceEngine::TensorDesc& td,
                                                    const uint32_t range = 10,
                                                    const int32_t start_from = 0,
                                                    const int32_t resolution = 1,
                                                    const int seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X)                                                                          \
    case X:                                                                              \
        ov::test::utils::fill_data_random<X>(blob, range, start_from, resolution, seed); \
        break;
        CASE(InferenceEngine::Precision::FP64)
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::BF16)
        CASE(InferenceEngine::Precision::U4)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U32)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::U64)
        CASE(InferenceEngine::Precision::I4)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobConsistently(const InferenceEngine::TensorDesc& td,
                                                                const uint32_t range,
                                                                const int32_t start_from,
                                                                const int32_t resolution) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X)                                                                          \
    case X:                                                                              \
        ov::test::utils::fill_data_consistently<X>(blob, range, start_from, resolution); \
        break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobUniqueSequence(const InferenceEngine::TensorDesc& td,
                                                                  const int32_t start_from = 0,
                                                                  const int32_t resolution = 1,
                                                                  const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    auto shape = td.getDims();
    auto range = std::accumulate(begin(shape), end(shape), uint64_t(1), std::multiplies<uint64_t>()) * 2;
    switch (td.getPrecision()) {
#define CASE(X)                                                                                     \
    case X:                                                                                         \
        ov::test::utils::fill_random_unique_sequence<X>(blob, range, start_from, resolution, seed); \
        break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::I32)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

template <typename dType>
inline void fillInputsBySinValues(dType* data, size_t size) {
    if (std::is_same<dType, float>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sin(static_cast<float>(i));
        }
    } else if (std::is_same<dType, short>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(sin(static_cast<float>(i)));
        }
    }
}

inline int fillInputsBySinValues(InferenceEngine::Blob::Ptr blob) {
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        return -1;
    }
    if (mblob->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
        return -2;
    }
    auto lm = mblob->rwmap();
    fillInputsBySinValues(lm.as<float*>(), mblob->size());
    return 0;
}

namespace Bf16TestUtils {

#if defined __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-aliasing"
#    pragma GCC diagnostic ignored "-Wuninitialized"
#endif

inline float reducePrecisionBitwise(const float in) {
    float f = in;
    int* i = reinterpret_cast<int*>(&f);
    int t2 = *i & 0xFFFF0000;
    float ft1;
    memcpy(&ft1, &t2, sizeof(float));
    if ((*i & 0x8000) && (*i & 0x007F0000) != 0x007F0000) {
        t2 += 0x10000;
        memcpy(&ft1, &t2, sizeof(float));
    }
    return ft1;
}

inline short reducePrecisionBitwiseS(const float in) {
    float f = reducePrecisionBitwise(in);
    int intf = *reinterpret_cast<int*>(&f);
    intf = intf >> 16;
    short s = intf;
    return s;
}

#if defined __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace Bf16TestUtils

enum class BlobType {
    Memory,
    Compound,
    Remote,
};

inline std::ostream& operator<<(std::ostream& os, BlobType type) {
    switch (type) {
    case BlobType::Memory:
        return os << "Memory";
    case BlobType::Remote:
        return os << "Remote";
    default:
        IE_THROW() << "Not supported blob type";
    }
}

inline bool checkLayout(InferenceEngine::Layout layout, const std::vector<size_t>& inputShapes) {
    bool check = false;
    switch (layout) {
    case InferenceEngine::Layout::SCALAR:
        check = inputShapes.size() == 0;
        break;
    case InferenceEngine::Layout::C:
        check = 1 == inputShapes.size();
        break;
    case InferenceEngine::Layout::BLOCKED:
    case InferenceEngine::Layout::ANY:
        check = true;
        break;
    case InferenceEngine::Layout::GOIDHW:
        check = 6 == inputShapes.size();
        break;
    case InferenceEngine::Layout::NCDHW:
    case InferenceEngine::Layout::NDHWC:
    case InferenceEngine::Layout::OIDHW:
    case InferenceEngine::Layout::GOIHW:
        check = 5 == inputShapes.size();
        break;
    case InferenceEngine::Layout::OIHW:
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NHWC:
        check = 4 == inputShapes.size();
        break;
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::HWC:
        check = 3 == inputShapes.size();
        break;
    case InferenceEngine::Layout::CN:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::HW:
        check = 2 == inputShapes.size();
        break;
    default:
        break;
    }
    return check;
}
}  // namespace FuncTestUtils
