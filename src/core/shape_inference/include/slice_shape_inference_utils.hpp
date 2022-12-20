// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/op/constant.hpp>

#include "sequnce_generator.hpp"
#include "utils.hpp"

namespace ov {
namespace internal {
/**
 * \brief Check if value of type T has got maximum value of type U.
 *
 * \tparam T     Input value type
 * \tparam U     Type to get its minimum for comparision. Default same as T.
 *
 * \param value  Input value.
 *
 * \return       True if input value has got maximum value of type U otherwise false.
 */
template <class T, class U = T>
constexpr bool is_max(const T& value) {
    return std::numeric_limits<U>::max() == value;
}

/**
 * \brief Check if value of type T has got minimum value of type U.
 *
 * \tparam T     Input value type.
 * \tparam U     Type to get its minimum for comparision. Default same as T.
 *
 * \param value  Input value.
 *
 * \return       True if input value has got minimum value of type U otherwise false.
 */
template <class T, class U = T>
constexpr bool is_min(const T& value) {
    return std::numeric_limits<U>::min() == value;
}
}  // namespace internal

namespace element {
/**
 * \brief  Check if value has got maximum value of ov::element::Type_t
 *
 * \tparam T     Input value type.
 *
 * \param type   ov::element type to get its maximum.
 * \param value  Input value for check.
 *
 * \return True if input value has got maximum number specified by ov::element type otherwise false.
 */
template <class T>
bool is_max_of(const element::Type_t& type, const T& value) {
    switch (type) {
    case element::i32:
        return internal::is_max<T, typename element_type_traits<element::i32>::value_type>(value);
    case element::i64:
        return internal::is_max<T, typename element_type_traits<element::i64>::value_type>(value);
    default:
        return false;
    }
}

/**
 * \brief  Check if value has got minimum value of ov::element::Type_t
 *
 * \tparam T     Input value type.
 *
 * \param type   ov::element type to get its minimum.
 * \param value  Input value for check.
 *
 * \return True if input value has got minimum number specified by ov::element type otherwise false.
 */
template <class T>
bool is_min_of(const element::Type_t type, const T& value) {
    switch (type) {
    case element::i32:
        return internal::is_min<T, typename element_type_traits<element::i32>::value_type>(value);
    case element::i64:
        return internal::is_min<T, typename element_type_traits<element::i64>::value_type>(value);
    default:
        return false;
    }
}

/**
 * \brief  Checks input value for element type maximum or minimum and return limit or value.
 *
 * \tparam T     Type of input value.
 * \tparam U     Type of return value. Default same as T.
 *
 * \param type   Type of ov::element::Type_t
 * \param value  Input value for check.
 *
 * \return If value is maximum or minimum get limit of U otherwise value as U.
 */
template <class T, class U = T>
U get_value_or_limit_of(const element::Type_t& type, const T& value) {
    if (is_min_of(type, value)) {
        return std::numeric_limits<U>::min();
    } else if (is_max_of(type, value)) {
        return std::numeric_limits<U>::max();
    } else {
        return static_cast<U>(value);
    }
}

}  // namespace element

namespace op {
namespace slice {

/**
 * \brief Get sliced value in step for given dimension value and start, stop, step.
 *
 * \note This function cannot be use for step 0 (division by 0)
 *
 * \param dim    Dimension value.
 * \param start  Start of slice.
 * \param stop   Stop of slice.
 * \param step   Step of slice.
 *
 * \return -1 for infinite number otherwise [0..int64_max] for finit step.
 */
inline int64_t get_sliced_value(const int64_t& dim, const int64_t& start, const int64_t& stop, const int64_t& step) {
    const auto is_reverse_step = step < 0;

    constexpr int64_t min_bound = 0;
    constexpr int64_t inf_bound = -1;

    const auto& norm_dim = dim == inf_bound ? std::numeric_limits<int64_t>::max() : dim;
    const auto is_norm_dim_max = ov::internal::is_max(norm_dim);
    const int64_t lower_max = is_reverse_step ? norm_dim - 1 : norm_dim;
    const int64_t upper_min = is_reverse_step ? inf_bound : min_bound;

    const auto is_start_lt_min_bound = start < min_bound;
    const auto are_bounds_diff_sign = is_start_lt_min_bound != (stop < 0);

    const auto is_start_max = ov::internal::is_max(start);
    const auto is_start_limit = is_start_max || ov::internal::is_min(start);
    const auto any_bound_max = is_start_max || ov::internal::is_max(stop);
    // Prepare bounds for number of elements calculation.
    int64_t lb, ub;
    if (is_norm_dim_max && (are_bounds_diff_sign || any_bound_max || is_start_limit)) {
        if (is_reverse_step) {
            ub = (is_start_lt_min_bound || any_bound_max) ? inf_bound : inf_bound - start;
        } else {
            ub = (is_start_lt_min_bound && !is_start_limit) ? stop : inf_bound;
        }
        lb = min_bound;
    } else {
        lb = clip(normalize(start, norm_dim), min_bound, lower_max);
        ub = clip(normalize(stop, norm_dim), upper_min, norm_dim);
    }

    // Calculate elements in step from bounds and step.
    if (is_norm_dim_max && lb == min_bound && ub == inf_bound) {
        return inf_bound;
    } else {
        // Limit elements count to not-positive for negative step or not-negative for positive step
        auto sliced_value =
            is_reverse_step ? std::min<int64_t>(min_bound, (ub - lb)) : std::max<int64_t>(min_bound, (ub - lb));

        if (step == -1) {
            // Elements count is negative for negative step return opposite
            sliced_value = -sliced_value;
        } else if (sliced_value != 0 && step != 1) {
            // Need to calculate elements in step. Depends on step direction reduce number element
            // in order to calculate elements in steps in one-step division (no modulo required)
            is_reverse_step ? ++sliced_value : --sliced_value;
            sliced_value /= step;
            ++sliced_value;
        } else {
            // There is no need for calculations as number of elements is 0 or step is 1.
        }
        return sliced_value;
    }
}

using Bounds = std::pair<int64_t, int64_t>;  //!< Alias to dimension bounds for slice.

/**
 * \brief Get the input bounds from constant input (constant map) or evaluate bunds
 *  and return them as vector of pairs (lower, upper).
 *
 * \tparam TShape        Shape type.
 *
 * \param op             Operator pointer.
 * \param idx            Input index.
 * \param constant_data  Map with constant data.
 *
 * \return Return vector of slice::Bounds.
 */
template <class TShape, class TResult = std::vector<Bounds>>
std::unique_ptr<TResult> get_input_bounds(const ov::Node* op,
                                          size_t idx,
                                          const std::map<size_t, HostTensorPtr>& constant_data) {
    const auto& input_et = op->get_input_element_type(idx);
    // Helper to create TResult from lowers and uppers.
    const auto make_bounds_vec = [&input_et](const std::vector<int64_t>& lowers, const std::vector<int64_t>& uppers) {
        TResult out;
        out.reserve(lowers.size());
        std::transform(lowers.begin(),
                       lowers.end(),
                       uppers.begin(),
                       std::back_inserter(out),
                       [&input_et](int64_t lb, int64_t ub) {
                           return std::make_pair(element::get_value_or_limit_of(input_et, lb),
                                                 element::get_value_or_limit_of(input_et, ub));
                       });
        return out;
    };

    std::unique_ptr<TResult> out;
    if (auto lowers = op::get_input_const_data_as<TShape, int64_t>(op, idx, constant_data)) {
        out.reset(new TResult(make_bounds_vec(*lowers, *lowers)));
    } else {
        auto bounds = ngraph::evaluate_both_bounds(op->get_input_source_output(idx));
        if (bounds.first && bounds.second) {
            auto lowers = std::make_shared<op::v0::Constant>(bounds.first)->cast_vector<int64_t>();
            auto uppers = std::make_shared<op::v0::Constant>(bounds.second)->cast_vector<int64_t>();
            out.reset(new TResult(make_bounds_vec(lowers, uppers)));
        }
    }
    return out;
}

/**
 * \brief Make sliced dimension for input dimension by step from start to stop bounds.
 *
 * \tparam TDim   Type of in/out dimension.
 *
 * \param dim
 * \param start  Slice start bounds.
 * \param stop   Slice stop bounds.
 * \param step   Slice step.
 *
 * \return Dimension with upper/lower values set according slice inputs.
 */
template <class TDim>
TDim make_dim(const TDim& dim, const Bounds& start, const Bounds& stop, int64_t step) {
    auto lb = get_sliced_value(dim.get_min_length(), start.second, stop.first, step);
    auto ub = get_sliced_value(dim.get_max_length(), start.first, stop.second, step);

    return {lb, ub};
}
}  // namespace slice
}  // namespace op
}  // namespace ov
