// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            static inline void reduce_logical_and(const char* arg,
                                                  char* out,
                                                  const Shape& in_shape,
                                                  const AxisSet& reduction_axes)
            {
                auto out_shape = reduce(in_shape, reduction_axes, false);
                std::fill(out, out + shape_size(out_shape), 1);

                const auto in_strides = row_major_strides(in_shape);
                const auto out_strides = row_major_strides(out_shape);

                CoordinateTransformBasic input_transform(in_shape);
                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, false);

                    size_t in_idx = std::inner_product(
                        input_coord.begin(), input_coord.end(), in_strides.begin(), 0);
                    size_t out_idx = std::inner_product(
                        output_coord.begin(), output_coord.end(), out_strides.begin(), 0);

                    out[out_idx] = out[out_idx] && arg[in_idx];
                }
            }

            // DEPRECATED, can't be removed currently due to arm-plugin dependency
            static inline void reduce_logical_and(const char* arg,
                                                  char* out,
                                                  const Shape& input_shape,
                                                  const AxisSet& reduction_axes,
                                                  bool)
            {
                reduce_logical_and(arg, out, input_shape, reduction_axes);
            }

            static inline void reduce_logical_or(const char* arg,
                                                 char* out,
                                                 const Shape& in_shape,
                                                 const AxisSet& reduction_axes)
            {
                auto out_shape = reduce(in_shape, reduction_axes, false);
                std::fill(out, out + shape_size(out_shape), 0);

                const auto in_strides = row_major_strides(in_shape);
                const auto out_strides = row_major_strides(out_shape);

                CoordinateTransformBasic input_transform(in_shape);
                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, false);

                    size_t in_idx = std::inner_product(
                        input_coord.begin(), input_coord.end(), in_strides.begin(), 0);
                    size_t out_idx = std::inner_product(
                        output_coord.begin(), output_coord.end(), out_strides.begin(), 0);

                    out[out_idx] = out[out_idx] || arg[in_idx];
                }
            }

            // DEPRECATED, can't be removed currently due to arm-plugin dependency
            static inline void reduce_logical_or(const char* arg,
                                                 char* out,
                                                 const Shape& input_shape,
                                                 const AxisSet& reduction_axes,
                                                 bool)
            {
                reduce_logical_or(arg, out, input_shape, reduction_axes);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
