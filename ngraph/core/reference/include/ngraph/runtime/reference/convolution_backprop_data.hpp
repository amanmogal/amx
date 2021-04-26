// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <functional>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                template <typename T>
                void extend_with_zeros(const Strides& strides,
                                       const Shape& input_shape,
                                       const T*& in,
                                       Shape& output_shape,
                                       std::vector<T>& input_zeros)
                {
                    std::vector<int> input_3d = {1, 1, 1};
                    std::vector<int> strides_3d = {1, 1, 1};
                    std::vector<int> new_input_3d = {1, 1, 1};

                    for (size_t i = 0; i < strides.size(); ++i)
                    {
                        output_shape[i + 2] =
                            input_shape[i + 2] + (strides[i] - 1) * (input_shape[i + 2] - 1);
                        input_3d[input_3d.size() - strides.size() + i] = input_shape[i + 2];
                        strides_3d[strides_3d.size() - strides.size() + i] = strides[i];
                        new_input_3d[new_input_3d.size() - strides.size() + i] =
                            output_shape[i + 2];
                    }

                    for (int i_z = 0; i_z < input_3d[0]; ++i_z)
                    {
                        for (int i_y = 0; i_y < input_3d[1]; ++i_y)
                        {
                            for (int i_x = 0; i_x < input_3d[2]; ++i_x)
                            {
                                input_zeros.push_back(
                                    in[i_x + i_y * input_3d[2] + i_z * input_3d[2] * input_3d[1]]);

                                if (i_x < input_3d[2] - 1)
                                {
                                    for (int k = 0; k < strides_3d[2] - 1; k++)
                                    {
                                        input_zeros.push_back(0);
                                    }
                                }
                            }

                            if (i_y < input_3d[1] - 1)
                            {
                                for (int y_dim = 0; y_dim < new_input_3d[2] * (strides_3d[1] - 1);
                                     y_dim++)
                                {
                                    input_zeros.push_back(0);
                                }
                            }
                        }

                        if (i_z < input_3d[0] - 1)
                        {
                            for (int y_dim = 0;
                                 y_dim < new_input_3d[1] * new_input_3d[2] * (strides_3d[0] - 1);
                                 y_dim++)
                            {
                                input_zeros.push_back(0);
                            }
                        }
                    }
                }
            } // namespace

            using namespace reference::convolution_ref;
            template <typename T>
            void convolution_backprop_impl(const T* in,
                                           const T* f,
                                           T* out,
                                           const Shape& in_shape,
                                           const Shape& f_shape,
                                           const Shape& out_shape,
                                           const Strides& strides,
                                           const Strides& dilation,
                                           const CoordinateDiff& pads_begin,
                                           const CoordinateDiff& pads_end,
                                           const CoordinateDiff& output_padding)

            {
                // this implementation supports 1D, 2D and 3D convolutions
                NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5,
                             "Unsupported input rank: ",
                             in_shape);

                NGRAPH_CHECK(f_shape.size() >= 3 && f_shape.size() <= 5,
                             "Unsupported kernel rank: ",
                             f_shape);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                ConvolutionParams params{strides, dilation, pads_begin, pads_end, output_padding};

                // here we are extending spatial dimensions to 3D, because we are going to use 3D
                // convolution implementation to convolve also in 1D & 2D case
                Shape input_shape{in_shape};
                Shape filters_shape{f_shape};
                if (in_shape.size() < 5)
                {
                    convolution_ref::extend_to_3D(params, input_shape, filters_shape);
                }

                for (size_t i = 0; i < input_shape.size() - 2; ++i)
                {
                    if (input_shape[i + 2] > 1 || filters_shape[i + 2] > 1)
                    {
                        params.pads_begin[i] = filters_shape[i + 2] - params.pads_begin[i] - 1;
                        params.pads_end[i] = filters_shape[i + 2] - params.pads_end[i] - 1;
                    }
                    else
                    {
                        params.pads_begin[i] = 0;
                        params.pads_end[i] = 0;
                    }
                }

                // convert output shape to 3D, contains only dimensions
                Shape out_shape_3d{out_shape.begin() + 2, out_shape.end()};

                int out_shape_rank = out_shape.size() - 2;
                if (out_shape_rank < 3)
                {
                    int missing_dims = 3 - out_shape_rank;
                    out_shape_3d.insert(
                        std::next(out_shape_3d.end(), -out_shape_rank), missing_dims, 1);
                }

                // modify params.pads_end when output_shape was provided in ctor in order to
                // calculate expected number of output elements
                for (size_t i = 0; i < out_shape_3d.size(); i++)
                {
                    if (out_shape_3d[i] > 1)
                    {
                        // expected_dim = (in - 1)* strides + filter - 2*padding + out_padding
                        // strides is already applied (through 0's extension in input)
                        // padding = pads_begin + pads_end, formula below is using
                        // params.pad_begin/params.pads_end:
                        size_t expected_dim = (input_shape[i + 2] - 1) - filters_shape[i + 2] +
                                              params.pads_begin[i] + params.pads_end[i] + 2 +
                                              params.output_padding[i];
                        if (out_shape_3d[i] != expected_dim)
                        {
                            params.pads_end[i] += out_shape_3d[i] - expected_dim;
                        }
                    }
                }

                const size_t filters_count = filters_shape[filter_out_ch_axis];
                const Shape filter_shape(++filters_shape.begin(), filters_shape.end());
                const size_t filter_size = shape_size(filter_shape);

                const size_t batches_count = input_shape[in_batch_axis];
                Shape batch_shape(++input_shape.begin(), input_shape.end());
                const size_t batch_size = shape_size(batch_shape);

                auto batch = in;

                for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx)
                {
                    auto filter = f;
                    for (size_t f_idx = 0; f_idx < filters_count; ++f_idx)
                    {
                        convolution_ref::convolve_3D_channels(
                            params, batch, batch_shape, filter, filter_shape, out);
                        filter += filter_size;
                    }
                    batch += batch_size;
                }
            }

            template <typename T>
            void convolution_backprop_in(const T* delta_in,
                                         const T* filter,
                                         T* delta_out,
                                         const Shape& in_shape,
                                         const Shape& filter_shape,
                                         const Shape& out_shape,
                                         const Strides& in_dilation,
                                         const Strides& filter_dilation,
                                         const CoordinateDiff& forward_in_pad_bellow,
                                         const CoordinateDiff& forward_in_pad_above,
                                         const Strides& stride,
                                         const CoordinateDiff& output_padding)
            {
                std::vector<T> extended_input;
                std::vector<T> extended_filter;
                AxisSet reverse_axes;

                Shape conv_input_shape = in_shape;
                Shape conv_filter_shape = filter_shape;
                Strides conv_stride = stride;
                Strides conv_filter_dilation = filter_dilation;
                auto conv_input_data = delta_in;

                // Note that we only reverse the spatial dimensions here (loop
                // starts at 2)
                std::vector<T> reversed(shape_size(filter_shape));
                for (size_t i = 2; i < filter_shape.size(); ++i)
                {
                    reverse_axes.insert(i);
                }
                reverse(reinterpret_cast<const char*>(filter),
                        reinterpret_cast<char*>(&reversed[0]),
                        filter_shape,
                        filter_shape,
                        reverse_axes,
                        sizeof(T));

                auto conv_filter_data = &reversed[0];

                // if channel number for output is > 1 then reverse order of filter coefficients as
                // it is required by convolve_3D_channels() function.
                if (filter_shape[1] > 1)
                {
                    std::vector<T> temp_reversed(reversed);
                    const Shape filter_dim_shape(filter_shape.begin() + 2, filter_shape.end());
                    const size_t filter_size = shape_size(filter_dim_shape);

                    for (size_t i = 0; i < filter_shape[0] * filter_shape[1]; i++)
                    {
                        auto delta = temp_reversed.begin();
                        if (i < filter_shape[0])
                        {
                            delta = delta + i * filter_shape[1] * filter_size;
                        }
                        else
                        {
                            delta = delta + filter_size +
                                    (i - filter_shape[0]) * filter_shape[1] * filter_size;
                        }

                        std::copy(delta, delta + filter_size, reversed.begin() + i * filter_size);
                    }
                }

                // swap filter batch and channels
                std::iter_swap(conv_filter_shape.begin(), conv_filter_shape.begin() + 1);

                // extend stride and filter inputs with zero padding for stride and filter_dilation
                // > 1, after that set stride and filter params to 1.
                size_t stride_dim =
                    std::accumulate(stride.begin(), stride.end(), 1, std::multiplies<size_t>());
                if (stride_dim >= 2)
                {
                    extend_with_zeros(stride, in_shape, delta_in, conv_input_shape, extended_input);
                    std::fill(conv_stride.begin(), conv_stride.end(), 1);
                    conv_input_data = &extended_input[0];
                }

                size_t dilation_dim = std::accumulate(
                    filter_dilation.begin(), filter_dilation.end(), 1, std::multiplies<size_t>());
                if (dilation_dim >= 2)
                {
                    extend_with_zeros<T>(filter_dilation,
                                         filter_shape,
                                         reinterpret_cast<const T*&>(reversed),
                                         conv_filter_shape,
                                         extended_filter);
                    std::fill(conv_filter_dilation.begin(), conv_filter_dilation.end(), 1);
                    conv_filter_data = &extended_filter[0];
                }

                convolution_backprop_impl(conv_input_data,
                                          conv_filter_data,
                                          delta_out,
                                          conv_input_shape,
                                          conv_filter_shape,
                                          out_shape,
                                          conv_stride,
                                          conv_filter_dilation,
                                          forward_in_pad_bellow,
                                          forward_in_pad_above,
                                          output_padding);
            }

            // DEPRECATED, can't be removed currently due to kmb-plugin dependency
            template <typename OUTPUT,
                      typename FILTER,
                      typename INPUT,
                      typename ACCUMULATION = typename widen<INPUT>::type>
            void convolution_backprop_in(const INPUT* delta_in,
                                         const FILTER* filter,
                                         OUTPUT* delta_out,
                                         const Shape& in_shape,
                                         const Shape& filter_shape,
                                         const Shape& out_shape,
                                         const Strides& in_dilation,
                                         const Strides& filter_dilation,
                                         const CoordinateDiff& forward_in_pad_bellow,
                                         const CoordinateDiff& forward_in_pad_above,
                                         const Strides& stride,
                                         const CoordinateDiff& output_padding)
            {
                convolution_backprop_in(delta_in,
                                        filter,
                                        delta_out,
                                        in_shape,
                                        filter_shape,
                                        out_shape,
                                        in_dilation,
                                        filter_dilation,
                                        forward_in_pad_bellow,
                                        forward_in_pad_above,
                                        stride,
                                        output_padding);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
