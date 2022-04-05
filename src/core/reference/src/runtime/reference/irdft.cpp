//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/runtime/reference/irdft.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <ngraph/runtime/reference/utils/fft_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph {
namespace runtime {
namespace reference {
namespace {
using complex_type = std::complex<float>;

// When we reverted shape, we need to revert IRDFT axes.
std::vector<int64_t> reverse_fft_axes(const std::vector<int64_t>& axes, int64_t complex_data_rank) {
    auto result = axes;
    for (int64_t& axis : result) {
        axis = complex_data_rank - 1 - axis;
    }
    return result;
}

// Helper function to get only length with respect to given axes.
std::vector<int64_t> get_lengths(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes) {
    std::vector<int64_t> lengths;
    for (int64_t axis : axes) {
        lengths.push_back(shape[axis]);
    }
    return lengths;
}

// This function calculates 'outer axes', that is axes that are not transformed by IRDFT.
std::vector<int64_t> get_outer_axes(const std::vector<int64_t>& inner_axes, int64_t complex_data_rank) {
    int64_t num_of_inner_axes = static_cast<int64_t>(inner_axes.size());
    int64_t num_of_outer_axes = complex_data_rank - num_of_inner_axes;

    std::vector<int64_t> outer_axes(num_of_outer_axes);

    int64_t fft_axes_as_bitset = 0;
    for (int64_t axis : inner_axes) {
        assert(axis < 64);
        fft_axes_as_bitset |= static_cast<int64_t>(1) << axis;
    }

    for (int64_t j = 0, i = 0; i < complex_data_rank; ++i) {
        if ((fft_axes_as_bitset & (static_cast<int64_t>(1) << i)) == 0) {
            outer_axes[j] = i;
            ++j;
        }
    }

    return outer_axes;
}

// This function gets a complex value from given coords of this value
complex_type get_value_from_input(const complex_type* input_data,
                                  int64_t src_index,
                                  const std::vector<int64_t>& coords,
                                  const std::vector<int64_t>& input_fft_lengths,
                                  const std::vector<int64_t>& input_fft_strides) {
    int64_t offset = 0;
    int64_t num_of_fft_axes = static_cast<int64_t>(coords.size());
    for (int64_t i = 0; i < num_of_fft_axes; ++i) {
        int64_t coord = coords[i];
        if (coord >= input_fft_lengths[i]) {
            return complex_type{0.0f, 0.0f};
        }
        offset += coord * input_fft_strides[i];
    }

    return input_data[src_index + offset];
}

// Copying input data to the given memory domain. Returns true if the copied blob is zero, and false otherwise.
bool copy_data_from_input_and_check_is_blob_zero(complex_type* result,
                                                 const complex_type* input_data,
                                                 int64_t src_index,
                                                 int64_t fft_size,
                                                 const std::vector<int64_t>& fft_strides,
                                                 const std::vector<int64_t>& input_fft_lengths,
                                                 const std::vector<int64_t>& input_fft_strides,
                                                 int64_t last_axis_upper_bound) {
    bool blob_is_zero = true;
    for (int64_t idx = 0; idx < fft_size; ++idx) {
        auto coords = fft_common::coords_from_index(idx, fft_strides);
        if (coords.back() >= last_axis_upper_bound) {
            continue;
        }
        complex_type value = get_value_from_input(input_data, src_index, coords, input_fft_lengths, input_fft_strides);
        result[idx] = value;
        blob_is_zero = blob_is_zero && (value == complex_type{0.0f, 0.0f});
    }
    return blob_is_zero;
}

template <typename T>
void print_vector(const std::vector<T>& v, const std::string& prefix) {
    std::cout << prefix;
    for (const auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

inline bool is_power_of_two(int64_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

// This function calculates internal FFT buffer size using lengths of FFT axes.
int64_t compute_buffer_size(const std::vector<int64_t>& fft_lengths) {
    int64_t buffer_size = 0;

    for (int64_t length : fft_lengths) {
        int64_t current_size = is_power_of_two(length) ? (2 * length) : length;
        buffer_size = std::max(buffer_size, current_size);
    }

    return buffer_size;
}

// Calculation of IRDFT
void irdft_calculation(const float* input_data,
                       const Shape& input_data_shape,
                       const std::vector<int64_t>& axes_data,
                       float* fft_result,
                       const Shape& fft_output_shape) {
    std::cout << "We are in the function irdft_calculation()...\n";
    std::cout << "input_data_shape: " << input_data_shape << "\n";
    std::cout << "fft_output_shape: " << fft_output_shape << "\n";

    const complex_type* complex_input_data_ptr = reinterpret_cast<const complex_type*>(input_data);
    complex_type* complex_output_ptr = reinterpret_cast<complex_type*>(fft_result);
    std::cout << "input_data pointer:  " << complex_input_data_ptr << "\n";
    std::cout << "output data pointer: " << complex_output_ptr << "\n";

    print_vector(axes_data, "axes_data: ");

    const int64_t complex_data_rank = static_cast<int64_t>(input_data_shape.size()) - 1;
    std::cout << "complex_data_rank: " << complex_data_rank << "\n";
    const auto fft_axes = reverse_fft_axes(axes_data, complex_data_rank);
    print_vector(fft_axes, "fft_axes: ");

    const auto reversed_output_shape = fft_common::reverse_shape_of_emulated_complex_tensor(fft_output_shape);
    print_vector(reversed_output_shape, "reversed_output_shape: ");

    const int64_t fft_rank = fft_axes.size();
    std::cout << "fft_rank: " << fft_rank << "\n";

    const auto fft_lengths = get_lengths(reversed_output_shape, fft_axes);
    print_vector(fft_lengths, "fft_lengths: ");

    const auto fft_strides = fft_common::compute_strides(fft_lengths);
    print_vector(fft_strides, "fft_strides: ");

    const int64_t fft_size = fft_strides[fft_rank];
    std::cout << "fft_size: " << fft_size << "\n";

    if (fft_size <= 0) {
        return;
    }

    const int64_t buffer_size = compute_buffer_size(fft_lengths);
    std::cout << "buffer_size: " << buffer_size << "\n";

    std::vector<complex_type> data(fft_size);
    std::vector<complex_type> buffer(buffer_size);

    const auto outer_axes = get_outer_axes(fft_axes, complex_data_rank);
    print_vector(outer_axes, "outer_axes: ");

    const int64_t outer_rank = outer_axes.size();
    std::cout << "outer_rank: " << outer_rank << "\n";

    const auto outer_lengths = get_lengths(reversed_output_shape, outer_axes);
    print_vector(outer_lengths, "outer_lengths: ");

    const auto outer_strides = fft_common::compute_strides(outer_lengths);
    print_vector(outer_strides, "outer_strides: ");

    const int64_t outer_size = outer_strides[outer_rank];
    std::cout << "outer_size: " << outer_size << "\n";

    const auto output_strides = fft_common::compute_strides(reversed_output_shape);
    const auto output_fft_strides = get_lengths(output_strides, fft_axes);
    const auto output_outer_strides = get_lengths(output_strides, outer_axes);
    print_vector(output_strides, "output_strides: ");
    print_vector(output_fft_strides, "output_fft_strides: ");
    print_vector(output_outer_strides, "output_outer_strides: ");

    const auto reversed_input_shape = fft_common::reverse_shape_of_emulated_complex_tensor(input_data_shape);
    const auto input_fft_lengths = get_lengths(reversed_input_shape, fft_axes);
    const auto input_strides = fft_common::compute_strides(reversed_input_shape);
    const auto input_fft_strides = get_lengths(input_strides, fft_axes);
    const auto input_outer_strides = get_lengths(input_strides, outer_axes);
    print_vector(reversed_input_shape, "reversed_input_shape: ");
    print_vector(input_fft_lengths, "input_fft_lengths: ");
    print_vector(input_strides, "input_strides: ");
    print_vector(input_fft_strides, "input_fft_strides: ");
    print_vector(input_outer_strides, "input_outer_strides: ");

    const int64_t last_axis_upper_bound = fft_lengths.back() / 2 + 1;
    std::cout << "last_axis_upper_bound: " << last_axis_upper_bound << "\n";
    // Loop along with 'outer' dimensions, that is along with
    // not transformed dimensions.
    for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        const auto outer_coords = fft_common::coords_from_index(outer_idx, outer_strides);
        int64_t outer_input_offset = fft_common::offset_from_coords_and_strides(outer_coords, input_outer_strides);
        print_vector(outer_coords, "outer_coords: ");
        std::cout << "outer_input_offset: " << outer_input_offset << "\n";

        // Copying current data to transform
        bool blob_is_zero = copy_data_from_input_and_check_is_blob_zero(data.data(),
                                                                        complex_input_data_ptr,
                                                                        outer_input_offset,
                                                                        fft_size,
                                                                        fft_strides,
                                                                        input_fft_lengths,
                                                                        input_fft_strides,
                                                                        last_axis_upper_bound);
        if (!blob_is_zero) {
        }
    }
}

void irdft_postprocessing(const complex_type* intermediate_results,
                          float* results,
                          const Shape& output_shape) {
    const size_t output_size = shape_size(output_shape);
    for (size_t i = 0; i < output_size; ++i) {
        results[i] = std::real(intermediate_results[i]);
    }
}
}  // namespace

void irdft(const float* input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           float* fft_result,
           const Shape& fft_output_shape,
           const Shape& output_shape) {
    std::vector<complex_type> intermediate_results(shape_size(fft_output_shape) / 2);
    irdft_calculation(input_data,
                      input_data_shape,
                      axes_data,
                      reinterpret_cast<float*>(intermediate_results.data()),
                      fft_output_shape);
    irdft_postprocessing(intermediate_results.data(), fft_result, output_shape);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
