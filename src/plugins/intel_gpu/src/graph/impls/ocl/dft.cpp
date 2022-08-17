// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dft/dft_kernel_ref.h>
#include <dft/dft_kernel_selector.h>
#include <dft_inst.h>
#include <kernel_selector_helper.h>

#include <impls/implementation_map.hpp>
#include <intel_gpu/runtime/error_handler.hpp>

#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct dft_impl : typed_primitive_impl_ocl<dft> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<dft_impl>(*this);
    }

    static primitive_impl* create(const dft_node& arg) {
        auto params = get_default_params<kernel_selector::dft_params>(arg);
        auto primitive = arg.get_primitive();
        params.axes = primitive->axes;
        if (primitive->kind == dft_kind::inverse) {
            params.kind = kernel_selector::dft_params::inverse;
        }
        auto optional_params = get_default_optional_params<kernel_selector::dft_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::dft_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new dft_impl{arg, best_kernels.front()};
    }
};

namespace detail {

attach_dft_impl::attach_dft_impl() {
    implementation_map<dft>::add(impl_types::ocl, dft_impl::create, {
        MAKE_TUPLE2(bfyx,   f32, f16),
        MAKE_TUPLE2(bfzyx,  f32, f16),
        MAKE_TUPLE2(bfwzyx, f32, f16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
