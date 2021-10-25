// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_exceptions.hpp>

#ifdef tensorflow_ngraph_frontend_EXPORTS
#    define TF_API NGRAPH_HELPER_DLL_EXPORT
#else
#    define TF_API NGRAPH_HELPER_DLL_IMPORT
#endif  // tensorflow_ngraph_frontend_EXPORTS

#include <ngraph/log.hpp>

namespace ov {
namespace frontend {
namespace tf {

void extract_operation_name_and_port(const std::string& port_name,
                                     std::string& operation_name,
                                     size_t& port_index,
                                     std::string& port_type);
}  // namespace tf
}  // namespace frontend
}  // namespace ov
