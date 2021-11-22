// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DivisionToZeroFP16Resolver;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief: clamps eps into fp16 minimal normalized value in input_1/Maximum(input_2, eps) and input_1/Add(input_2, eps) patterns
 *
 * eps must be always nonzero to prevent from NaNs in such expressions if input_1 and input_2 simultaneously happened to be zero.
 * We should keep in such patterns eps >= fp16 minimal normalized value so that
 * CompressFloatConstants should not cast them into zero during compression into f16.
 */
class ov::pass::DivisionToZeroFP16Resolver: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DivisionToZeroFP16Resolver();
};
