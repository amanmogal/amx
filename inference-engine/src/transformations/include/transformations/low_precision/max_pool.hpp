// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API MaxPoolTransformation : public LayerTransformation {
public:
    MaxPoolTransformation(const Params& params) : LayerTransformation(params) {}
    ~MaxPoolTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

#if 0  // TODO: LPT-TO-NGRAPH
    bool isBroadcastByChannels(std::shared_ptr<Node> layer) const;

    static bool isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2) noexcept;
    static bool isBroadcasted(const TensorDesc& tensorDesc) noexcept;
#endif
private:
#if 0  // TODO: LPT-TO-NGRAPH
    static int getNotEmpty(const CNNLayer& eltwise);
#endif
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
