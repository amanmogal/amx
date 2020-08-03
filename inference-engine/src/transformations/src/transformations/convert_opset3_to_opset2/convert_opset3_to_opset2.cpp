// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp"

#include "transformations/convert_opset3_to_opset2/convert_broadcast3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_nms3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_shapeof3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_shuffle_channels3.hpp"
#include "transformations/convert_opset3_to_opset2/convert_topk3.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

bool ngraph::pass::ConvertOpSet3ToOpSet2::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;

    manager.register_pass<ngraph::pass::ConvertBroadcast3>();
    manager.register_pass<ngraph::pass::ConvertNMS3>();
    manager.register_pass<ngraph::pass::ConvertShapeOf3>();
    manager.register_pass<ngraph::pass::ConvertShuffleChannels3>();
    manager.register_pass<ngraph::pass::ConvertTopK3>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
