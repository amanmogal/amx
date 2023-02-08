// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/unsupported_constant.hpp"
#include "openvino/opsets/opset8.hpp"

#include "openvino/core/type/non_tensor_type.hpp"
#include "openvino/op/str_ops.hpp"


using namespace std;
using namespace ov::opset8;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_const_op(const NodeContext& node) {
    try {
        auto ov_type = node.get_attribute_as_any("dtype");
        if(!ov_type.is<element::Type>()) {
            // FIXME: kind of a work around, the next line should raise StructuralTypeWA with necessary data encapsulated
            node.get_attribute<ov::Tensor>("value");
            std::cerr << "----------------------------- CANNOT BE ------------------------------\n";
        }
        std::shared_ptr<Node> const_node;
        if (ov_type == element::undefined) {
            const_node = std::make_shared<UnsupportedConstant>();
        } else {
            auto tensor = node.get_attribute<Tensor>("value");
            const_node = std::make_shared<Constant>(tensor.get_element_type(), tensor.get_shape(), tensor.data());
        }
        set_node_name(node.get_name(), const_node);
        return {const_node};
    } catch(const StructuralTypeWA& str_wa) {
        std::cerr << "[ STR WA CATCH ]";
        if(!str_wa.m_structural_type.is<ov::element::StructuralType::Str>()) {
            std::cerr << "This is not a string\n";
            throw;
        }
        auto& tensor = str_wa.m_tensor.as<ov::Tensor>();
        // FIXME: Is this a data copy?
        auto res = std::make_shared<ov::opset8::Constant>(tensor.get_element_type(), tensor.get_shape(), tensor.data());
        set_node_name(node.get_name(), res);
        return {std::make_shared<StructPack>(OutputVector{res},
            str_wa.m_structural_type,
            PartialShape{})};
    } catch(...) {
        std::cerr << "[ ERROR ] Cannot decode ov::Tensor from node with name " << node.get_name() << "\n";
        throw;
    }

}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
