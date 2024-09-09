#include "common_op_table.hpp"
#include "openvino/op/scatter_nd_update.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_tensor_scatter_add_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorScatterAdd"});
    auto data = node.get_input(0);
    auto indices = node.get_input(1);
    auto updates = node.get_input(2);
    // // optional name attribute
    // auto name = node.get_attribute<std::string>("name", "");
    // auto node_name = name.empty() ? node.get_name() : name;
    auto reduction = ov::op::v15::ScatterNDUpdate::Reduction::SUM;
    auto scatter_add_op = make_shared<ov::op::v15::ScatterNDUpdate>(data, indices, updates, reduction);
    set_node_name(node.get_name(), scatter_add_op);

    return {scatter_add_op};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
