// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/matcher.hpp"  // ov::pattern::Matcher
#include "openvino/pass/graph_rewrite.hpp"  // ov::pattern::Matcher
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "ngraph/opsets/opset.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>
#include <transformations/serialize.hpp>

#include "pyopenvino/graph/passes/matcher.hpp"

namespace py = pybind11;

void regclass_graph_pattern_Matcher(py::module m) {
    py::class_<ov::pass::pattern::Matcher, std::shared_ptr<ov::pass::pattern::Matcher>> matcher(m, "Matcher");
    matcher.doc() = "openvino.impl.Matcher wraps ov::pass::pattern::Matcher";
    matcher.def(py::init([](const std::shared_ptr<ov::Node>& node,
                            const std::string& name) {
                     return std::make_shared<ov::pass::pattern::Matcher>(node, name);
                 }),
                 py::arg("node"),
                 py::arg("name"),
                 R"(
                    Create user-defined Function which is a representation of a model.

                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.

                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");

    matcher.def(py::init([](ov::Output<ov::Node> & output,
                                    const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(output, name);
                }),
                py::arg("output"),
                py::arg("name"),
                R"(
                    Create user-defined Function which is a representation of a model.

                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.

                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");

    matcher.def("get_match_root", &ov::pass::pattern::Matcher::get_match_root);
    matcher.def("get_pattern_value_map", &ov::pass::pattern::Matcher::get_pattern_value_map);
}

void regclass_graph_pattern_PassBase(py::module m) {
    py::class_<ov::pass::PassBase, std::shared_ptr<ov::pass::PassBase>> pass_base(m, "PassBase");
    pass_base.doc() = "openvino.impl.MatcherPass wraps ov::pass::MatcherPass";
}

void regclass_transformations(py::module m) {
    py::class_<ov::pass::Serialize, std::shared_ptr<ov::pass::Serialize>, ov::pass::PassBase> serialize(m, "Serialize");
    serialize.doc() = "openvino.impl.Serialize transformation";
    serialize.def(py::init([](const std::string & path_to_xml,
                              const std::string & path_to_bin) {
                                   return std::make_shared<ov::pass::Serialize>(path_to_xml, path_to_bin);
                           }));
}

// WA: to expose protected method
class PyMatcherPass: public ov::pass::MatcherPass {
public:
    using ov::pass::MatcherPass::register_matcher;
};

void regclass_graph_pattern_MatcherPass(py::module m) {
    py::class_<ov::pass::MatcherPass,
               std::shared_ptr<ov::pass::MatcherPass>,
               ov::pass::PassBase> matcher_pass(m, "MatcherPass");
    matcher_pass.doc() = "openvino.impl.MatcherPass wraps ov::pass::MatcherPass";
    matcher_pass.def(py::init<>());
    matcher_pass.def(py::init([](const std::shared_ptr<ov::pass::pattern::Matcher>& m,
                                 ov::graph_rewrite_callback callback) {
                    return std::make_shared<ov::pass::MatcherPass>(m, callback);
                }),
                py::arg("m"),
                py::arg("callback"),
                R"(
                    Create user-defined Function which is a representation of a model.

                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.

                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");
    matcher_pass.def("register_new_node", &PyMatcherPass::register_node);
    matcher_pass.def("register_matcher", &PyMatcherPass::register_matcher);
}

ov::NodeTypeInfo get_type(const std::string & type_name) {
    // TODO: allow to specify opset version
    const ngraph::OpSet& m_opset = ngraph::get_opset8();
    if (!m_opset.contains_type(type_name)) {
        throw std::runtime_error("Wrong pattern type:" +type_name + " in not in opset8");
    }
    return m_opset.create(type_name)->get_type_info();
}

std::vector<ov::NodeTypeInfo> get_types(const std::vector<std::string> & type_names) {
    std::vector<ov::NodeTypeInfo> types;
    for (const auto & type_name : type_names) {
        types.emplace_back(get_type(type_name));
    }
    return types;
}

void regclass_graph_patterns(py::module m) {
    py::class_<ov::pass::pattern::op::WrapType,
            std::shared_ptr<ov::pass::pattern::op::WrapType>,
            ov::Node> wrap_type(m, "WrapType");
    wrap_type.doc() = "openvino.impl.MatcherPass wraps ov::pass::MatcherPass";

    wrap_type.def(py::init([](std::string name) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name));
    }));

    wrap_type.def(py::init([](std::string name, const ov::Output<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](std::string name, const ov::OutputVector& inputs) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, inputs);
    }));

    wrap_type.def(py::init([](std::vector<std::string> names) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names));
    }));

    wrap_type.def(py::init([](std::vector<std::string> names, const ov::Output<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](std::vector<std::string> names, const ov::OutputVector& input_values) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, input_values);
    }));
}
