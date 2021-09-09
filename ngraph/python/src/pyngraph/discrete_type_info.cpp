// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyngraph/discrete_type_info.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "ngraph/type.hpp"

namespace py = pybind11;

void regclass_pyngraph_DiscreteTypeInfo(py::module m) {
    py::class_<ngraph::DiscreteTypeInfo, std::shared_ptr<ngraph::DiscreteTypeInfo>> discrete_type_info(
        m,
        "DiscreteTypeInfo");
    discrete_type_info.doc() = "ngraph.impl.DiscreteTypeInfo wraps ngraph::DiscreteTypeInfo";
    discrete_type_info.def(
        "__lt__",
        [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
            return self.version < other.version || (self.version == other.version && strcmp(self.name, other.name) < 0);
        },
        py::is_operator());
    discrete_type_info.def(
        "__le__",
        [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
            return self.version < other.version ||
                   (self.version == other.version && strcmp(self.name, other.name) <= 0);
        },
        py::is_operator());
    discrete_type_info.def(
        "__gt__",
        [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
            return self.version < other.version || (self.version == other.version && strcmp(self.name, other.name) > 0);
        },
        py::is_operator());
    discrete_type_info.def(
        "__ge__",
        [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
            return self.version < other.version ||
                   (self.version == other.version && strcmp(self.name, other.name) >= 0);
        },
        py::is_operator());
    discrete_type_info.def(
        "__eq__",
        [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
            return self.version == other.version && strcmp(self.name, other.name) == 0;
        },
        py::is_operator());
    discrete_type_info.def(
        "__ne__",
        [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
            return self.version != other.version || strcmp(self.name, other.name) != 0;
        },
        py::is_operator());
    discrete_type_info.def("__repr__", [](const ngraph::DiscreteTypeInfo& self) {
        std::string name = std::string(self.name);
        std::string version = std::to_string(self.version);
        if (self.parent != nullptr) {
            std::string parent_version = std::to_string(self.parent->version);
            std::string parent_name = self.parent->name;
            return "<DiscreteTypeInfo: " + name +" v" + version + " Parent(" + parent_name +" v"+ parent_version + ")" + ">";
        } else {
            return "<DiscreteTypeInfo: " + name +" v" + version + ">";
        }
    });

    discrete_type_info.def_property_readonly("name", [](const ngraph::DiscreteTypeInfo& self) {
        return self.name;
    });
    discrete_type_info.def_property_readonly("version", [](const ngraph::DiscreteTypeInfo& self) {
        return self.version;
    });
    discrete_type_info.def_property_readonly("parent", [](const ngraph::DiscreteTypeInfo& self) {
        return self.parent;
    });
}
