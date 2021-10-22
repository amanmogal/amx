// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <frontend_manager/frontend_exceptions.hpp>
#include <ngraph/file_util.hpp>

#include "place.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

NGRAPH_SUPPRESS_DEPRECATED_START

InputModelONNX::InputModelONNX(const std::string& path)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(path)} {}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModelONNX::InputModelONNX(const std::wstring& path)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(path)} {}
#endif

InputModelONNX::InputModelONNX(std::istream& model_stream)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(model_stream)} {}

InputModelONNX::InputModelONNX(std::istream& model_stream, const std::string& path)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(model_stream, path)} {}

InputModelONNX::InputModelONNX(std::istream& model_stream, const std::wstring& path)
    : InputModelONNX(model_stream, file_util::wstring_to_string(path)) {}

std::vector<Place::Ptr> InputModelONNX::get_inputs() const {
    const auto& inputs = m_editor->model_inputs();
    std::vector<Place::Ptr> in_places;
    in_places.reserve(inputs.size());
    for (const auto& input : inputs) {
        in_places.push_back(std::make_shared<PlaceTensorONNX>(input, m_editor));
    }
    return in_places;
}

std::vector<Place::Ptr> InputModelONNX::get_outputs() const {
    const auto& outputs = m_editor->model_outputs();
    std::vector<Place::Ptr> out_places;
    out_places.reserve(outputs.size());
    for (const auto& output : outputs) {
        out_places.push_back(std::make_shared<PlaceTensorONNX>(output, m_editor));
    }
    return out_places;
}

Place::Ptr InputModelONNX::get_place_by_tensor_name(const std::string& tensor_name) const {
    if (m_editor->is_correct_tensor_name(tensor_name)) {
        return std::make_shared<PlaceTensorONNX>(tensor_name, m_editor);
    }
    return nullptr;
}

Place::Ptr InputModelONNX::get_place_by_operation_name(const std::string& operation_name) const {
    if (m_editor->is_correct_and_unambiguous_node(operation_name)) {
        return std::make_shared<PlaceOpONNX>(onnx_editor::EditorNode{operation_name}, m_editor);
    }
    return nullptr;
}

Place::Ptr InputModelONNX::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                      int input_port_index) {
    const auto op = get_place_by_operation_name(operation_name);
    if (op != nullptr) {
        return op->get_input_port(input_port_index);
    }
    return nullptr;
}

Place::Ptr InputModelONNX::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                       int output_port_index) {
    const auto op = get_place_by_operation_name(operation_name);
    if (op != nullptr) {
        return op->get_output_port(output_port_index);
    }
    return nullptr;
}

void InputModelONNX::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) {
    std::map<std::string, ngraph::PartialShape> m;
    m[place->get_names()[0]] = shape;
    m_editor->set_input_shapes(m);
}

ngraph::PartialShape InputModelONNX::get_partial_shape(Place::Ptr place) const {
    return m_editor->get_tensor_shape(place->get_names().at(0));
}

void InputModelONNX::set_element_type(Place::Ptr place, const ngraph::element::Type& type) {
    std::map<std::string, ngraph::element::Type_t> m;
    m[place->get_names()[0]] = type;
    m_editor->set_input_types(m);
}

std::shared_ptr<Function> InputModelONNX::decode() {
    return m_editor->decode();
}

std::shared_ptr<Function> InputModelONNX::convert() {
    return m_editor->get_function();
}

// Editor features
void InputModelONNX::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    extract_subgraph({}, outputs);
    NGRAPH_CHECK(m_editor->model_outputs().size() == outputs.size(),
                 "Unexpected number of outputs after override_all_outputs");
    NGRAPH_CHECK(std::all_of(std::begin(outputs),
                             std::end(outputs),
                             [](const Place::Ptr& place) {
                                 return place->is_output();
                             }),
                 "Not all provided arguments of override_all_outputs are new outputs of the model");
}

void InputModelONNX::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    const auto outputs_before_extraction = m_editor->model_outputs();
    extract_subgraph({inputs}, {});
    NGRAPH_CHECK(std::equal(std::begin(outputs_before_extraction),
                            std::end(outputs_before_extraction),
                            std::begin(m_editor->model_outputs())),
                 "All outputs should be preserved after override_all_inputs. Provided inputs does "
                 "not satisfy all outputs");
    NGRAPH_CHECK(m_editor->model_inputs().size() == inputs.size(),
                 "Unexpected number of inputs after override_all_inputs");
}

void InputModelONNX::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    std::vector<onnx_editor::InputEdge> onnx_inputs;
    onnx_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (const auto input_port = std::dynamic_pointer_cast<PlaceInputEdgeONNX>(input)) {
            onnx_inputs.push_back(input_port->get_input_edge());
        } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(input)) {
            auto name = tensor->get_names()[0];
            const auto consumers = m_editor->find_output_consumers(name);
            std::transform(std::begin(consumers),
                           std::end(consumers),
                           std::back_inserter(onnx_inputs),
                           [](const onnx_editor::InputEdge& edge) {
                               return edge;
                           });
        } else if (const auto op = std::dynamic_pointer_cast<PlaceOpONNX>(input)) {
            const auto editor_node = op->get_editor_node();
            const auto op_inputs = m_editor->get_input_ports(editor_node);
            int node_idx = m_editor->get_node_index(editor_node);
            int port_idx = 0;
            std::transform(std::begin(op_inputs),
                           std::end(op_inputs),
                           std::back_inserter(onnx_inputs),
                           [&node_idx, &port_idx](const std::string&) {
                               return onnx_editor::InputEdge{node_idx, port_idx++};
                           });
        }
    }

    std::vector<onnx_editor::OutputEdge> onnx_outputs;
    onnx_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        if (const auto output_port = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(output)) {
            onnx_outputs.push_back(output_port->get_output_edge());
        } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(output)) {
            const auto output_port = tensor->get_producing_port();
            const auto onnx_output_edge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(output_port);
            NGRAPH_CHECK(onnx_output_edge, "Non-onnx output place was passed as extraction subgraph argument");
            onnx_outputs.push_back(onnx_output_edge->get_output_edge());
        } else if (const auto op = std::dynamic_pointer_cast<PlaceOpONNX>(output)) {
            const auto editor_node = op->get_editor_node();
            const auto op_outputs = m_editor->get_output_ports(editor_node);
            int node_idx = m_editor->get_node_index(editor_node);
            int port_idx = 0;
            std::transform(std::begin(op_outputs),
                           std::end(op_outputs),
                           std::back_inserter(onnx_outputs),
                           [&node_idx, &port_idx](const std::string&) {
                               return onnx_editor::OutputEdge{node_idx, port_idx++};
                           });
        }
    }
    m_editor->cut_graph_fragment(onnx_inputs, onnx_outputs);
    m_editor->serialize("extracted_subgraph.onnx");
}

Place::Ptr InputModelONNX::add_output(Place::Ptr place) {
    std::string name = place->get_names().at(0);

    const auto& outputs = m_editor->model_outputs();
    const auto& inputs = m_editor->model_inputs();

    auto find_output = std::find(std::begin(outputs), std::end(outputs), name);
    auto find_input = std::find(std::begin(inputs), std::end(inputs), name);

    if (find_input != inputs.end()) {
        return nullptr;
    }

    const auto output_port = place->get_producing_port();

    if (find_output != outputs.end()) {
        return place;
    } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(place)) {
        auto tensor_name = tensor->get_names()[0];
        auto output_edge = m_editor->find_output_edge(tensor_name);
        m_editor->add_output(output_edge);
    } else if (const auto onnx_output_edge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(output_port)) {
        NGRAPH_CHECK(onnx_output_edge, "Non-onnx output place was passed.");
        m_editor->add_output(onnx_output_edge->get_output_edge());
    } else {
        return nullptr;
    }

    return std::make_shared<PlaceTensorONNX>(name, m_editor);
}

void InputModelONNX::remove_output(Place::Ptr place) {
    std::string name = place->get_names().at(0);
    std::vector<Place::Ptr> outputs = get_outputs();
    const auto& output_names = m_editor->model_outputs();

    auto find_output = std::find(output_names.begin(), output_names.end(), name);

    if (find_output != output_names.end()) {
        outputs.erase(std::remove_if(outputs.begin(),
                                     outputs.end(),
                                     [&place](Place::Ptr const& output) {
                                         return output->is_equal(place);
                                     }),
                      outputs.end());

        extract_subgraph({}, {outputs});
    }
}

void InputModelONNX::cut_and_add_new_input(Place::Ptr place, const std::string& new_name_optional) {
    //std::string name = place->get_names().at(0);
    std::vector<Place::Ptr> inputs = get_inputs();
    std::vector<Place::Ptr> outputs = get_outputs();
    const auto& input_names = m_editor->model_inputs();
    std::vector<Place::Ptr> place_vector;

    for(auto & input : inputs) {
        std::cout << input->get_names().at(0) << std::endl;
    }

    //auto find_output = std::find(input_names.begin(), input_names.end(), name);

    //if (find_output == input_names.end()) {
        std::cout << "editing subgraph" << std::endl;
        place_vector.push_back(place);
        /*std::cout << "looking for non input." << std::endl;
        inputs.erase(std::remove_if(inputs.begin(),
                                     inputs.end(),
                                     [&place](Place::Ptr const& input) {
                                         return place->is_equal_data(input);
                                     }),
                      inputs.end());
        for (auto & input : inputs) {
            std::cout << input->get_names().at(0) << std::endl;
        }*/

        /*extract_subgraph({place_vector}, {outputs});

        inputs = get_inputs();
        for(auto & input : inputs) {
            std::cout << "input name: " << input->get_names().at(0) << std::endl;
        }*/
    //}

    m_editor->serialize("testing_subgraph.onnx");
}

void InputModelONNX::set_tensor_value(Place::Ptr place, const void* value) {
    std::cout << "editing set_tensor_value" << std::endl;
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> map;

    if (const auto var_place = std::dynamic_pointer_cast<PlaceTensorONNX>(place)) {
        std::cout << "found  PlaceTensorONNX" << std::endl;
        auto name = place->get_names().at(0);
        auto p_shape = m_editor->get_tensor_shape(name);
        auto el_type = m_editor->get_element_type(name);
        //auto constant = ngraph::op::Constant::create(el_type, p_shape.to_shape(), value);
        //constant->set_friendly_name(name);
        std::shared_ptr<ngraph::op::Constant> constant = ngraph::op::Constant::create(el_type, p_shape.to_shape(), value);
        constant->set_friendly_name(name);
        map[name] = constant;
        m_editor->set_input_values(map);
    }


    /*
    auto tensor_place = pdpd::castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto p_shape1 = place_type->get_source_tensor
    place_type->
    auto type = tensor_place->get_element_type();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->get_names()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;*/

}