// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "color_utils.hpp"
#include "function_guard.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/function.hpp"
#include "preprocess_steps_impl.hpp"

namespace ov {
namespace preprocess {

/// \brief InputTensorInfoImpl - internal data structure
class InputTensorInfo::InputTensorInfoImpl {
public:
    InputTensorInfoImpl() = default;

    void set_element_type(const element::Type& type) {
        m_type = type;
        m_type_set = true;
    }
    bool is_element_type_set() const {
        return m_type_set;
    }
    const element::Type& get_element_type() const {
        return m_type;
    }

    void set_layout(const Layout& layout) {
        m_layout = layout;
        m_layout_set = true;
    }
    bool is_layout_set() const {
        return m_layout_set;
    }
    const Layout& get_layout() const {
        return m_layout;
    }

    bool is_spatial_shape_set() const {
        return m_spatial_shape_set;
    }

    int get_spatial_width() const {
        return m_spatial_width;
    }

    int get_spatial_height() const {
        return m_spatial_height;
    }

    bool is_spatial_shape_dynamic() const {
        return m_spatial_shape_set && m_spatial_width == -1 && m_spatial_height == -1;
    }

    void set_spatial_dynamic_shape() {
        m_spatial_shape_set = true;
        m_spatial_width = -1;
        m_spatial_height = -1;
    }

    void set_spatial_static_shape(size_t height, size_t width) & {
        m_spatial_shape_set = true;
        m_spatial_height = static_cast<int>(height);
        m_spatial_width = static_cast<int>(width);
    }

    const ColorFormat& get_color_format() const {
        return m_color_format;
    }

    void set_color_format(ColorFormat format, const std::vector<std::string>& sub_names) {
        auto info = ColorFormatInfo::get(format);
        if (info->planes_count() == 1) {
            OPENVINO_ASSERT(sub_names.empty(),
                            "Plane names are not allowed for single plane color format '",
                            color_format_name(format),
                            "'");
        } else if (!sub_names.empty()) {
            OPENVINO_ASSERT(sub_names.size() == info->planes_count(),
                            "Number of sub-names (",
                            sub_names.size(),
                            ") shall match with number of planes for '",
                            color_format_name(format),
                            "' color format (",
                            info->planes_count(),
                            ")");
        }
        m_planes_sub_names = sub_names;
        m_color_format = format;
    }

    const std::vector<std::string>& planes_sub_names() const {
        return m_planes_sub_names;
    }

private:
    ColorFormat m_color_format = ColorFormat::UNDEFINED;
    std::vector<std::string> m_planes_sub_names;

    element::Type m_type = element::dynamic;
    bool m_type_set = false;

    Layout m_layout = Layout();
    bool m_layout_set = false;

    int m_spatial_width = -1;
    int m_spatial_height = -1;
    bool m_spatial_shape_set = false;
};

/// \brief InputNetworkInfoImpl - internal data structure
class InputNetworkInfo::InputNetworkInfoImpl {
public:
    InputNetworkInfoImpl() = default;

    void set_layout(const Layout& layout) {
        m_layout = layout;
        m_layout_set = true;
    }
    bool is_layout_set() const {
        return m_layout_set;
    }
    const Layout& get_layout() const {
        return m_layout;
    }

private:
    Layout m_layout = Layout();
    bool m_layout_set = false;
};

/// \brief InputInfoImpl - internal data structure
struct InputInfo::InputInfoImpl {
    InputInfoImpl() = default;
    explicit InputInfoImpl(size_t idx) : m_has_index(true), m_index(idx) {}

    bool has_index() const {
        return m_has_index;
    }

    void create_tensor_data(const element::Type& type, const Layout& layout) {
        auto data = std::unique_ptr<InputTensorInfo::InputTensorInfoImpl>(new InputTensorInfo::InputTensorInfoImpl());
        data->set_layout(layout);
        data->set_element_type(type);
        m_tensor_data = std::move(data);
    }

    bool m_has_index = false;
    size_t m_index = 0;
    std::unique_ptr<InputTensorInfo::InputTensorInfoImpl> m_tensor_data;
    std::unique_ptr<PreProcessSteps::PreProcessStepsImpl> m_preprocess;
    std::unique_ptr<InputNetworkInfo::InputNetworkInfoImpl> m_network_data;
    std::shared_ptr<op::v0::Parameter> m_resolved_param;
};

//-------------- InputInfo ------------------
InputInfo::InputInfo() : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl)) {}
InputInfo::InputInfo(size_t input_index) : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl(input_index))) {}
InputInfo::InputInfo(InputInfo&&) noexcept = default;
InputInfo& InputInfo::operator=(InputInfo&&) noexcept = default;
InputInfo::~InputInfo() = default;

InputInfo& InputInfo::tensor(InputTensorInfo&& builder) & {
    m_impl->m_tensor_data = std::move(builder.m_impl);
    return *this;
}

InputInfo&& InputInfo::tensor(InputTensorInfo&& builder) && {
    m_impl->m_tensor_data = std::move(builder.m_impl);
    return std::move(*this);
}

InputInfo&& InputInfo::preprocess(PreProcessSteps&& builder) && {
    m_impl->m_preprocess = std::move(builder.m_impl);
    return std::move(*this);
}

InputInfo& InputInfo::preprocess(PreProcessSteps&& builder) & {
    m_impl->m_preprocess = std::move(builder.m_impl);
    return *this;
}

InputInfo& InputInfo::network(InputNetworkInfo&& builder) & {
    m_impl->m_network_data = std::move(builder.m_impl);
    return *this;
}

InputInfo&& InputInfo::network(InputNetworkInfo&& builder) && {
    m_impl->m_network_data = std::move(builder.m_impl);
    return std::move(*this);
}

// ------------------------ PrePostProcessor --------------------
struct PrePostProcessor::PrePostProcessorImpl {
public:
    std::list<std::unique_ptr<InputInfo::InputInfoImpl>> in_contexts;
};

PrePostProcessor::PrePostProcessor() : m_impl(std::unique_ptr<PrePostProcessorImpl>(new PrePostProcessorImpl())) {}
PrePostProcessor::PrePostProcessor(PrePostProcessor&&) noexcept = default;
PrePostProcessor& PrePostProcessor::operator=(PrePostProcessor&&) noexcept = default;
PrePostProcessor::~PrePostProcessor() = default;

PrePostProcessor& PrePostProcessor::input(InputInfo&& builder) & {
    m_impl->in_contexts.push_back(std::move(builder.m_impl));
    return *this;
}

PrePostProcessor&& PrePostProcessor::input(InputInfo&& builder) && {
    m_impl->in_contexts.push_back(std::move(builder.m_impl));
    return std::move(*this);
}

std::shared_ptr<Function> PrePostProcessor::build(const std::shared_ptr<Function>& function) {
    FunctionGuard guard(function);
    bool tensor_data_updated = false;
    for (const auto& input : m_impl->in_contexts) {
        std::shared_ptr<op::v0::Parameter> param;
        OPENVINO_ASSERT(input, "Internal error: Invalid preprocessing input, please report a problem");
        if (input->has_index()) {
            param = function->get_parameters().at(input->m_index);
        } else {
            // Default case
            OPENVINO_ASSERT(function->get_parameters().size() == 1,
                            std::string("Preprocessing info expects having 1 input, however function has ") +
                                std::to_string(function->get_parameters().size()) +
                                " inputs. Please use ov::preprocess::InputInfo constructor specifying "
                                "particular input instead of default one");
            param = function->get_parameters().front();
        }
        // Set parameter layout from 'network' information
        if (input->m_network_data && input->m_network_data->is_layout_set() && param->get_layout().empty()) {
            param->set_layout(input->m_network_data->get_layout());
        }
        input->m_resolved_param = param;
    }

    for (const auto& input : m_impl->in_contexts) {
        auto param = input->m_resolved_param;
        auto consumers = param->output(0).get_target_inputs();
        if (!input->m_tensor_data) {
            input->create_tensor_data(param->get_element_type(), param->get_layout());
        }
        if (!input->m_tensor_data->is_element_type_set()) {
            input->m_tensor_data->set_element_type(param->get_element_type());
        }
        auto color_info = ColorFormatInfo::get(input->m_tensor_data->get_color_format());
        if (!input->m_tensor_data->is_layout_set()) {
            if (!color_info->default_layout().empty()) {
                input->m_tensor_data->set_layout(color_info->default_layout());
            } else if (!param->get_layout().empty()) {
                input->m_tensor_data->set_layout(param->get_layout());
            }
        }

        auto net_shape = param->get_partial_shape();
        auto new_param_shape = net_shape;
        if (input->m_tensor_data->is_layout_set() && !param->get_layout().empty() &&
            param->get_layout() != input->m_tensor_data->get_layout()) {
            // Find transpose between network and tensor layouts and update tensor shape
            auto net_to_tensor =
                layout::find_permutation(param->get_layout(), net_shape.rank(), input->m_tensor_data->get_layout());
            std::vector<ov::Dimension> dims(new_param_shape.size());
            std::transform(net_to_tensor.begin(), net_to_tensor.end(), dims.begin(), [&](int64_t v) {
                return new_param_shape[v];
            });
            new_param_shape = PartialShape(dims);
        }
        if (input->m_tensor_data->is_spatial_shape_set()) {
            auto height_idx = get_and_check_height_idx(input->m_tensor_data->get_layout(), new_param_shape);
            auto width_idx = get_and_check_width_idx(input->m_tensor_data->get_layout(), new_param_shape);
            if (input->m_tensor_data->is_spatial_shape_dynamic()) {
                // Use dynamic spatial dimensions
                new_param_shape[height_idx] = Dimension::dynamic();
                new_param_shape[width_idx] = Dimension::dynamic();
            } else {
                // Use static spatial dimensions
                new_param_shape[height_idx] = input->m_tensor_data->get_spatial_height();
                new_param_shape[width_idx] = input->m_tensor_data->get_spatial_width();
            }
        }

        std::vector<std::shared_ptr<ov::Node>> nodes;
        std::vector<std::shared_ptr<op::v0::Parameter>> new_params;

        // Create separate parameter for each plane. Shape and friendly name is based on color format
        for (size_t plane = 0; plane < color_info->planes_count(); plane++) {
            auto plane_shape = color_info->shape(plane, new_param_shape);
            auto plane_param =
                std::make_shared<op::v0::Parameter>(input->m_tensor_data->get_element_type(), plane_shape);
            if (plane < input->m_tensor_data->planes_sub_names().size()) {
                auto sub_name = std::string("/") + input->m_tensor_data->planes_sub_names()[plane];
                inherit_friendly_names(function, param, plane_param, sub_name, false);
            } else {
                auto sub_name = color_info->friendly_suffix(plane);
                inherit_friendly_names(function, param, plane_param, sub_name);
            }
            if (!input->m_tensor_data->get_layout().empty()) {
                plane_param->set_layout(input->m_tensor_data->get_layout());
            }
            new_params.push_back(plane_param);
            nodes.push_back(plane_param);
        }

        PreprocessingContext context(input->m_tensor_data->get_layout());
        context.color_format() = input->m_tensor_data->get_color_format();
        context.network_layout() = param->get_layout();
        context.network_shape() = param->get_partial_shape();

        // 2. Apply preprocessing
        if (input->m_preprocess) {
            for (const auto& action : input->m_preprocess->actions()) {
                auto node = std::get<0>(action)(nodes, function, context);
                nodes = {node};
                tensor_data_updated |= std::get<1>(action);
            }
        }

        OPENVINO_ASSERT(nodes.size() == 1,
                        "Multiple plane input is not allowed as network input. Consider using of convert_color "
                        "preprocessing operation. Current format is '",
                        color_format_name(context.color_format()),
                        "'");
        OPENVINO_ASSERT(is_rgb_family(context.color_format()) || context.color_format() == ColorFormat::UNDEFINED,
                        "Network shall have RGB/BGR color format. Consider add 'convert_color' preprocessing operation "
                        "to convert current color format '",
                        color_format_name(context.color_format()),
                        "'to RGB/BGR");
        auto node = nodes[0];
        // Check final type
        OPENVINO_ASSERT(node->get_element_type() == param->get_element_type(),
                        std::string("Element type after preprocessing {") + node->get_element_type().c_type_string() +
                            std::string("} doesn't match with network element type {") +
                            param->get_element_type().c_type_string() +
                            "}. Please add 'convert_element_type' explicitly");

        // Replace parameter
        for (auto consumer : consumers) {
            consumer.replace_source_output(node);
        }
        function->add_parameters(new_params);
        // remove old parameter
        function->remove_parameter(param);
    }
    if (tensor_data_updated) {
        function->validate_nodes_and_infer_types();
    }
    guard.reset();
    return function;
}

// --------------------- InputTensorInfo ------------------
InputTensorInfo::InputTensorInfo() : m_impl(std::unique_ptr<InputTensorInfoImpl>(new InputTensorInfoImpl())) {}
InputTensorInfo::InputTensorInfo(InputTensorInfo&&) noexcept = default;
InputTensorInfo& InputTensorInfo::operator=(InputTensorInfo&&) noexcept = default;
InputTensorInfo::~InputTensorInfo() = default;

InputTensorInfo& InputTensorInfo::set_element_type(const element::Type& type) & {
    m_impl->set_element_type(type);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_element_type(const element::Type& type) && {
    m_impl->set_element_type(type);
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_layout(const Layout& layout) & {
    m_impl->set_layout(layout);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_layout(const Layout& layout) && {
    m_impl->set_layout(layout);
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_spatial_dynamic_shape() & {
    m_impl->set_spatial_dynamic_shape();
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_spatial_dynamic_shape() && {
    m_impl->set_spatial_dynamic_shape();
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_spatial_static_shape(size_t height, size_t width) & {
    m_impl->set_spatial_static_shape(height, width);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_spatial_static_shape(size_t height, size_t width) && {
    m_impl->set_spatial_static_shape(height, width);
    return std::move(*this);
}

// --------------------- InputNetworkInfo ------------------
InputNetworkInfo::InputNetworkInfo() : m_impl(std::unique_ptr<InputNetworkInfoImpl>(new InputNetworkInfoImpl())) {}
InputNetworkInfo::InputNetworkInfo(InputNetworkInfo&&) noexcept = default;
InputNetworkInfo& InputNetworkInfo::operator=(InputNetworkInfo&&) noexcept = default;
InputNetworkInfo::~InputNetworkInfo() = default;

InputNetworkInfo& InputNetworkInfo::set_layout(const Layout& layout) & {
    m_impl->set_layout(layout);
    return *this;
}

InputNetworkInfo&& InputNetworkInfo::set_layout(const Layout& layout) && {
    m_impl->set_layout(layout);
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_color_format(const ov::preprocess::ColorFormat& format,
                                                   const std::vector<std::string>& sub_names) & {
    m_impl->set_color_format(format, sub_names);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_color_format(const ov::preprocess::ColorFormat& format,
                                                    const std::vector<std::string>& sub_names) && {
    m_impl->set_color_format(format, sub_names);
    return std::move(*this);
}

// --------------------- PreProcessSteps ------------------

PreProcessSteps::PreProcessSteps() : m_impl(std::unique_ptr<PreProcessStepsImpl>(new PreProcessStepsImpl())) {}
PreProcessSteps::PreProcessSteps(PreProcessSteps&&) noexcept = default;
PreProcessSteps& PreProcessSteps::operator=(PreProcessSteps&&) noexcept = default;
PreProcessSteps::~PreProcessSteps() = default;

PreProcessSteps& PreProcessSteps::scale(float value) & {
    m_impl->add_scale_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps&& PreProcessSteps::scale(float value) && {
    m_impl->add_scale_impl(std::vector<float>{value});
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::scale(const std::vector<float>& values) & {
    m_impl->add_scale_impl(values);
    return *this;
}

PreProcessSteps&& PreProcessSteps::scale(const std::vector<float>& values) && {
    m_impl->add_scale_impl(values);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::mean(float value) & {
    m_impl->add_mean_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps&& PreProcessSteps::mean(float value) && {
    m_impl->add_mean_impl(std::vector<float>{value});
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::mean(const std::vector<float>& values) & {
    m_impl->add_mean_impl(values);
    return *this;
}

PreProcessSteps&& PreProcessSteps::mean(const std::vector<float>& values) && {
    m_impl->add_mean_impl(values);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::convert_element_type(const element::Type& type) & {
    m_impl->add_convert_impl(type);
    return *this;
}

PreProcessSteps&& PreProcessSteps::convert_element_type(const element::Type& type) && {
    m_impl->add_convert_impl(type);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) & {
    OPENVINO_ASSERT(dst_height <= std::numeric_limits<int>::max() && dst_width <= std::numeric_limits<int>::max(),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return *this;
}

PreProcessSteps&& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) && {
    OPENVINO_ASSERT(dst_height <= std::numeric_limits<int>::max() && dst_width <= std::numeric_limits<int>::max(),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg) & {
    m_impl->add_resize_impl(alg, -1, -1);
    return *this;
}

PreProcessSteps&& PreProcessSteps::resize(ResizeAlgorithm alg) && {
    m_impl->add_resize_impl(alg, -1, -1);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::convert_layout(const Layout& dst_layout) & {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PreProcessSteps&& PreProcessSteps::convert_layout(const Layout& dst_layout) && {
    m_impl->add_convert_layout_impl(dst_layout);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::convert_color(const ov::preprocess::ColorFormat& dst_format) & {
    m_impl->add_convert_color_impl(dst_format);
    return *this;
}

PreProcessSteps&& PreProcessSteps::convert_color(const ov::preprocess::ColorFormat& dst_format) && {
    m_impl->add_convert_color_impl(dst_format);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) & {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(std::make_tuple(
        [preprocess_cb](const std::vector<std::shared_ptr<ov::Node>>& nodes,
                        const std::shared_ptr<ov::Function>&,
                        PreprocessingContext&) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                            "current image to RGB/BGR color format using 'convert_color'");
            return {preprocess_cb(nodes[0])};
        },
        true));
    return *this;
}

PreProcessSteps&& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) && {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(std::make_tuple(
        [preprocess_cb](const std::vector<std::shared_ptr<ov::Node>>& nodes,
                        const std::shared_ptr<ov::Function>&,
                        PreprocessingContext&) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                            "current image to RGB/BGR color format using 'convert_color'");
            return {preprocess_cb(nodes[0])};
        },
        true));
    return std::move(*this);
}

}  // namespace preprocess
}  // namespace ov
