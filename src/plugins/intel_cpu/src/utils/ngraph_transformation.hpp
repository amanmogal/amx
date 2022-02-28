// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include "config.h"
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>

namespace ov {
namespace intel_cpu {

class TransformationDumper {
public:
    explicit TransformationDumper(const Config& config, const Config::TransformationFilter::Type type,
                                  const std::shared_ptr<ngraph::Function>& nGraphFunc)
        : config(config), type(type), nGraphFunc(nGraphFunc) {
        for (auto prev = infoMap.at(type).prev; prev != TransformationType::NumOfTypes;
             prev = infoMap.at(prev).prev) {
            // no need to serialize input graph if there was no transformations from previous dump
            if (config.disable.transformations.filter[prev])
                continue;
            if (!config.dumpIR.transformations.filter[prev])
                break;
            if (wasDumped()[prev])
                return;
        }
        dump("_in");
    }
    ~TransformationDumper() {
        dump("_out");
        wasDumped().set(type);
    }

private:
    const Config& config;
    const std::shared_ptr<ngraph::Function>& nGraphFunc;
    using TransformationType = Config::TransformationFilter::Type;
    const TransformationType type;

    struct TransformationInfo {
        std::string name;
        TransformationType prev;
    };
    // std::hash<int> is necessary for Ubuntu-16.04 (gcc-5.4 and defect in C++11 standart)
    const std::unordered_map<TransformationType, TransformationInfo, std::hash<uint8_t>> infoMap =
        {{TransformationType::Common,     {"common", TransformationType::NumOfTypes}},
         {TransformationType::Lpt,        {"lpt", TransformationType::NumOfTypes}},
         {TransformationType::Snippets,   {"snippets", TransformationType::Common}},
         {TransformationType::Specific,   {"cpuSpecificOpSet", TransformationType::Snippets}}};
    std::bitset<TransformationType::NumOfTypes>& wasDumped(void) {
        static std::bitset<TransformationType::NumOfTypes> wasDumped;
        return wasDumped;
    }
    void dump(const std::string&& postfix) {
        static int num = 0; // just to keep dumped IRs ordered in filesystem
        const auto pathAndName = config.dumpIR.dir + "/ir_" + std::to_string(num) + '_' +
                                 infoMap.at(type).name + postfix;
        ov::pass::Manager serializer;
        if (config.dumpIR.format.filter[Config::IrFormatFilter::Xml])
            serializer.register_pass<ov::pass::Serialize>(pathAndName + ".xml", "");
        if (config.dumpIR.format.filter[Config::IrFormatFilter::Svg]) {
            serializer.register_pass<ov::pass::VisualizeTree>(pathAndName + ".svg");
        } else if (config.dumpIR.format.filter[Config::IrFormatFilter::Dot]) {
            serializer.register_pass<ov::pass::VisualizeTree>(pathAndName + ".dot");
        }
        serializer.run_passes(nGraphFunc);
        num++;
    }
};

}   // namespace intel_cpu
}   // namespace ov

#  define CPU_DEBUG_CAP_TRANSFORMATION_RETURN_OR_DUMP(_type)                                            \
    if (config.disable.transformations.filter[Config::TransformationFilter::Type::_type])               \
        return;                                                                                         \
    auto dumperPtr = config.dumpIR.transformations.filter[Config::TransformationFilter::Type::_type] ?  \
        std::unique_ptr<TransformationDumper>(new TransformationDumper(config,                          \
                                              Config::TransformationFilter::Type::_type, nGraphFunc)) : \
        nullptr
#else
#  define CPU_DEBUG_CAP_TRANSFORMATION_RETURN_OR_DUMP(_type)
#endif // CPU_DEBUG_CAPS
