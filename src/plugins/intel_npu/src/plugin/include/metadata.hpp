// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace intel_npu {

constexpr std::string_view MAGIC_BYTES = "OVNPU";

constexpr uint16_t make_version(uint8_t major, uint8_t minor) {
    return ((0 | major) << sizeof(major) * 8) | minor;
}

constexpr uint16_t METADATA_VERSION_1_0 { make_version(1, 0) };

constexpr uint16_t CURRENT_METADATA_VERSION { METADATA_VERSION_1_0 };

struct OpenvinoVersion {
    std::string version;
    uint32_t size;

    OpenvinoVersion(const std::string& version);

    void read(std::istream& stream);
};

struct MetadataBase {
    virtual void read(std::istream& stream) = 0;
    virtual void write(std::ostream& stream) = 0;
    virtual bool isCompatible() = 0;
    virtual ~MetadataBase() = default;
};

template <uint16_t version>
struct Metadata : public MetadataBase {};

template <>
struct Metadata<METADATA_VERSION_1_0> : public MetadataBase {
    uint16_t version;
    OpenvinoVersion ovVersion;

    Metadata();

    void read(std::istream& stream) override;

    void write(std::ostream& stream) override;

    bool isCompatible() override;
};

std::unique_ptr<MetadataBase> createMetadata(uint16_t version);

std::unique_ptr<MetadataBase> read_metadata_from(std::vector<uint8_t>& blob);

}  // namespace intel_npu
