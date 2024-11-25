// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace intel_npu {

constexpr std::string_view MAGIC_BYTES = "OVNPU";

constexpr uint32_t make_version(uint16_t major, uint16_t minor) {
    return (major << sizeof(major) * 8) | (minor & 0x0000ffff);
}

constexpr uint32_t METADATA_VERSION_1_0 { make_version(1, 0) };

constexpr uint32_t CURRENT_METADATA_VERSION { METADATA_VERSION_1_0 };

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

template <uint32_t version>
struct Metadata : public MetadataBase {};

template <>
struct Metadata<METADATA_VERSION_1_0> : public MetadataBase {
    uint32_t version;
    OpenvinoVersion ovVersion;

    Metadata();

    void read(std::istream& stream) override;

    void write(std::ostream& stream) override;

    bool isCompatible() override;
};

std::unique_ptr<MetadataBase> createMetadata(uint32_t version);

std::unique_ptr<MetadataBase> read_metadata_from(std::vector<uint8_t>& blob);

}  // namespace intel_npu
