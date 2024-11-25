// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <cstring>
#include <sstream>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/version.hpp"

namespace intel_npu {

OpenvinoVersion::OpenvinoVersion(const std::string& version) {
    this->version = version;
    this->size = static_cast<uint32_t>(version.size());
}

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    stream.read(&version[0], size);
}

Metadata<METADATA_VERSION_1_0>::Metadata() : ovVersion{ov::get_openvino_version().buildNumber} { version = METADATA_VERSION_1_0; }

void Metadata<METADATA_VERSION_1_0>::read(std::istream& stream) {
    ovVersion.read(stream);
}

void Metadata<METADATA_VERSION_1_0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));

    stream.write(reinterpret_cast<const char*>(&ovVersion.size), sizeof(ovVersion.size));
    stream.write(ovVersion.version.c_str(), ovVersion.version.size());
}

std::unique_ptr<MetadataBase> createMetadata(uint32_t version) {
    switch (version) {
        case METADATA_VERSION_1_0:
            return std::make_unique<Metadata<METADATA_VERSION_1_0>>();

        default:
            return nullptr;
    }
}

bool Metadata<METADATA_VERSION_1_0>::isCompatible() {
    Logger _logger("NPUPlugin", Logger::global().level());
    // checking if we still support the format
    if (version != CURRENT_METADATA_VERSION) {
        return false;
    }

    std::string_view currentOvVersion(ov::get_openvino_version().buildNumber);
    // checking if we can import the blob
    if (ovVersion.version != currentOvVersion) {
        _logger.warning("Imported blob metadata version: %s, but the current OpenVINO version is: %s", ovVersion.version, currentOvVersion.data());
        return false;
    }
    return true;;
}

std::unique_ptr<MetadataBase> read_metadata_from(std::vector<uint8_t>& blob) {
    Logger _logger("NPUPlugin", Logger::global().level());
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes(magicBytesSize, '\0');

    auto metadataIterator = blob.end() - magicBytesSize;
    memcpy(blobMagicBytes.data(), &(*metadataIterator), magicBytesSize);
    if (MAGIC_BYTES != blobMagicBytes) {
        _logger.error("Blob is not versioned");
        return nullptr;
    }

    size_t blobDataSize;
    metadataIterator -= sizeof(blobDataSize);
    memcpy(&blobDataSize, &(*metadataIterator), sizeof(blobDataSize));
    metadataIterator = blob.begin() + blobDataSize;

    std::stringstream metadataStream;
    metadataStream.write(reinterpret_cast<const char*>(&(*metadataIterator)),
                         blob.end() - metadataIterator - sizeof(blobDataSize));

    uint32_t metaVersion;
    metadataStream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    auto storedMeta = createMetadata(metaVersion);
    if (storedMeta != nullptr) {
        storedMeta->read(metadataStream);
    }
    return storedMeta;
}

}  // namespace intel_npu
