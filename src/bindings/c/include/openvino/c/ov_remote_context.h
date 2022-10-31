// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_remote_context C API
 *
 * @file ov_remote_context.h
 */
#pragma once
#include "openvino/c/ov_common.h"
#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"

typedef struct ov_remote_context ov_remote_context_t;

/**
 * @brief remote context property key
 */

//!< Read-write property: shared device context type, can be either pure OpenCL (OCL) or
//!< shared video decoder (VA_SHARED) context.
//!< Value is string, it can be one of below strings:
//!<    "OCL"       - Pure OpenCL context
//!<    "VA_SHARED" - Context shared with a video decoding device
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_context_type;

//!< Read-write property<void *>: identifies OpenCL context handle in a shared context or shared memory blob
//!< parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_context;

//!< Read-write property<int string>: ID of device in OpenCL context if multiple devices are present in the context.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_context_device_id;

//!< Read-write property<int string>: In case of multi-tile system, this key identifies tile within given context.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_tile_id;

//!< Read-write property<void *>: OpenCL queue handle in a shared context
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_queue;

//!< Read-write property<void *>: video acceleration device/display handle in a shared context or shared
//!< memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_va_device;

//!< Read-write property: type of internal shared memory in a shared memory blob
//!< parameter map.
//!< Value is string, it can be one of below strings:
//!<    "OCL_BUFFER"        - Shared OpenCL buffer blob
//!<    "OCL_IMAGE2D"       - Shared OpenCL 2D image blob
//!<    "USM_USER_BUFFER"   - Shared USM pointer allocated by user
//!<    "USM_HOST_BUFFER"   - Shared USM pointer type with host allocation type allocated by plugin
//!<    "USM_DEVICE_BUFFER" - Shared USM pointer type with device allocation type allocated by plugin
//!<    "VA_SURFACE"        - Shared video decoder surface or D3D 2D texture blob
//!<    "DX_BUFFER"         - Shared D3D buffer blob
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_shared_mem_type;

//!< Read-write property<void *>: OpenCL memory handle in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_mem_handle;

//!< Read-write property<uint32_t string>: video decoder surface handle in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_dev_object_handle;

//!< Read-write property<uint32_t string>: video decoder surface plane in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_va_plane;

// RemoteContext
/**
 * @defgroup remote_context remote_context
 * @ingroup openvino_c
 * Set of functions representing of RemoteContext.
 * @{
 */

/**
 * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
 * using the specified tensor description and low-level device-specific parameters.
 * Returns a pointer to the object that implements the RemoteTensor interface.
 * @ingroup remote_context
 * @param context A pointer to the ov_remote_context_t instance.
 * @param type Defines the element type of the tensor.
 * @param shape Defines the shape of the tensor.
 * @param object_args_size Size of the low-level tensor object parameters.
 * @param remote_tensor Pointer to returned ov_tensor_t that contains remote tensor instance.
 * @param variadic params Contains low-level tensor object parameters.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_create_tensor(const ov_remote_context_t* context,
                                const ov_element_type_e type,
                                const ov_shape_t shape,
                                const size_t object_args_size,
                                ov_tensor_t** remote_tensor,
                                ...);

/**
 * @brief Returns name of a device on which underlying object is allocated.
 * @ingroup remote_context
 * @param context A pointer to the ov_remote_context_t instance.
 * @param device_name Device name will be returned.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_get_device_name(const ov_remote_context_t* context, char** device_name);

/**
 * @brief Returns a string contains device-specific parameters required for low-level
 * operations with the underlying object.
 * Parameters include device/context handles, access flags,
 * etc. Content of the returned map depends on a remote execution context that is
 * currently set on the device (working scenario).
 * One actaul example: "CONTEXT_TYPE:OCL;OCL_CONTEXT:0x559ff6dab620;OCL_QUEUE:0x559ff6df06a0;"
 * @ingroup remote_context
 * @param context A pointer to the ov_remote_context_t instance.
 * @param size The size of param pairs.
 * @param params Param name:value list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_get_params(const ov_remote_context_t* context, size_t* size, char** params);

/**
 * @brief This method is used to create a host tensor object friendly for the device in current context.
 * For example, GPU context may allocate USM host memory (if corresponding extension is available),
 * which could be more efficient than regular host memory.
 * @ingroup remote_context
 * @param context A pointer to the ov_remote_context_t instance.
 * @param type Defines the element type of the tensor.
 * @param shape Defines the shape of the tensor.
 * @param tensor Pointer to ov_tensor_t that contains host tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_context_create_host_tensor(const ov_remote_context_t* context,
                                     const ov_element_type_e type,
                                     const ov_shape_t shape,
                                     ov_tensor_t** tensor);

/**
 * @brief Release the memory allocated by ov_remote_context_t.
 * @ingroup remote_context
 * @param context A pointer to the ov_remote_context_t to free memory.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(void) ov_remote_context_free(ov_remote_context_t* context);

/**
 * @brief Returns a string contains device-specific parameters required for low-level
 * operations with underlying object.
 * Parameters include device/context/surface/buffer handles, access flags,
 * etc. Content of the returned map depends on remote execution context that is
 * currently set on the device (working scenario).
 * One example: "MEM_HANDLE:0x559ff6904b00;OCL_CONTEXT:0x559ff71d62f0;SHARED_MEM_TYPE:OCL_BUFFER;"
 * @ingroup remote_context
 * @param tensor Pointer to ov_tensor_t that contains host tensor.
 * @param size The size of param pairs.
 * @param params Param name:value list.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_tensor_get_params(ov_tensor_t* tensor, size_t* size, char** params);

/**
 * @brief Returns name of a device on which underlying object is allocated.
 * @ingroup remote_context
 * @param remote_tensor A pointer to the remote tensor instance.
 * @param device_name Device name will be return.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_remote_tensor_get_device_name(ov_tensor_t* remote_tensor, char** device_name);