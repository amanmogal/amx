# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino)

#
# Add openvino library
#

add_library(${TARGET_NAME}
    $<TARGET_OBJECTS:ngraph_obj>
    $<TARGET_OBJECTS:frontend_common_obj>
    $<TARGET_OBJECTS:inference_engine_obj>
    $<TARGET_OBJECTS:inference_engine_transformations_obj>
    $<TARGET_OBJECTS:inference_engine_lp_transformations_obj>)

add_library(openvino::runtime ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime)

ie_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "OpenVINO runtime library")
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/frontends/common/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/include/ie>)

target_link_libraries(${TARGET_NAME} PRIVATE ngraph_reference
                                             ngraph_builders
                                             ov_shape_inference
                                             pugixml::static
                                             ${CMAKE_DL_LIBS}
                                             Threads::Threads)

if (TBBBIND_2_5_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${TBBBIND_2_5_IMPORTED_TARGETS})
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

set_ie_threading_interface_for(${TARGET_NAME})
ie_mark_target_as_cc(${TARGET_NAME})

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_register_plugins(MAIN_TARGET ${TARGET_NAME})

# Export for build tree

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        NAMELINK_COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        INCLUDES DESTINATION ${OV_CPACK_INCLUDEDIR}
                             ${OV_CPACK_INCLUDEDIR}/ie)

# OpenVINO runtime library dev

#
# Add openvin::dev target
#

add_library(${TARGET_NAME}_dev INTERFACE)
add_library(openvino::runtime::dev ALIAS ${TARGET_NAME}_dev)

target_include_directories(${TARGET_NAME}_dev INTERFACE
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/common/transformations/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/dev_api>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/dev_api>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/common/low_precision_transformations/include>
    $<TARGET_PROPERTY:openvino_gapi_preproc,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET_NAME}_dev INTERFACE
    $<TARGET_PROPERTY:openvino_gapi_preproc,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(${TARGET_NAME}_dev INTERFACE ${TARGET_NAME} pugixml::static openvino::itt openvino::util)

set_ie_threading_interface_for(${TARGET_NAME}_dev)
set_target_properties(${TARGET_NAME}_dev PROPERTIES EXPORT_NAME runtime::dev)

openvino_developer_export_targets(COMPONENT core TARGETS openvino::runtime::dev)

# Install static libraries for case BUILD_SHARED_LIBS=OFF
ov_install_static_lib(${TARGET_NAME}_dev ${OV_CPACK_COMP_CORE})

#
# Install OpenVINO runtime
#

list(APPEND PATH_VARS "IE_INCLUDE_DIR")

# TODO: dpkg-shlibdeps does not work otherwise
# TODO: define proper library version, currently SOVERSION 2022
# set_target_properties(${TARGET_NAME} PROPERTIES
#     SOVERSION ${OpenVINO_VERSION_MAJOR}
#     VERSION ${OpenVINO_VERSION})

if(ENABLE_INTEL_GNA)
    list(APPEND PATH_VARS "GNA_PATH")
endif()

ie_cpack_add_component(core REQUIRED DEPENDS ${core_components})
ie_cpack_add_component(core_dev REQUIRED DEPENDS core ${core_dev_components})

if(BUILD_SHARED_LIBS)
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION ${OV_CPACK_PLUGINSDIR}
            COMPONENT core)

    # for InferenceEngineUnitTest
    # For public tests
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
    # For private tests
    if (NOT WIN32)
        install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
                DESTINATION tests/lib COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endif()

# Install cmake scripts

install(EXPORT OpenVINOTargets
        FILE OpenVINOTargets.cmake
        NAMESPACE openvino::
        DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})

set(PUBLIC_HEADERS_DIR "${OpenVINO_SOURCE_DIR}/src/inference/include")
set(IE_INCLUDE_DIR "${PUBLIC_HEADERS_DIR}/ie")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
                               INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                               PATH_VARS ${PATH_VARS})

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                              PATH_VARS ${PATH_VARS})

set(IE_INCLUDE_DIR "${OV_CPACK_INCLUDEDIR}/ie")
set(IE_TBB_DIR "${IE_TBB_DIR_INSTALL}")
set(IE_TBBBIND_DIR "${IE_TBBBIND_DIR_INSTALL}")
set(GNA_PATH "${OV_CPACK_RUNTIMEDIR}")
if(WIN32)
    set(GNA_PATH "${OV_CPACK_LIBRARYDIR}/../Release")
endif()

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
                              INSTALL_DESTINATION ${OV_CPACK_IE_CMAKEDIR}
                              PATH_VARS ${PATH_VARS})

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
                              PATH_VARS ${PATH_VARS})

configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake" @ONLY)
configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake" @ONLY)

install(FILES "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
              "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
        DESTINATION ${OV_CPACK_IE_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})

install(FILES "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
              "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake"
        DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})
