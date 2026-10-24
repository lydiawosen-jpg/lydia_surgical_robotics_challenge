cmake_minimum_required(VERSION 3.16)
project(ring_wire_internal_constraint_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(AMBF)

find_package(Boost COMPONENTS program_options filesystem)

include_directories(${AMBF_INCLUDE_DIRS})

add_definitions(${AMBF_DEFINITIONS})

set(PLUGIN_NAME ring_wire_internal_constraint)

add_library(${PLUGIN_NAME} SHARED ring_wire_internal_constraint_plugin.cpp)

target_link_libraries(${PLUGIN_NAME} PRIVATE ambf_framework)

set_target_properties(${PLUGIN_NAME} PROPERTIES
  PREFIX "lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BUILD_DIR}"
)
