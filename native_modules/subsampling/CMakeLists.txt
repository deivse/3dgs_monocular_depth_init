cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pointcloud_subsampling src/pointcloud_subsampling.cpp)
install(TARGETS _pointcloud_subsampling DESTINATION .)

# This is passing in the version as a define just as an example
target_compile_definitions(_pointcloud_subsampling PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _pointcloud_subsampling DESTINATION pointcloud_subsampling)
