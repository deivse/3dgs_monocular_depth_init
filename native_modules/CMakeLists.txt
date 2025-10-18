cmake_minimum_required(VERSION 3.15...4.0)
project("scale_factor_interpolation" LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

set(PYBIND11_FINDPYTHON ON)

add_subdirectory(common)
add_subdirectory(interpolation)
add_subdirectory(subsampling)
