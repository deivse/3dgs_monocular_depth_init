set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)

set(module_name "pointcloud_subsampling")
set(native_target "_pointcloud_subsampling")

file(GLOB_RECURSE SUBSAMPLING_SRC CONFIGURE_DEPENDS src/*.cpp)
pybind11_add_module(${native_target} ${SUBSAMPLING_SRC})
target_link_libraries(${native_target} PRIVATE Eigen3::Eigen mdi_common)


# Install Python package files
install(DIRECTORY src/${module_name}/ 
        DESTINATION ${module_name}
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi")

# Install the native module itself
install(TARGETS ${native_target} DESTINATION ${module_name})
