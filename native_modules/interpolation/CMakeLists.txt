find_package(pybind11 REQUIRED CONFIG)
find_package(cgal REQUIRED CONFIG)
find_package(fmt REQUIRED CONFIG)

set(module_name "scale_factor_interpolation")
set(native_target "_scale_factor_interpolation")

file(GLOB_RECURSE INTERP_SRC CONFIGURE_DEPENDS src/*.cpp)
pybind11_add_module(${native_target} ${INTERP_SRC})
target_link_libraries(${native_target} PRIVATE CGAL::CGAL mdi_common fmt::fmt)

# Install Python package files
install(DIRECTORY src/${module_name}/ 
        DESTINATION ${module_name}
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi")

# Install the native module itself
install(TARGETS ${native_target} DESTINATION ${module_name})
