cmake_minimum_required(VERSION 3.16)

set(addon_name "ngsolve_amgcl")
project(${addon_name})

include(ngsolve_addon.cmake)

# amgcl is a header-only library
add_library(amgcl INTERFACE)
target_include_directories(amgcl INTERFACE ${CMAKE_SOURCE_DIR}/extern/amgcl)

# Add the ngsolve addon
add_ngsolve_addon(${addon_name}
  src/ngsolve_amgcl.cpp
  src/amgcl_precond.cpp
)

target_include_directories(${addon_name} PRIVATE ${CMAKE_SOURCE_DIR}/extern/amgcl)

# Find OpenMP - amgcl can use OpenMP for shared memory parallelism
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${addon_name} PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found - enabling shared memory parallelism for amgcl")
else()
  message(STATUS "OpenMP not found - amgcl will run without shared memory parallelism")
endif()

# Install the module
install(TARGETS ${addon_name} DESTINATION ${ADDON_INSTALL_DIR_PYTHON}/${addon_name})

# Install __init__.py
install(FILES ${CMAKE_SOURCE_DIR}/src/__init__.py DESTINATION ${ADDON_INSTALL_DIR_PYTHON}/${addon_name})

# Install demos
install(DIRECTORY ${CMAKE_SOURCE_DIR}/demos/ DESTINATION ${ADDON_INSTALL_DIR_PYTHON}/${addon_name}/demos
  FILES_MATCHING PATTERN "*.py"
)

# Generate stub files
ngsolve_generate_stub_files(${addon_name})
