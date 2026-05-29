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

# Backend selection: use NGSolve's SparseMatrix/BaseVector (TaskManager parallel)
# or AMGCL's builtin backend (OpenMP parallel)
option(USE_NGSOLVE_BACKEND "Use NGSolve backend (TaskManager) instead of builtin (OpenMP)" ON)
if(USE_NGSOLVE_BACKEND)
  target_compile_definitions(${addon_name} PRIVATE USE_NGSOLVE_BACKEND=1)
  message(STATUS "Using NGSolve backend (TaskManager parallelism)")
else()
  target_compile_definitions(${addon_name} PRIVATE USE_NGSOLVE_BACKEND=0)
  message(STATUS "Using AMGCL builtin backend (OpenMP parallelism)")
endif()

# Boost is optional - amgcl uses it for property_tree parameter passing,
# but we only use struct-based params so it's not needed.
option(USE_BOOST "Use Boost for amgcl (optional, not required)" ON)
if(USE_BOOST)
  cmake_policy(SET CMP0167 OLD)
  find_package(Boost QUIET)
endif()
if(Boost_FOUND)
  message(STATUS "Boost found - using Boost-enabled amgcl")
  target_link_libraries(${addon_name} PRIVATE Boost::boost)
else()
  message(STATUS "Building amgcl without Boost (AMGCL_NO_BOOST)")
  target_compile_definitions(${addon_name} PRIVATE AMGCL_NO_BOOST)
endif()

# Find OpenMP - only needed for AMGCL builtin backend
if(NOT USE_NGSOLVE_BACKEND)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${addon_name} PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found")
  else()
    message(STATUS "OpenMP not found - builtin backend will run single-threaded")
  endif()
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
