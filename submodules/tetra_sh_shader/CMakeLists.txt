cmake_minimum_required(VERSION 3.16)
project(tetra_sh_shader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tetra_sh_shader STATIC
  src/voxel_node.cpp
  src/voxel_grid.cpp
  src/pre_resterization.cpp
)
target_include_directories(tetra_sh_shader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tetra_sh_shader PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "tetra_sh_shader ready. Use `cmake -S . -B build` then `cmake --build build`.")
