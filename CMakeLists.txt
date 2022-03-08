# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.18)

project ("Vulkan")

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

configure_file (
  "${PROJECT_SOURCE_DIR}/HazelConfig.h.in"
  "${PROJECT_SOURCE_DIR}/src/Config.h"
)

function(assign_source_group)
  foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE "${_source}")
      file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
    else()
      set(_source_rel "${_source}")
    endif()
    get_filename_component(_source_path "${_source_rel}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

function(my_add_executable)
  foreach(_source IN ITEMS ${ARGN})
    assign_source_group(${_source})
  endforeach()
  add_executable(${ARGV})
endfunction(my_add_executable)

add_subdirectory("Thirdparty")

file(GLOB_RECURSE ${PROJECT_NAME}_HEADER_FILES src/*.h *.hpp)
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCE_FILES src/*.cpp)

my_add_executable(${PROJECT_NAME}
  ${${PROJECT_NAME}_HEADER_FILES}
  ${${PROJECT_NAME}_SOURCE_FILES}
)

find_package(Vulkan REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${Vulkan_INCLUDE_DIR}
  "Thirdparty/GLFW/include"
  "Thirdparty/glm"
  "Thirdparty/stb_image"
  "Thirdparty/imgui"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${Vulkan_LIBRARIES}
  glfw
  stb_image
  imgui
)
