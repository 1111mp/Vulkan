message(STATUS "################Add Module Thgirdparty################")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )

add_subdirectory(GLFW)

add_library(stb_image
  stb_image/stb_image.h
  stb_image/stb_image.cpp
)
