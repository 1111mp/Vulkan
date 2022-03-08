message(STATUS "################Add Module Thgirdparty################")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )

add_subdirectory(GLFW)

add_library(imgui
  imgui/imconfig.h
  imgui/imgui.h
  imgui/imgui.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_internal.h
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/imstb_rectpack.h
  imgui/imstb_textedit.h
  imgui/imstb_truetype.h
  imgui/imgui_demo.cpp
)
add_library(stb_image
  stb_image/stb_image.h
  stb_image/stb_image.cpp
)
