cmake_minimum_required(VERSION 3.24)

project(vulkan_tutorial VERSION 0.1.0 LANGUAGES CXX)

file(GLOB SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB INCLUDE_FILES "${PROJECT_SOURCE_DIR}/src/*.hpp")
file(GLOB SHADERS
  "${PROJECT_SOURCE_DIR}/src/shaders/*.frag"
  "${PROJECT_SOURCE_DIR}/src/shaders/*.vert"
)

find_package(Vulkan REQUIRED COMPONENTS glslc)

set(SPIRV_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${SPIRV_BINARY_DIR}
  OUTPUT ${SPIRV_BINARY_DIR}
)

foreach(source IN LISTS SHADERS)
  get_filename_component(FILENAME ${source} NAME)
  add_custom_command(
    COMMAND
      ${Vulkan_GLSLC_EXECUTABLE}
      -o ${SPIRV_BINARY_DIR}/${FILENAME}.spv
      ${source}
    OUTPUT ${SPIRV_BINARY_DIR}/${FILENAME}.spv
    DEPENDS ${source} ${SPIRV_BINARY_DIR}
  )
  list(APPEND SPV_SHADERS ${SPIRV_BINARY_DIR}/${FILENAME}.spv)
endforeach()

add_custom_target(binaryShaders ALL DEPENDS ${SPV_SHADERS})

add_executable(VulkanTest ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(VulkanTest PRIVATE "./external")
add_dependencies(VulkanTest binaryShaders)

find_package(Lua 5.4 REQUIRED)
target_link_libraries(VulkanTest PRIVATE ${LUA_LIBRARIES})
target_include_directories(VulkanTest PRIVATE ${LUA_INCLUDE_DIR})

target_include_directories(VulkanTest PRIVATE "./external/sol2/include")

target_link_libraries(VulkanTest PRIVATE Vulkan::Vulkan)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(VulkanTest PRIVATE glfw)

find_package(glm REQUIRED)
target_link_libraries(VulkanTest PRIVATE glm::glm)

target_compile_features(VulkanTest PRIVATE cxx_std_20)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(VulkanTest PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers -Wno-unknown-pragmas)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(VulkanTest PRIVATE /W4 /WX)
else()
  message(FATAL_ERROR "Unsupported compiler type")
endif()

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
