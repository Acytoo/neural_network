
set(module_name core)

project(
  ${module_name}
  VERSION 0.0.1
  LANGUAGES CXX
  )

message(STATUS "Configuring ${module_name}")



set(sources
    src/simple_net.cc
)

set(headers
    include/simple_net.h
)

add_library(${module_name} ${sources} ${headers})


target_include_directories(
    ${module_name}
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
