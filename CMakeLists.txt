
# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.13)
project(neural_network VERSION 0.0.1)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}))

add_subdirectory(core)
add_subdirectory(test)

set(TARGET_EXECUTABLE_GUI class_test)
add_executable(${TARGET_EXECUTABLE_GUI} test/class_test.cc)
target_link_libraries(${TARGET_EXECUTABLE_GUI} PRIVATE ${OpenCV_LIBS} core)
