
# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.13)
project(neural_network VERSION 0.0.2)

find_package(OpenCV REQUIRED)

if (NOT OpenCV_FOUND)
  message(FATAL_ERROR "FATAL: OpenCV 3.2.0 is needed")
endif()

if (NOT "${OpenCV_VERSION}" STREQUAL "3.2.0")
  message(FATAL_ERROR "FATAL: Only OpenCV 3.2.0 is supported")
else()
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")


add_subdirectory(core)
add_subdirectory(test)

set(TARGET_EXECUTABLE class_test)
add_executable(${TARGET_EXECUTABLE} test/class_test2.cc)
target_link_libraries(${TARGET_EXECUTABLE} PRIVATE ${OpenCV_LIBS} core)



set(TARGET_EXECUTABLE mnist_example)
add_executable(${TARGET_EXECUTABLE} mnist_example/start.cc)
target_link_libraries(${TARGET_EXECUTABLE} PRIVATE ${OpenCV_LIBS} core)
