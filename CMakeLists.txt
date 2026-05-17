cmake_minimum_required(VERSION 3.18)
project(opencl_tf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Python + TensorFlow flag discovery
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import tensorflow as tf; print(';'.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TF_CFLAGS_RC
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import tensorflow as tf; print(';'.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TF_LFLAGS_RC
)
if(NOT TF_CFLAGS_RC EQUAL 0 OR NOT TF_LFLAGS_RC EQUAL 0)
    message(FATAL_ERROR
        "Could not query TensorFlow flags. Is `tensorflow` installed in the "
        "Python interpreter at ${Python3_EXECUTABLE}?")
endif()

message(STATUS "TF compile flags: ${TF_CFLAGS}")
message(STATUS "TF link flags:    ${TF_LFLAGS}")

# ---------------------------------------------------------------------------
# OpenCL
# ---------------------------------------------------------------------------
find_package(OpenCL REQUIRED)
message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libs:    ${OpenCL_LIBRARIES}")

# ---------------------------------------------------------------------------
# The custom-op shared library
# ---------------------------------------------------------------------------
set(OPENCL_TF_SOURCES
    src/cl_backend.cc
    src/ops/conv2d_ops.cc
)

add_library(opencl_tf_ops SHARED ${OPENCL_TF_SOURCES})

target_include_directories(opencl_tf_ops
    PRIVATE
        src
        ${OpenCL_INCLUDE_DIRS}
)

# TF wheel ABI flags arrive as a list — apply them as compile options.
target_compile_options(opencl_tf_ops PRIVATE ${TF_CFLAGS} -Wall -Wextra)
target_link_options   (opencl_tf_ops PRIVATE ${TF_LFLAGS})
target_link_libraries (opencl_tf_ops PRIVATE OpenCL::OpenCL ${CMAKE_DL_LIBS})

# TF expects a bare `.so` filename (no `lib` prefix) when calling
# tf.load_op_library, and the Python package looks for the file next to
# itself, so put the build output directly into opencl_tf/.
set_target_properties(opencl_tf_ops PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/opencl_tf"
)

# ---------------------------------------------------------------------------
# Install rules (optional)
# ---------------------------------------------------------------------------
install(TARGETS opencl_tf_ops LIBRARY DESTINATION opencl_tf)
install(DIRECTORY kernels/ DESTINATION kernels FILES_MATCHING PATTERN "*.cl")
