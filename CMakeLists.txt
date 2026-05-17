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

# Split TF_LFLAGS into search dirs ('-L...') and library specs ('-l...').
# Library specs MUST go through target_link_libraries; routing them via
# target_link_options causes CMake to drop or mangle them and the resulting
# .so will be missing tensorflow_framework, producing
#   "undefined symbol: _ZTIN10tensorflow8OpKernelE"
# at tf.load_op_library time.
set(TF_LINK_DIRS "")
set(TF_LINK_LIBS "")
foreach(flag IN LISTS TF_LFLAGS)
    if(flag MATCHES "^-L(.+)$")
        list(APPEND TF_LINK_DIRS "${CMAKE_MATCH_1}")
    elseif(flag MATCHES "^-l")
        list(APPEND TF_LINK_LIBS "${flag}")
    else()
        # Pass-through for anything unusual (rare in TF's output).
        list(APPEND TF_LINK_LIBS "${flag}")
    endif()
endforeach()
message(STATUS "TF link dirs: ${TF_LINK_DIRS}")
message(STATUS "TF link libs: ${TF_LINK_LIBS}")

# ---------------------------------------------------------------------------
# OpenCL
# ---------------------------------------------------------------------------
find_package(OpenCL REQUIRED)
message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libs:    ${OpenCL_LIBRARIES}")

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(OPENCL_TF_SOURCES
    src/cl_backend.cc
    src/ops/conv2d_ops.cc
    src/ops/depthwise_conv2d_ops.cc
    src/ops/relu_ops.cc
    src/ops/batchnorm_ops.cc
    src/ops/sigmoid_ops.cc
    src/ops/dense_ops.cc
    src/ops/upsampling_ops.cc
)

add_library(opencl_tf_ops SHARED ${OPENCL_TF_SOURCES})

target_include_directories(opencl_tf_ops
    PRIVATE
        src
        ${OpenCL_INCLUDE_DIRS}
)

target_compile_options(opencl_tf_ops PRIVATE ${TF_CFLAGS} -Wall -Wextra)

# Library search directories (the '-L...' bits)
target_link_directories(opencl_tf_ops PRIVATE ${TF_LINK_DIRS})

# Libraries to link. TF's library specs (including '-l:libtensorflow_framework.so.2')
# go here, NOT into target_link_options.
target_link_libraries(opencl_tf_ops PRIVATE
    ${TF_LINK_LIBS}
    OpenCL::OpenCL
    ${CMAKE_DL_LIBS}
)

# Default output: next to the Python package in the source tree (dev workflow).
# pip install overrides this via -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=<staging_dir>.
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/opencl_tf")
endif()

set_target_properties(opencl_tf_ops PROPERTIES PREFIX "")

# ---------------------------------------------------------------------------
# Install rules (optional)
# ---------------------------------------------------------------------------
install(TARGETS opencl_tf_ops LIBRARY DESTINATION opencl_tf)
install(DIRECTORY kernels/ DESTINATION kernels FILES_MATCHING PATTERN "*.cl")
