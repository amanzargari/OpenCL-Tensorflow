# =============================================================================
# Makefile -- build the OpenCL TensorFlow ops shared library.
# Alternative to CMakeLists.txt. Pick whichever you prefer.
# =============================================================================

CXX      ?= g++

TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra -Isrc $(TF_CFLAGS)
LDFLAGS  := -shared $(TF_LFLAGS) -lOpenCL -ldl

SRC := src/cl_backend.cc src/ops/conv2d_ops.cc
OBJ := $(SRC:.cc=.o)

TARGET := opencl_tf/opencl_tf_ops.so

.PHONY: all clean test info

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)
	@echo ""
	@echo "Built $@"

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

test: $(TARGET)
	pytest tests/ -v

info:
	@echo "CXX       = $(CXX)"
	@echo "TF_CFLAGS = $(TF_CFLAGS)"
	@echo "TF_LFLAGS = $(TF_LFLAGS)"
	@echo "SRC       = $(SRC)"
	@echo "TARGET    = $(TARGET)"
