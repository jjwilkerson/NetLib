# Compiler settings
NVCC = /usr/local/cuda-12.1/bin/nvcc
CCFLAGS = -I/usr/local/include/utf8 -I/usr/local/include/jsoncpp -I/usr/include/boost -I/usr/local/cuda-samples/Common -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -ccbin g++

# Find all .cpp and .cu files in the current directory and subdirectories
CPP_SRCS = $(wildcard *.cpp) $(wildcard */*.cpp) $(wildcard */*/*.cpp)
CU_SRCS = $(wildcard *.cu) $(wildcard */*.cu) $(wildcard */*/*.cu)

# Object files corresponding to source files
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
CU_OBJS = $(CU_SRCS:.cu=.cu.o)
OBJS = $(CPP_OBJS) $(CU_OBJS)

# Output library name
LIB = libNetLib.a

# Default target
all: $(LIB)

# Compile source files into object files
%.o: %.cpp
	$(NVCC) $(CCFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(CCFLAGS) -c $< -o $@

# Create static library
$(LIB): $(OBJS)
	$(NVCC) -lib -o $@ $^

# Clean up
clean:
	rm -f $(OBJS) $(LIB)
