# Compilers and flags
NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3  # Flags for NVCC (GPU)

GPP = g++
GPP_FLAGS = -O3              # Flags for G++ (CPU)

# Target executable names
TARGET_GPU = convolution_gpu
TARGET_CPU = convolution_cpu

# Default rule - build both
all: $(TARGET_GPU) $(TARGET_CPU)

# Rule to build GPU program
$(TARGET_GPU): convolution.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET_GPU) convolution.cu

# Rule to build CPU program
$(TARGET_CPU): convolution_cpu.cpp
	$(GPP) $(GPP_FLAGS) -o $(TARGET_CPU) convolution_cpu.cpp

# Run the GPU program
run_gpu: $(TARGET_GPU)
	./$(TARGET_GPU)

# Run the CPU program
run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU)

# Clean up everything
clean:
	rm -f $(TARGET_GPU) $(TARGET_CPU) *.o

# Declare phony targets
.PHONY: all run_gpu run_cpu clean
