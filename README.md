# CUDA 2D Convolution

A high-performance implementation of 2D convolution for NVIDIA GPUs, leveraging CUDA's parallel computing architecture, shared memory for data reuse, and constant memory for fast mask access. This project demonstrates core optimization techniques used in image processing and deep learning.

##  Overview

Convolution is a fundamental operation in signal and image processing (e.g., blurring, sharpening, edge detection) and is the core building block of Convolutional Neural Networks (CNNs). This implementation optimizes this operation by:
*   **Parallelization:** Distributing the computational workload across hundreds of GPU threads.
*   **Shared Memory:** Utilizing on-chip shared memory to minimize expensive global memory accesses and reduce bandwidth bottlenecks.
*   **Constant Memory:** Storing the small, read-only convolution mask/kernel in constant memory for cached, high-speed access.

##  Architecture & Optimization

### Kernel Design: `convo`
The CUDA kernel is designed to efficiently compute convolution for each output element.

1.  **Tiling:** Each thread block processes a tile of the input matrix of size `BLOCK_DIM x BLOCK_DIM`.
2.  **Halo Loading:** To compute the convolution for border pixels of the tile, each thread block loads a larger shared memory array (`[BLOCK_DIM + MASK_DIM - 1]` squared), including a "halo" region from neighboring tiles.
3.  **Shared Memory:** Threads collaboratively load data from global memory into fast shared memory. This allows data reused by multiple threads (e.g., pixels in the overlapping halo regions) to be fetched only once.
4.  **Constant Memory:** The convolution mask is stored in `__constant__` memory, which is cached for broadcast to all threads, making accesses extremely fast.

### Key Parameters
*   `MASK_DIM`: Dimension of the square convolution mask (e.g., 3 for a 3x3 kernel).
*   `BLOCK_DIM`: Dimension of the square thread block (e.g., 4, 8, 16, 32). Optimal size depends on GPU architecture.

##  How to Build and Run

### Prerequisites
*   **NVIDIA GPU** with Compute Capability 3.5+.
*   **NVIDIA CUDA Toolkit** (version 11.x recommended).
*   A C++ compiler (like `g++`).

### Compilation and Execution
1.  Compile the source code using `nvcc`, the CUDA compiler driver:
    ```bash
    nvcc -o convolution convolution.cu
    ```
2.  Run the generated executable:
    ```bash
    ./convolution
    ```

## 📊 Example Output

The program will:
1.  Generate a random 8x8 input matrix.
2.  Use a pre-defined 3x3 Laplacian edge-detection mask.
3.  Perform the convolution on the GPU.
4.  Print the original input, the mask, and the convolved output to the console.
