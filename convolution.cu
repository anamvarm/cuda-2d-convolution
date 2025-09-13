/**
 * @file convolution.cu
 * @brief CUDA implementation of 2D convolution with shared and constant memory.
 * @details
 * This code demonstrates optimizing a 2D convolution operation for a GPU.
 * Key optimizations include:
 *   - Using __shared__ memory to tile data and reduce global memory accesses.
 *   - Using __constant__ memory to cache the convolution mask for all threads.
 *   - Handling halo regions for correct tiled convolution.
 *
 * The example uses a Laplacian filter for edge detection.
 */

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define MASK_DIM 3       // Dimension of the convolution mask (3x3)
#define BLOCK_DIM 4      // Dimension of the thread block (4x4 threads per block)

// Constant memory declaration for the convolution mask - cached for all threads
__constant__ float mask_mat[MASK_DIM * MASK_DIM];

/**
 * @brief CUDA kernel for 2D convolution using shared memory
 * @param input_mat Pointer to input matrix in global memory
 * @param output_mat Pointer to output matrix in global memory
 * @param mat_width Width of the square input/output matrix
 * @param mask_width Width of the square convolution mask
 * 
 * @details Each thread block processes a tile of the input matrix and utilizes
 * shared memory to minimize global memory accesses. The kernel handles halo
 * regions for correct convolution at tile boundaries.
 */
__global__ void convo(float* input_mat, float* output_mat, int mat_width, int mask_width) {
    // Shared memory declaration for tile data + halo regions
    __shared__ float shared_mem[BLOCK_DIM + MASK_DIM - 1][BLOCK_DIM + MASK_DIM - 1];

    // Thread indices within block
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    // Global indices within the entire matrix
    int global_x = blockIdx.x * BLOCK_DIM + thread_x;
    int global_y = blockIdx.y * BLOCK_DIM + thread_y;
    
    // Halo size (number of pixels needed from neighboring tiles)
    int halo = mask_width / 2;

    // Position in shared memory (offset by halo size)
    int shared_x = thread_x + halo;
    int shared_y = thread_y + halo;

    // Load main tile data into shared memory
    if (global_x < mat_width && global_y < mat_width) {
        shared_mem[shared_y][shared_x] = input_mat[global_y * mat_width + global_x];
    } else {
        // Pad with zeros if out of bounds
        shared_mem[shared_y][shared_x] = 0.0f;
    }
    
    // Synchronize to ensure all threads have loaded their data
    __syncthreads();

    // Only threads that correspond to valid output elements perform computation
    if (thread_x < BLOCK_DIM && thread_y < BLOCK_DIM && 
        global_x < mat_width && global_y < mat_width) {
        
        float accumulation = 0.0f;
        
        // Perform convolution using the shared memory tile
        for (int i = -halo; i <= halo; ++i) {
            for (int j = -halo; j <= halo; ++j) {
                int shared_i = shared_y + i;
                int shared_j = shared_x + j;
                
                // Ensure we're within shared memory bounds
                if (shared_i >= 0 && shared_i < BLOCK_DIM + MASK_DIM - 1 && 
                    shared_j >= 0 && shared_j < BLOCK_DIM + MASK_DIM - 1) {
                    
                    // Accumulate: input value * mask weight
                    accumulation += shared_mem[shared_i][shared_j] * 
                                   mask_mat[(i + halo) * mask_width + (j + halo)];
                }
            }
        }
        
        // Write result to global memory
        output_mat[global_y * mat_width + global_x] = accumulation;
    }
}

/**
 * @brief Helper function to copy mask from host to constant memory
 * @param host_mask Pointer to host memory containing the mask values
 */
void uploadMaskToConstantMemory(float* host_mask) {
    cudaMemcpyToSymbol(mask_mat, host_mask, MASK_DIM * MASK_DIM * sizeof(float));
}

/**
 * @brief Main function demonstrating 2D convolution
 */
int main() {
    // Matrix and mask dimensions
    int mat_size = 8;
    int mask_size = MASK_DIM;
    int total_bytes = mat_size * mat_size * sizeof(float);
    
    // Memory pointers
    float *host_input, *host_output, *device_input, *device_output;
    
    // Laplacian edge detection mask (3x3)
    float host_mask[MASK_DIM * MASK_DIM] = { 
        0.0f,  1.0f,  0.0f,
        1.0f, -4.0f,  1.0f,
        0.0f,  1.0f,  0.0f 
    };

    // Allocate host memory
    host_input = (float*)malloc(total_bytes);
    host_output = (float*)malloc(total_bytes);
    
    if (!host_input || !host_output) {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    // Initialize input matrix with random values
    for (int i = 0; i < mat_size * mat_size; i++) {
        host_input[i] = (float)(rand() % 10);
    }

    // Print input matrix
    printf("Input Matrix (%dx%d):\n", mat_size, mat_size);
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            printf("%6.2f ", host_input[i * mat_size + j]);
        }
        printf("\n");
    }

    // Print mask matrix
    printf("\nMask Matrix (%dx%d - Laplacian):\n", MASK_DIM, MASK_DIM);
    for (int i = 0; i < MASK_DIM; i++) {
        for (int j = 0; j < MASK_DIM; j++) {
            printf("%6.2f ", host_mask[i * MASK_DIM + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc((void**)&device_input, total_bytes);
    if (err != cudaSuccess) {
        printf("CUDA device input allocation failed: %s\n", cudaGetErrorString(err));
        free(host_input);
        free(host_output);
        return 1;
    }
    
    err = cudaMalloc((void**)&device_output, total_bytes);
    if (err != cudaSuccess) {
        printf("CUDA device output allocation failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_input);
        free(host_input);
        free(host_output);
        return 1;
    }

    // Copy input data to device
    err = cudaMemcpy(device_input, host_input, total_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_input);
        cudaFree(device_output);
        free(host_input);
        free(host_output);
        return 1;
    }

    // Upload mask to constant memory
    uploadMaskToConstantMemory(host_mask);

    // Configure kernel launch parameters
    dim3 block_dimensions(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dimensions((mat_size + BLOCK_DIM - 1) / BLOCK_DIM, 
                         (mat_size + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch convolution kernel
    printf("Launching kernel with grid (%d, %d) and block (%d, %d)...\n",
           grid_dimensions.x, grid_dimensions.y, 
           block_dimensions.x, block_dimensions.y);
    
    convo<<<grid_dimensions, block_dimensions>>>(device_input, device_output, mat_size, mask_size);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_input);
        cudaFree(device_output);
        free(host_input);
        free(host_output);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(host_output, device_output, total_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_input);
        cudaFree(device_output);
        free(host_input);
        free(host_output);
        return 1;
    }

    // Print resulting matrix
    printf("Convolved Output Matrix:\n");
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            printf("%6.2f ", host_output[i * mat_size + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);

    printf("\nConvolution completed successfully!\n");
    return 0;
}
