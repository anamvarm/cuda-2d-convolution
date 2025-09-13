#include <iostream>
#include <stdlib.h>

void convolution_cpu(float* input, float* output, float* mask, int mat_width, int mask_width) {
    int halo = mask_width / 2;
    
    for (int y = 0; y < mat_width; y++) {
        for (int x = 0; x < mat_width; x++) {
            float sum = 0.0f;
            
            for (int my = 0; my < mask_width; my++) {
                for (int mx = 0; mx < mask_width; mx++) {
                    // Calculate input indices, handling borders by clamping
                    int in_x = std::min(std::max(x + mx - halo, 0), mat_width - 1);
                    int in_y = std::min(std::max(y + my - halo, 0), mat_width - 1);
                    
                    sum += input[in_y * mat_width + in_x] * mask[my * mask_width + mx];
                }
            }
            output[y * mat_width + x] = sum;
        }
    }
}

int main() {
    int mat_size = 8;
    int mask_size = 3;
    int total_elements = mat_size * mat_size;
    int total_bytes = total_elements * sizeof(float);
    
    float host_mask[9] = { 0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f };
    float *host_input, *host_output_cpu;
    
    host_input = (float*)malloc(total_bytes);
    host_output_cpu = (float*)malloc(total_bytes);
    
    // Initialize with the same random values as the GPU version for fair comparison
    srand(123); // Seed for reproducible results
    for (int i = 0; i < total_elements; i++) {
        host_input[i] = (float)(rand() % 10);
    }
    
    // Run CPU convolution
    convolution_cpu(host_input, host_output_cpu, host_mask, mat_size, mask_size);
    
    // Print results
    std::cout << "CPU Result Matrix:" << std::endl;
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            std::cout << host_output_cpu[i * mat_size + j] << " ";
        }
        std::cout << std::endl;
    }
    
    free(host_input);
    free(host_output_cpu);
    return 0;
}
