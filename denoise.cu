#include "denoise.cuh"
#include <cuda_runtime.h>

__global__ void kernel_denoise(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = input[idx]; // Identity (no actual denoise yet)
}

void denoise(const float* input, float* output, int width, int height) {
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    kernel_denoise<<<blocks, threads>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
