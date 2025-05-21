#include "denoise.cuh"
#include <cuda_runtime.h>
#include <algorithm>


std::string to_string(DenoiseMethod method) {
    switch (method) {
    case IDENTITY:      return "identity";
    case BOX_FILTER:    return "box";
    default:            return "unknown";
    }
}

__global__ void kernel_identity(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = input[idx]; // Identity (no actual denoise yet)
}

__global__ void kernel_box_filter(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }

    output[y * width + x] = sum / count;
}


void denoise(const float* input, float* output, int width, int height, DenoiseMethod method) {
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    switch (method) {
        case IDENTITY:
            kernel_identity<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case BOX_FILTER:
            kernel_box_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

    }

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
