#include "denoise.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>


std::string to_string(DenoiseMethod method) {
    switch (method) {
    case IDENTITY:      return "identity";
    case BOX_FILTER:    return "box";
    case GAUSSIAN:      return "gaussian";
    case MEDIAN:        return "median";
    case BILATERAL:     return "bilateral";
    default:            return "unknown";
    }
}


__global__ void kernel_identity(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = input[idx]; //(no actual denoise)
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

__global__ void kernel_gaussian_filter(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    float sum = 0.0f;
    float weightSum = 0.0f;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                float val = input[ny * width + nx];
                float weight = kernel[dy + 1][dx + 1];
                sum += val * weight;
                weightSum += weight;
            }
        }

    output[y * width + x] = sum / weightSum;
}

__global__ void kernel_median_filter(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float values[9];
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                values[count++] = input[ny * width + nx];
            }
        }

    // Bubble sort 9 elements
    for (int i = 0; i < count - 1; ++i)
        for (int j = 0; j < count - i - 1; ++j)
            if (values[j] > values[j + 1]) {
                float tmp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = tmp;
            }

    output[y * width + x] = values[count / 2];
}

__global__ void kernel_bilateral_filter(const float* input, float* output, int width, int height, float sigma_s, float sigma_r) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float center = input[idx];

    float sum = 0.0f;
    float wsum = 0.0f;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                float neighbor = input[ny * width + nx];

                float spatial_dist2 = dx * dx + dy * dy;
                float intensity_diff = neighbor - center;
                float intensity_diff2 = intensity_diff * intensity_diff;

                float w = expf(-spatial_dist2 / (2 * sigma_s * sigma_s) - intensity_diff2 / (2 * sigma_r * sigma_r));

                sum += neighbor * w;
                wsum += w;
            }
        }
    }

    output[idx] = sum / wsum;
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

        case GAUSSIAN:
            kernel_gaussian_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case MEDIAN:
            kernel_median_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case BILATERAL:
            float sigma_s = 75.0f;  // térbeli szórás
            float sigma_r = 75.0f;  // intenzitásbeli szórás
            kernel_bilateral_filter<<<blocks, threads>>>(d_input, d_output, width, height, sigma_s, sigma_r);
            break;


    }

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
