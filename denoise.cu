#include "denoise.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>


std::string to_string(DenoiseMethod method) {
    switch (method) {
    case IDENTITY:      return "identity";
    case BOX_FILTER:    return "box";
    case GAUSSIAN:      return "gaussian";
    case MEDIAN:        return "median";
    case BILATERAL:     return "bilateral";
    case NLM:           return "nlm";
    default:            return "unknown";
    }
}

__constant__ float kernel_3[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

__constant__ float kernel_5[5][5] = {
    {0.003663f, 0.014652f, 0.023173f, 0.014652f, 0.003663f},
    {0.014652f, 0.058608f, 0.092103f, 0.058608f, 0.014652f},
    {0.023173f, 0.092103f, 0.144448f, 0.092103f, 0.023173f},
    {0.014652f, 0.058608f, 0.092103f, 0.058608f, 0.014652f},
    {0.003663f, 0.014652f, 0.023173f, 0.014652f, 0.003663f}
};

__constant__ float kernel_9[9][9] = {
    {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00024404, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
    {0.00002292, 0.00078634, 0.00655965, 0.01331827, 0.00838894, 0.01331827, 0.00655965, 0.00078634, 0.00002292},
    {0.00019117, 0.00655965, 0.05472157, 0.11156508, 0.07025366, 0.11156508, 0.05472157, 0.00655965, 0.00019117},
    {0.00038771, 0.01331827, 0.11156508, 0.22749645, 0.14323822, 0.22749645, 0.11156508, 0.01331827, 0.00038771},
    {0.00024404, 0.00838894, 0.07025366, 0.14323822, 0.09037601, 0.14323822, 0.07025366, 0.00838894, 0.00024404},
    {0.00038771, 0.01331827, 0.11156508, 0.22749645, 0.14323822, 0.22749645, 0.11156508, 0.01331827, 0.00038771},
    {0.00019117, 0.00655965, 0.05472157, 0.11156508, 0.07025366, 0.11156508, 0.05472157, 0.00655965, 0.00019117},
    {0.00002292, 0.00078634, 0.00655965, 0.01331827, 0.00838894, 0.01331827, 0.00655965, 0.00078634, 0.00002292},
    {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00024404, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
};


__global__ void kernel_identity(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = input[idx]; //(no actual denoise)
}

__global__ void kernel_box_filter(const float* input, float* output, int width, int height, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;

    for (int dy = -window; dy <= window; ++dy)
        for (int dx = -window; dx <= window; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }

    output[y * width + x] = sum / count;
}

__global__ void kernel_gaussian_filter(const float* input, float* output, int width, int height, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weightSum = 0.0f;

    for (int dy = -window; dy <= window; ++dy)
        for (int dx = -window; dx <= window; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                float val = input[ny * width + nx];

                float weight;
                if (window == 1)
                    weight= kernel_3[dy + window][dx + window];
                if (window == 2)
                    weight= kernel_5[dy + window][dx + window];
                if (window == 4)
                    weight= kernel_9[dy + window][dx + window];
                sum += val * weight;
                weightSum += weight;
            }
        }

    output[y * width + x] = sum / weightSum;
}

__global__ void kernel_median_filter(const float* input, float* output, int width, int height, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    //should be (window*2+1)*(window*2+1)
    float values[81];
    int count = 0;

    for (int dy = -window; dy <= window; ++dy)
        for (int dx = -window; dx <= window; ++dx) {
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

__global__ void kernel_bilateral_filter(const float* input, float* output, int width, int height, float sigma_s, float sigma_r, int window) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float center = input[idx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dy = -window; dy <= window; ++dy) {
        for (int dx = -window; dx <= window; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                float neighbor = input[ny * width + nx];

                float spatial_dist2 = dx * dx + dy * dy;
                float intensity_diff = neighbor - center;
                float intensity_diff2 = intensity_diff * intensity_diff;

                float weight = expf(-spatial_dist2 / (2 * sigma_s * sigma_s) - intensity_diff2 / (2 * sigma_r * sigma_r));

                sum += neighbor * weight;
                weight_sum += weight;
            }
        }
    }

    output[idx] = sum / weight_sum;
}

__global__ void kernel_nlm_filter(const float* input, float* output, int width, int height, float h, int patch_radius,
                                  int search_radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int center_idx = y * width + x;

    float weight_sum = 0.0f;
    float result = 0.0f;

    for (int dy = -search_radius; dy <= search_radius; ++dy) {
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            int sx = x + dx;
            int sy = y + dy;

            if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                continue;

            float dist2 = 0.0f;

            for (int py = -patch_radius; py <= patch_radius; ++py) {
                for (int px = -patch_radius; px <= patch_radius; ++px) {
                    int cx = x + px;
                    int cy = y + py;
                    int qx = sx + px;
                    int qy = sy + py;

                    if (cx >= 0 && cy >= 0 && cx < width && cy < height &&
                        qx >= 0 && qy >= 0 && qx < width && qy < height) {

                        float diff = input[cy * width + cx] - input[qy * width + qx];
                        dist2 += diff * diff;
                        }
                }
            }

            float weight = expf(-dist2 / (h * h));
            result += input[sy * width + sx] * weight;
            weight_sum += weight;
        }
    }

    output[center_idx] = result / (weight_sum + 1e-12f); //prevent div by 0
}

__global__ void kernel_identity_3D(const float* input, float* output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * height * width + y * width + x;
    output[idx] = input[idx];
}

__global__ void kernel_box_filter_3D(const float* input, float* output, int width, int height, int depth, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    float sum = 0.0f;
    int count = 0;

    for (int dz = -window; dz <= window; ++dz) {
        for (int dy = -window; dy <= window; ++dy) {
            for (int dx = -window; dx <= window; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    sum += input[nz * width * height + ny * width + nx];
                    count++;
                }
            }
        }
    }

    output[z * width * height + y * width + x] = sum / count;
}

__global__ void kernel_gaussian_filter_3D(const float* input, float* output,
                                          int width, int height, int depth, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    float sum = 0.0f;
    float weightSum = 0.0f;

    for (int dz = -window; dz <= window; ++dz) {
        for (int dy = -window; dy <= window; ++dy) {
            for (int dx = -window; dx <= window; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    float val = input[nz * height * width + ny * width + nx];

                    // Compute 3D Gaussian weight
                    float distanceSq = dx * dx + dy * dy + dz * dz;
                    float sigma = float(window);  // adjust as needed
                    float weight = expf(-distanceSq / (2.0f * sigma * sigma));

                    sum += val * weight;
                    weightSum += weight;
                }
            }
        }
    }

    output[z * height * width + y * width + x] = sum / weightSum;
}


__global__ void kernel_median_filter_3D(const float* input, float* output, int width, int height, int depth, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;
    //should be (window*2+1)*(window*2+1)*(window*2+1)
    float values[729];
    int count = 0;

    for (int dz = -window; dz <= window; ++dz) {
        for (int dy = -window; dy <= window; ++dy) {
            for (int dx = -window; dx <= window; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    values[count++] = input[nz * width * height + ny * width + nx];
                }
            }
        }
    }

    // Simple bubble sort for small count
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (values[j] > values[j + 1]) {
                float tmp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = tmp;
            }
        }
    }

    output[z * width * height + y * width + x] = values[count / 2];
}

__global__ void kernel_bilateral_filter_3D(const float* input, float* output,
                                           int width, int height, int depth,
                                           float sigma_s, float sigma_r, int window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * width * height + y * width + x;
    float center_val = input[center_idx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dz = -window; dz <= window; ++dz) {
        for (int dy = -window; dy <= window; ++dy) {
            for (int dx = -window; dx <= window; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    int neighbor_idx = nz * width * height + ny * width + nx;
                    float neighbor_val = input[neighbor_idx];

                    float spatial_dist2 = dx * dx + dy * dy + dz * dz;
                    float intensity_diff2 = (neighbor_val - center_val) * (neighbor_val - center_val);

                    float weight = expf(-spatial_dist2 / (2.0f * sigma_s * sigma_s)
                                        - intensity_diff2 / (2.0f * sigma_r * sigma_r));

                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    output[center_idx] = sum / weight_sum;
}

__global__ void kernel_nlm_filter_3D(const float* input, float* output,
                                     int width, int height, int depth,
                                     float h, int search_radius, int patch_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * width * height + y * width + x;

    float weight_sum = 0.0f;
    float result = 0.0f;

    for (int dz = -search_radius; dz <= search_radius; ++dz) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int sx = x + dx;
                int sy = y + dy;
                int sz = z + dz;

                if (sx < 0 || sy < 0 || sz < 0 || sx >= width || sy >= height || sz >= depth)
                    continue;

                float dist2 = 0.0f;

                for (int pz = -patch_radius; pz <= patch_radius; ++pz) {
                    for (int py = -patch_radius; py <= patch_radius; ++py) {
                        for (int px = -patch_radius; px <= patch_radius; ++px) {
                            int cx = x + px, cy = y + py, cz = z + pz;
                            int qx = sx + px, qy = sy + py, qz = sz + pz;

                            if (cx < 0 || cy < 0 || cz < 0 || cx >= width || cy >= height || cz >= depth ||
                                qx < 0 || qy < 0 || qz < 0 || qx >= width || qy >= height || qz >= depth)
                                continue;

                            int c_idx = cz * width * height + cy * width + cx;
                            int q_idx = qz * width * height + qy * width + qx;

                            float diff = input[c_idx] - input[q_idx];
                            dist2 += diff * diff;
                        }
                    }
                }

                float weight = expf(-dist2 / (h * h));
                int s_idx = sz * width * height + sy * width + sx;
                result += input[s_idx] * weight;
                weight_sum += weight;
            }
        }
    }

    output[center_idx] = result / (weight_sum + 1e-12f); // prevent div by 0
}

void denoise(const float* input, float* output, int width, int height, DenoiseMethod method) {
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    switch (method) {
        case IDENTITY:
            kernel_identity<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case BOX_FILTER:
            kernel_box_filter<<<blocks, threads>>>(d_input, d_output, width, height, window);
            break;

        case GAUSSIAN:
            kernel_gaussian_filter<<<blocks, threads>>>(d_input, d_output, width, height, window);
            break;

        case MEDIAN:
            kernel_median_filter<<<blocks, threads>>>(d_input, d_output, width, height, window);
            break;

        case BILATERAL:
            kernel_bilateral_filter<<<blocks, threads>>>(d_input, d_output, width, height, sigma_s, sigma_r, window);
            break;

        case NLM:
            kernel_nlm_filter<<<blocks, threads>>>(d_input, d_output, width, height, h, patch_radius, search_radius);
            break;

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << std::endl;

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}


void denoise3D(const float* input, float* output, int width, int height, int depth, DenoiseMethod method) {
    float *d_input, *d_output;
    size_t size = width * height * depth * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(8,8,8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    switch (method) {
    case IDENTITY:
        kernel_identity_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
        break;

    case BOX_FILTER:
        kernel_box_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth, window);
        break;

    case GAUSSIAN:
        kernel_gaussian_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth, window);
        break;

    case MEDIAN:
        kernel_median_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth, window);
        break;

    case BILATERAL:
        kernel_bilateral_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth, sigma_s, sigma_r, window);
        break;

    case NLM:
        kernel_nlm_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth, h, patch_radius, search_radius);
        break;

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Denoise method " << to_string(method) << " took " << milliseconds << " ms" << std::endl;

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
