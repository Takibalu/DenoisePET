#include "denoise.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>



std::string to_string(DenoiseMethod method) {
    switch (method) {
    case IDENTITY:          return "identity";
    case BOX_FILTER:        return "box";
    case GAUSSIAN:          return "gaussian";
    case MEDIAN:            return "median";
    case BILATERAL:         return "bilateral";
    case NLM:               return "nlm";
    case JOINT_BILATERAL:   return "joint_bilateral";
    case JOINT_NLM:         return "joint_nlm";
    default:                return "unknown";
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

__global__ void kernel_box_filter(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;

    for (int dy = -WINDOW; dy <= WINDOW; ++dy)
        for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
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

    float sum = 0.0f;
    float weightSum = 0.0f;

    for (int dy = -WINDOW; dy <= WINDOW; ++dy)
        for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                float val = input[ny * width + nx];

                float weight;
                if (WINDOW == 1)
                    weight= kernel_3[dy + WINDOW][dx + WINDOW];
                if (WINDOW == 2)
                    weight= kernel_5[dy + WINDOW][dx + WINDOW];
                if (WINDOW == 4)
                    weight= kernel_9[dy + WINDOW][dx + WINDOW];
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
    //should be (window*2+1)*(window*2+1)
    float values[81];
    int count = 0;

    for (int dy = -WINDOW; dy <= WINDOW; ++dy)
        for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
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

__global__ void kernel_bilateral_filter(const float* input, float* output, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float center = input[idx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dy = -WINDOW; dy <= WINDOW; ++dy) {
        for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                float neighbor = input[ny * width + nx];

                float spatial_dist2 = dx * dx + dy * dy;
                float intensity_diff = neighbor - center;
                float intensity_diff2 = intensity_diff * intensity_diff;

                float weight = expf(-spatial_dist2 / (2 * SIGMA_SPATIAL * SIGMA_SPATIAL) - intensity_diff2 / (2 * SIGMA_INTENSITY * SIGMA_INTENSITY));

                sum += neighbor * weight;
                weight_sum += weight;
            }
        }
    }

    output[idx] = sum / weight_sum;
}

__global__ void kernel_nlm_filter(const float* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int center_idx = y * width + x;

    float weight_sum = 0.0f;
    float result = 0.0f;

    for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; ++dy) {
        for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; ++dx) {
            int sx = x + dx;
            int sy = y + dy;

            if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                continue;

            float dist2 = 0.0f;

            for (int py = -PATCH_RADIUS; py <= PATCH_RADIUS; ++py) {
                for (int px = -PATCH_RADIUS; px <= PATCH_RADIUS; ++px) {
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

            float weight = expf(-dist2 / (FILTER_POWER * FILTER_POWER));
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

__global__ void kernel_box_filter_3D(const float* input, float* output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    float sum = 0.0f;
    int count = 0;

    for (int dz = -WINDOW; dz <= WINDOW; ++dz) {
        for (int dy = -WINDOW; dy <= WINDOW; ++dy) {
            for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
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

__global__ void kernel_gaussian_filter_3D(const float* input, float* output, int width, int height, int depth){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    float sum = 0.0f;
    float weightSum = 0.0f;

    for (int dz = -WINDOW; dz <= WINDOW; ++dz) {
        for (int dy = -WINDOW; dy <= WINDOW; ++dy) {
            for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    float val = input[nz * height * width + ny * width + nx];

                    // Compute 3D Gaussian weight
                    float distanceSq = dx * dx + dy * dy + dz * dz;
                    float sigma = float(WINDOW);  // adjust as needed
                    float weight = expf(-distanceSq / (2.0f * sigma * sigma));

                    sum += val * weight;
                    weightSum += weight;
                }
            }
        }
    }

    output[z * height * width + y * width + x] = sum / weightSum;
}

__global__ void kernel_median_filter_3D(const float* input, float* output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;
    //should be (window*2+1)*(window*2+1)*(window*2+1)
    float values[729];
    int count = 0;

    for (int dz = -WINDOW; dz <= WINDOW; ++dz) {
        for (int dy = -WINDOW; dy <= WINDOW; ++dy) {
            for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
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

__global__ void kernel_bilateral_filter_3D(const float* input, float* output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * width * height + y * width + x;
    float center_val = input[center_idx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dz = -WINDOW; dz <= WINDOW; ++dz) {
        for (int dy = -WINDOW; dy <= WINDOW; ++dy) {
            for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    int neighbor_idx = nz * width * height + ny * width + nx;
                    float neighbor_val = input[neighbor_idx];

                    float spatial_dist2 = dx * dx + dy * dy + dz * dz;
                    float intensity_diff2 = (neighbor_val - center_val) * (neighbor_val - center_val);

                    float weight = expf(-spatial_dist2 / (2.0f * SIGMA_SPATIAL * SIGMA_SPATIAL)
                                        - intensity_diff2 / (2.0f * SIGMA_INTENSITY * SIGMA_INTENSITY));

                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    output[center_idx] = sum / weight_sum;
}

__global__ void kernel_nlm_filter_3D(const float* input, float* output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * width * height + y * width + x;

    float weight_sum = 0.0f;
    float result = 0.0f;

    for (int dz = -SEARCH_RADIUS; dz <= SEARCH_RADIUS; ++dz) {
        int sz = z + dz;
        if (sz < 0 || sz >= depth) continue;

        for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; ++dy) {
            int sy = y + dy;
            if (sy < 0 || sy >= height) continue;

            for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; ++dx) {
                int sx = x + dx;
                if (sx < 0 || sx >= width) continue;

                float dist2 = 0.0f;

                for (int pz = -PATCH_RADIUS; pz <= PATCH_RADIUS; ++pz) {
                    int cz = z + pz;
                    int qz = sz + pz;
                    if (cz < 0 || cz >= depth || qz < 0 || qz >= depth) continue;

                    for (int py = -PATCH_RADIUS; py <= PATCH_RADIUS; ++py) {
                        int cy = y + py;
                        int qy = sy + py;
                        if (cy < 0 || cy >= depth || qy < 0 || qy >= depth) continue;

                        for (int px = -PATCH_RADIUS; px <= PATCH_RADIUS; ++px) {
                            int cx = x + px;
                            int qx = sx + px;
                            if (cx < 0 || cx >= depth || qx < 0 || qx >= depth) continue;

                            int c_idx = cz * width * height + cy * width + cx;
                            int q_idx = qz * width * height + qy * width + qx;

                            float diff = input[c_idx] - input[q_idx];
                            dist2 += diff * diff;
                        }
                    }
                }

                float weight = expf(-dist2 / (FILTER_POWER * FILTER_POWER));
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
            kernel_box_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case GAUSSIAN:
            kernel_gaussian_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case MEDIAN:
            kernel_median_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case BILATERAL:
            kernel_bilateral_filter<<<blocks, threads>>>(d_input, d_output, width, height);
            break;

        case NLM:
            kernel_nlm_filter<<<blocks, threads>>>(d_input, d_output, width, height);
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
    dim3 blocks((width + threads.x - 1) / threads.x,
           (height + threads.y - 1) / threads.y,
           (depth + threads.z - 1) / threads.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    switch (method) {
    case IDENTITY:
        kernel_identity_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
        break;

    case BOX_FILTER:
        kernel_box_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
        break;

    case GAUSSIAN:
        kernel_gaussian_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
        break;

    case MEDIAN:
        kernel_median_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
        break;

    case BILATERAL:
        kernel_bilateral_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
        break;

    case NLM:
        kernel_nlm_filter_3D<<<blocks, threads>>>(d_input, d_output, width, height, depth);
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

__global__ void joint_bilateral_kernel(const float* pet, const float* ct, float* output,
                                       int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * width * height + y * width + x;
    float center_ct = ct[center_idx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dz = -WINDOW; dz <= WINDOW; ++dz) {
        int zz = z + dz;
        if (zz < 0 || zz >= depth) continue;

        for (int dy = -WINDOW; dy <= WINDOW; ++dy) {
            int yy = y + dy;
            if (yy < 0 || yy >= height) continue;

            for (int dx = -WINDOW; dx <= WINDOW; ++dx) {
                int xx = x + dx;
                if (xx < 0 || xx >= width) continue;

                int neighbor_idx = zz * width * height + yy * width + xx;
                float neighbor_ct = ct[neighbor_idx];
                float neighbor_pet = pet[neighbor_idx];

                float spatial_dist2 = dx * dx + dy * dy + dz * dz;
                float intensity_dist2 = (center_ct - neighbor_ct) * (center_ct - neighbor_ct);

                float weight = expf(-spatial_dist2 / (2.0f * SIGMA_SPATIAL * SIGMA_SPATIAL)
                               -intensity_dist2 / (2.0f * SIGMA_INTENSITY * SIGMA_INTENSITY));
                sum += neighbor_pet * weight;
                weight_sum += weight;
            }
        }
    }

    output[center_idx] = (weight_sum > 0.0f) ? sum / weight_sum : pet[center_idx];
}

__global__ void joint_nlm_kernel(const float* pet, const float* ct, float* output,
                                 int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * width * height + y * width + x;

    float result = 0.0f;
    float weight_sum = 0.0f;

    for (int dz = -SEARCH_RADIUS; dz <= SEARCH_RADIUS; ++dz) {
        int sz = z + dz;
        if (sz < PATCH_RADIUS || sz >= depth - PATCH_RADIUS) continue;

        for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; ++dy) {
            int sy = y + dy;
            if (sy < PATCH_RADIUS || sy >= height - PATCH_RADIUS) continue;

            for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; ++dx) {
                int sx = x + dx;
                if (sx < PATCH_RADIUS || sx >= width - PATCH_RADIUS) continue;

                float dist2 = 0.0f;

                for (int pz = -PATCH_RADIUS; pz <= PATCH_RADIUS; ++pz) {
                    int cz = z + pz;
                    int qz = sz + pz;
                    if (cz < 0 || cz >= depth || qz < 0 || qz >= depth) continue;

                    for (int py = -PATCH_RADIUS; py <= PATCH_RADIUS; ++py) {
                        int cy = y + py;
                        int qy = sy + py;
                        if (cy < 0 || cy >= depth || qy < 0 || qy >= depth) continue;

                        for (int px = -PATCH_RADIUS; px <= PATCH_RADIUS; ++px) {
                            int cx = x + px;
                            int qx = sx + px;
                            if (cx < 0 || cx >= depth || qx < 0 || qx >= depth) continue;

                            int c_idx = cz * width * height + cy * width + cx;
                            int q_idx = qz * width * height + qy * width + qx;

                            float diff = ct[c_idx] - ct[q_idx];
                            dist2 += diff * diff;
                        }
                    }
                }

                float weight = expf(-dist2 / (FILTER_POWER * FILTER_POWER));
                int s_idx = sz * width * height + sy * width + sx;
                result += pet[s_idx] * weight;
                weight_sum += weight;
            }
        }
    }

    output[center_idx] = (weight_sum > 0.0f) ? result / weight_sum : pet[center_idx];
}


void denoise3D_joint(const float* pet, const float* ct, float* output, int width, int height, int depth, DenoiseMethod method) {
    float *d_pet, *d_ct, *d_output;
    size_t size = width * height * depth * sizeof(float);

    cudaMalloc(&d_pet, size);
    cudaMalloc(&d_ct, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_pet, pet, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ct, ct, size, cudaMemcpyHostToDevice);

    dim3 threads(8,8,8);
    dim3 blocks((width + threads.x - 1) / threads.x,
           (height + threads.y - 1) / threads.y,
           (depth + threads.z - 1) / threads.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    switch (method) {
    case JOINT_BILATERAL:
        joint_bilateral_kernel<<<blocks, threads>>>(d_pet, d_ct, d_output, width, height, depth);
        break;
    case JOINT_NLM:
        joint_nlm_kernel<<<blocks, threads>>>(d_pet, d_ct, d_output, width, height, depth);
        break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Denoise method " << to_string(method) << " took " << milliseconds << " ms" << std::endl;

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pet);
    cudaFree(d_ct);
    cudaFree(d_output);
}
