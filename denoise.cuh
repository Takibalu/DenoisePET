#pragma once
#include <string>

enum DenoiseMethod {
    IDENTITY,
    BOX_FILTER,
    GAUSSIAN,
    MEDIAN,
};
std::string to_string(DenoiseMethod method);
void denoise(const float* input, float* output, int width, int height, DenoiseMethod method);
void run_denoising(int width, int height, DenoiseMethod method);
