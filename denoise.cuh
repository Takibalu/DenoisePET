#pragma once
#include <string>

enum DenoiseMethod {
    IDENTITY,
    BOX_FILTER,
    GAUSSIAN,
    MEDIAN,
    BILATERAL,
    NLM,
};
std::string to_string(DenoiseMethod method);
void denoise(const float* input, float* output, int width, int height, DenoiseMethod method);
void denoise3D(const float* input, float* output, int width, int height, int depth, DenoiseMethod method);
void run_denoising(int width, int height, DenoiseMethod method);
void run_denoising3D(int width, int height, int depth, DenoiseMethod method);
