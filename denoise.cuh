#pragma once

enum DenoiseMethod {
    BOX_FILTER,
};

void denoise(const float* input, float* output, int width, int height, DenoiseMethod method);
void run_denoising(int width, int height, DenoiseMethod method);
