#pragma once
#include <string>
#include <vector>

enum DenoiseMethod {
    IDENTITY,
    BOX_FILTER,
    GAUSSIAN,
    MEDIAN,
    BILATERAL,
    NLM,
    JOINT_BILATERAL,
    JOINT_NLM
};
std::string to_string(DenoiseMethod method);
void denoise(const float* input, float* output, int width, int height, DenoiseMethod method);
void denoise3D(const float* input, float* output, int width, int height, int depth, DenoiseMethod method);
void denoise3D_joint(const float* pet, const float* ct, float* output,
                     int width, int height, int depth, DenoiseMethod method);

void run_denoising(int width, int height, DenoiseMethod method);
void run_denoising3D(int width, int height, int depth, DenoiseMethod method);
void run_denoising3D_joint(int width, int height, int depth, DenoiseMethod method);

#define WINDOW 4                    // szűrőablak félmérete

#define SIGMA_SPATIAL 2.0f          // térbeli szórás
#define SIGMA_INTENSITY 1500.0f     // intenzitásbeli szórás

#define FILTER_POWER 5000.0f       // szűrés erőssége
#define PATCH_RADIUS 2             // kis minta mérete
#define SEARCH_RADIUS 6            // keresési ablak mérete