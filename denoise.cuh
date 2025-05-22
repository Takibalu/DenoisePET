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


static const int window = 2;

static const float sigma_s = 2.0f;  // térbeli szórás
static const float sigma_r = 1500.0f;  // intenzitásbeli szórás

static const float h = 3000.0f;            // szűrés erőssége
static const int patch_radius = 5;       // kis minta mérete
static const int search_radius = 25;      // keresési ablak mérete