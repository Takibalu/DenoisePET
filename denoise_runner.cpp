//
// Created by takib on 2025. 05. 21..
//

#include "denoise.cuh"
#include <fstream>
#include <vector>
#include <filesystem>

void run_denoising(int width, int height, DenoiseMethod method) {
    namespace fs = std::filesystem;

    std::string methodName = to_string(method);
    fs::path outputDir = fs::path("..") / ("denoise_" + methodName);
    create_directories(outputDir);

    for (const auto& entry : fs::directory_iterator("slices")) {
        std::ifstream in(entry.path(), std::ios::binary);
        std::vector<float> input(width * height);
        in.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        in.close();

        std::vector<float> output(width * height);
        denoise(input.data(), output.data(), width, height, method);

        fs::path outPath = outputDir / entry.path().filename();
        std::ofstream out(outPath, std::ios::binary| std::ios::trunc);
        out.write(reinterpret_cast<char*>(output.data()), output.size() * sizeof(float));
        out.close();
    }
}

