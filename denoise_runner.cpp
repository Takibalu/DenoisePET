//
// Created by takib on 2025. 05. 21..
//

#include "denoise.cuh"
#include <fstream>
#include <vector>
#include <filesystem>

void run_denoising(int width, int height) {
    namespace fs = std::filesystem;
    fs::create_directory("denoised");

    for (const auto& entry : fs::directory_iterator("slices")) {
        std::ifstream in(entry.path(), std::ios::binary);
        std::vector<float> input(width * height);
        in.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        in.close();

        std::vector<float> output(width * height);
        denoise(input.data(), output.data(), width, height);

        std::string outPath = "denoised/" + entry.path().filename().string();
        std::ofstream out(outPath, std::ios::binary);
        out.write(reinterpret_cast<char*>(output.data()), output.size() * sizeof(float));
        out.close();
    }
}

