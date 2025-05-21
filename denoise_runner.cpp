//
// Created by takib on 2025. 05. 21..
//

#include "denoise.cuh"
#include <fstream>
#include <vector>
#include <filesystem>

#include <numeric>
#include <limits>
#include <iostream>


void print_statistics(const std::vector<float>& data, const std::string& label) {
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    double sum = 0.0;

    for (float val : data) {
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
        sum += val;
    }

    float mean = static_cast<float>(sum / data.size());

    std::cout << label << " - Min: " << minVal
              << ", Max: " << maxVal
              << ", Mean: " << mean << std::endl;
}

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
        //print_statistics(input, "Original " + entry.path().filename().string());
        denoise(input.data(), output.data(), width, height, method);
        //print_statistics(output, "Denoised " + entry.path().filename().string());

        fs::path outPath = outputDir / entry.path().filename();
        std::ofstream out(outPath, std::ios::binary| std::ios::trunc);
        out.write(reinterpret_cast<char*>(output.data()), output.size() * sizeof(float));
        out.close();
    }
}




