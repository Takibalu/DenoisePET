//
// Created by takib on 2025. 05. 21..
//

#include <filesystem>
#include <fstream>
#include <vector>
#include "denoise.cuh"

#include <iostream>
#include <limits>

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
    fs::path outputDir = fs::path("..") / ("denoise_" + methodName + "_" + std::to_string(WINDOW*2+1)+ "x"+ std::to_string(WINDOW*2+1));
    create_directories(outputDir);


    for (const auto& entry : fs::directory_iterator(fs::path("..") / "result"/ "slices")) {
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

void run_denoising3D(int width, int height, int depth, DenoiseMethod method) {
    namespace fs = std::filesystem;

    std::string methodName = to_string(method);
    fs::path inputPath = fs::path("..") / "result" / "pet_image_file.raw";
    fs::path outputDir = fs::path("..") / ("denoise3D_" + methodName /*+ "_" + std::to_string(window*2+1)+ "x"+ std::to_string(window*2+1)*/ + "_5000_2_6_opt_opt");
    create_directories(outputDir);

    size_t volumeSize = static_cast<size_t>(width) * height * depth;
    std::vector<float> input(volumeSize);
    std::vector<float> output(volumeSize);

    std::ifstream in(inputPath, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open 3D input file: " << inputPath << std::endl;
        return;
    }
    in.read(reinterpret_cast<char*>(input.data()), volumeSize * sizeof(float));
    in.close();

    denoise3D(input.data(), output.data(), width, height, depth, method);

    for (int z = 0; z < depth; ++z) {
        fs::path outPath = outputDir / ("slice_" + std::to_string(z) + ".raw");
        std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "Failed to write slice: " << outPath << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = z * width * height + y * width + x;
                out.write(reinterpret_cast<char*>(&output[idx]), sizeof(float));
            }
        }

        out.close();
    }

    std::cout << "Denoised 3D volume sliced into " << depth
              << " slices and saved to: " << outputDir << std::endl;
}

void run_denoising3D_joint(int width, int height, int depth, DenoiseMethod method) {
    namespace fs = std::filesystem;

    std::string methodName = to_string(method);
    fs::path petInputPath = fs::path("..") / "result" / "pet_image_file.raw";
    fs::path ctInputPath = fs::path("..") / "result" / "ct_image_file.raw";
    fs::path outputDir = fs::path("..") / ("denoise3D_" + methodName + "_" + std::to_string(WINDOW*2+1)+ "x"+ std::to_string(WINDOW*2+1) + "_2_1500");
     create_directories(outputDir);

    size_t volumeSize = static_cast<size_t>(width) * height * depth;
    std::vector<float> petInput(volumeSize);
    std::vector<float> ctInput(volumeSize);
    std::vector<float> output(volumeSize);

    std::ifstream petIn(petInputPath, std::ios::binary);
    if (!petIn) {
        std::cerr << "Failed to open 3D input file: " << petInputPath << std::endl;
        return;
    }
    petIn.read(reinterpret_cast<char*>(petInput.data()), volumeSize * sizeof(float));
    petIn.close();

    std::ifstream ctIn(petInputPath, std::ios::binary);
    if (!ctIn) {
        std::cerr << "Failed to open 3D input file: " << petInputPath << std::endl;
        return;
    }
    ctIn.read(reinterpret_cast<char*>(petInput.data()), volumeSize * sizeof(float));
    ctIn.close();

    denoise3D_joint(petInput.data(), ctInput.data(), output.data(), width, height, depth, method);

    for (int z = 0; z < depth; ++z) {
        fs::path outPath = outputDir / ("slice_" + std::to_string(z) + ".raw");
        std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "Failed to write slice: " << outPath << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = z * width * height + y * width + x;
                out.write(reinterpret_cast<char*>(&output[idx]), sizeof(float));
            }
        }

        out.close();
    }

    std::cout << "Joint denoised 3D PET saved to: " << outputDir << std::endl;
}




