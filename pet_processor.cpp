//
// Created by takib on 2025. 05. 20..
//

#include "pet_processor.h"
#include <filesystem>
#include <fstream>
#include "nifti1_io.h"
#include <iostream>

PETProcessor::PETProcessor(const std::string &filePath)
    : path(filePath) {}

void PETProcessor::process(std::string dimension) {
    namespace fs = std::filesystem;
    std::cout << "Attempting to load file: " << path << std::endl;

    nifti_image *image = nifti_image_read(path.c_str(), 1);
    if (!image) {
        throw std::runtime_error("Failed to load PET NIfTI file: " + path);
    }

    if (image->datatype != DT_FLOAT32) {
        nifti_image_free(image);
        throw std::runtime_error("Unsupported data type in PET file (only float32 supported).");
    }

    std::cout << "File loaded: " << path << std::endl;
    std::cout << "Dimensions: " << image->nx << " x " << image->ny << " x " << image->nz << std::endl;
    std::cout << "Data type: " << image->datatype << std::endl;
    std::cout << "Number of voxels: " << image->nvox << std::endl;

    fs::path outDir = fs::path("..") / ("result");
    if (!exists(outDir)) {
        create_directory(outDir);
    }

    float *data = static_cast<float *>(image->data);
    int width = image->nx;
    int height = image->ny;
    int depth = image->nz;

    if (dimension == "2")
    {
        fs::path slicesDir = outDir / "slices";
        if (!exists(slicesDir)) {
            create_directory(slicesDir);
        }
        for (int z = 0; z < depth; ++z) {
            fs::path outPath = slicesDir / ("slice_" + std::to_string(z) + ".raw");
            std::ofstream out(outPath, std::ios::binary);

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = z * width * height + y * width + x;
                    out.write(reinterpret_cast<char*>(&data[idx]), sizeof(float));
                }
            }

            out.close();
        }

        std::cout << "Saved " << depth << " slice files.\n";
    }
    else if (dimension == "3")
    {
        fs::path outPath = outDir / "pet_image_file.raw";
        std::ofstream out(outPath, std::ios::binary);
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = z * width * height + y * width + x;
                    out.write(reinterpret_cast<char*>(&data[idx]), sizeof(float));
                }
            }
        }
        out.close();
        std::cout << "Saved 3D image to file.\n";
    }
    else
    {
        std::cerr << "Unknown dimension argument: " << dimension << std::endl;
    }


    nifti_image_free(image);
}
