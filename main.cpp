#include <iostream>
#include "pet_processor.h"
#include "denoise.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: MyCudaProject <path_to_nifti_file.nii.gz>" << std::endl;
        return 1;
    }

    std::string niftiFilePath = argv[1];
    DenoiseMethod method = BOX_FILTER; // alapértelmezett

    if (argc >= 3) {
        std::string methodStr = argv[2];
    }
    try {
        PETProcessor processor(niftiFilePath);
        processor.process();

        int width = 128;
        int height = 128;
        run_denoising(width, height, method);
    } catch (const std::exception &ex) {
        std::cerr << "Error during PET processing: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
