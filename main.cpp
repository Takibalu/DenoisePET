#include <iostream>
#include "pet_processor.h"
#include "denoise.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: MyCudaProject <path_to_nifti_file.nii.gz>" << std::endl;
        return 1;
    }

    std::string niftiFilePath = argv[1];
    DenoiseMethod method = BOX_FILTER;

    if (argc >= 3) {
        std::string methodStr = argv[2];
        if (methodStr == "identity") method = IDENTITY;
        else if (methodStr == "box") method = BOX_FILTER;
        else if (methodStr == "gaussian") method = GAUSSIAN;
        else if (methodStr == "median") method = MEDIAN;
    }
    try {
        PETProcessor processor(niftiFilePath);
        processor.process();

        int width = 400;
        int height = 400;
        run_denoising(width, height, method);
    } catch (const std::exception &ex) {
        std::cerr << "Error during PET processing: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
