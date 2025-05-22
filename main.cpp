#include <iostream>
#include "pet_processor.h"
#include "denoise.cuh"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: MyCudaProject <path_to_nifti_file.nii> <filterMethod>" << std::endl;
        return 1;
    }

    std::string niftiFilePath = argv[1];
    std::string methodStr = argv[2];
    std::string dimension = argv[3];
    DenoiseMethod method = BOX_FILTER;

    if (methodStr == "identity") method = IDENTITY;
    else if (methodStr == "box") method = BOX_FILTER;
    else if (methodStr == "gaussian") method = GAUSSIAN;
    else if (methodStr == "median") method = MEDIAN;
    else if (methodStr == "bilateral") method = BILATERAL;
    else if (methodStr == "nlm") method = NLM;

    try {
        PETProcessor processor(niftiFilePath);
        processor.process(dimension);

        int width = 400;
        int height = 400;
        int depth = 302;

        if (dimension == "2") run_denoising(width, height, method);
        else if (dimension == "3") run_denoising3D(width, height, depth, method);
    } catch (const std::exception &ex) {
        std::cerr << "Error during PET processing: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
