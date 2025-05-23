#include <iostream>
#include "pet_processor.h"
#include "denoise.cuh"
#include <chrono>
#include <ctime>
#include <iomanip>

#include "ct_processor.h"

void print_timestamp(const std::string& label = "") {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::cout << label
              << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S")
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: DenoisePET <pet.nii> <ct.nii> <filterMethod> <dimension>" << std::endl;
        return 1;
    }
    print_timestamp("Start: ");
    std::string petPath = argv[1];
    std::string ctPath = argv[2];
    std::string methodStr = argv[3];
    std::string dimension = argv[4];
    DenoiseMethod method = BOX_FILTER;

    if (methodStr == "identity") method = IDENTITY;
    else if (methodStr == "box") method = BOX_FILTER;
    else if (methodStr == "gaussian") method = GAUSSIAN;
    else if (methodStr == "median") method = MEDIAN;
    else if (methodStr == "bilateral") method = BILATERAL;
    else if (methodStr == "nlm") method = NLM;
    else if (methodStr == "joint_bilateral") method = JOINT_BILATERAL;
    else if (methodStr == "joint_nlm") method = JOINT_NLM;

    try {
        PETProcessor petProcessor(petPath);
        petProcessor.process(dimension);

        int width = 400;
        int height = 400;
        int depth = 304;

        if (dimension == "2") run_denoising(width, height, method);
        else if (dimension == "3")
        {
            if (method == JOINT_BILATERAL || method == JOINT_NLM) {
                CTProcessor ctProcessor(ctPath);
                ctProcessor.process();
                run_denoising3D_joint(width, height, depth, method);
            } else {
                run_denoising3D(width, height, depth, method);
            }
        }
        print_timestamp("End: ");
    } catch (const std::exception &ex) {
        std::cerr << "Error during PET processing: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
