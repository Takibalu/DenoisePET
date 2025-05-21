//
// Created by takib on 2025. 05. 20..
//

#ifndef PET_PROCESSOR_H
#define PET_PROCESSOR_H

#include <string>

class PETProcessor {
public:
    explicit PETProcessor(const std::string &filePath);

    void process();

private:
    std::string path;
};

#endif // PET_PROCESSOR_H
