//
// Created by takib on 2025. 05. 23..
//

#ifndef CT_PROCESSOR_H
#define CT_PROCESSOR_H

#include <string>
#include <vector>

class CTProcessor {
public:
    CTProcessor(const std::string& filePath);
    void process();

private:
    std::string path;
};

#endif //CT_PROCESSOR_H
