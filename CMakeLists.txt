cmake_minimum_required(VERSION 3.30)
project(DenoisePET CUDA CXX C)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)

include_directories(
        ${CMAKE_SOURCE_DIR}/nifti_clib/niftilib
        ${CMAKE_SOURCE_DIR}/nifti_clib/znzlib
)

file(GLOB NIFTI_SOURCES
        ${CMAKE_SOURCE_DIR}/nifti_clib/niftilib/*.c
        ${CMAKE_SOURCE_DIR}/nifti_clib/znzlib/*.c
)

add_library(nifti_lib ${NIFTI_SOURCES})

set_target_properties(nifti_lib PROPERTIES LINKER_LANGUAGE C)

find_package(ZLIB)
if(ZLIB_FOUND)
    target_link_libraries(nifti_lib ZLIB::ZLIB)
endif()

set(MY_SOURCES
        main.cpp
        kernel.cu
        pet_processor.cpp
        pet_processor.h
        denoise_runner.cpp
        denoise.cu
        denoise.cuh
        ct_processor.cpp
        ct_processor.h
)

add_executable(DenoisePET ${MY_SOURCES})

target_link_libraries(DenoisePET nifti_lib)

set_target_properties(DenoisePET PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
)