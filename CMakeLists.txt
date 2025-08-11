cmake_minimum_required(VERSION 3.18)
project(svd_slate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CXX_COMPILER MATCHES "CC$")
    message(FATAL_ERROR "Use: cmake -DCMAKE_CXX_COMPILER-CC ..")
endif()

# Find SLATE
find_path(SLATAE_INCLUDE_DIR slate/slate.hh)
find_library(SLATE_LIBRARY slate)

find_path(LAPACKPP_INCLUDE_DIR lapack.hh)
find_library(LAPACKPP_LIBRARY lapackpp)

find_path(BLASPP_INCLUDE_DIR lapack.hh)
find_library(BLASPP_LIBRARY blasspp)

if (NOT SLATE_INCLUDE_DIR OR NOT SLATE_LIBRARY)
    message(FATAL_ERROR "SLATE not found. Did you 'spack load slate'?")
endif()

if (NOT LAPACKPP_INCLUDE_DIR OR NOT LAPACKPP_LIBRARY)
    message(FATAL_ERROR "LAPACK++ not found. Did you 'spack load lapackpp'?")
endif()

if (NOT BLASPP_INCLUDE_DIR OR NOT BLASPP_LIBRARY)
    message(FATAL_ERROR "BLAS++ not found. Did you 'spack load blaspp'?")
endif()

include_directories(
    ${SLATE_INCLUDEDIR}
    ${LAPACKPP_INCLUDE_DIR}
    ${BLASPP_INCLUDE_DIR}
)

add_executable(svd_slate svd_slate.cpp)

target_link_libraries(svd_slate
    ${SLATE_LIBRARY}
    ${LAPACKPP_LIBRARY}
    ${BLASPP_LIBRARY}
)


