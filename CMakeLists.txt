cmake_minimum_required(VERSION 3.18)
project(svd_slate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require Cray compiler wrapper
if (NOT CMAKE_CXX_COMPILER MATCHES "CC$")
    message(FATAL_ERROR "Use: cmake -DCMAKE_CXX_COMPILER=CC ..")
endif()

# Hardcoded Spack install prefixes for Perlmutter E4S v0.22
set(SLATE_PREFIX    /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/slate-2023.11.05-eqi3ukwv3rsgh2sxmu7m6pprknrplgp7)
set(LAPACKPP_PREFIX /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/lapackpp-2023.11.05-zaow3ozein6yl2x2wib2ih7zuxsxqikp)
set(BLASPP_PREFIX   /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/blaspp-2023.11.05-6rpvyaew74upl772lz5v6cal6466azy6)

# Paths to headers
set(SLATE_INC    ${SLATE_PREFIX}/include)
set(LAPACKPP_INC ${LAPACKPP_PREFIX}/include)
set(BLASPP_INC   ${BLASPP_PREFIX}/include)

# Paths to library files
set(SLATE_LIB    ${SLATE_PREFIX}/lib64/libslate.so)
set(LAPACKPP_LIB ${LAPACKPP_PREFIX}/lib64/liblapackpp.so)
set(BLASPP_LIB   ${BLASPP_PREFIX}/lib64/libblaspp.so)

# Verification of paths
foreach(path_var IN ITEMS SLATE_INC LAPACKPP_INC BLASPP_INC)
    if (NOT EXISTS ${${path_var}})
        message(FATAL_ERROR "Missing include directory: ${${path_var}}")
    endif()
endforeach()

foreach(path_var IN ITEMS SLATE_LIB LAPACKPP_LIB BLASPP_LIB)
    if (NOT EXISTS ${${path_var}})
        message(FATAL_ERROR "Missing library file: ${${path_var}}")
    endif()
endforeach()

# Add include directories
include_directories(${SLATE_INC} ${LAPACKPP_INC} ${BLASPP_INC})

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Target
add_executable(svd_slate src/svd_slate.cpp)

# Link libraries explicitly
target_link_libraries(svd_slate
    ${SLATE_LIB}
    ${LAPACKPP_LIB}
    ${BLASPP_LIB}
    OpenMP::OpenMP_CXX
)

