cmake_minimum_required(VERSION 3.18)
project(svd_slate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CXX_COMPILER MATCHES "CC$")
    message(FATAL_ERROR "Use: cmake -DCMAKE_CXX_COMPILER-CC ..")
endif()

set(SLATE_PREFIX /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/slate-2023.11.05-x4dznn44tjb5l3tpl7s7uuogu3o5qfdz)
set(LAPACKPP_PREFIX /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/lapackpp-2023.11.05-zaow3ozein6yl2x2wib2ih7zuxsxqikp)
set(BLASPP_PREFIX /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/blaspp-2023.11.05-uffk4lufpj5ne7itkn27wcqcmj42zi5u)

include_directories(
    ${SLATE_PREFIX}/include
    ${LAPACKPP_PREFIX}/include
    ${BLASPP_PREFIX}/include
)

link_directories(
    ${SLATE_PREFIX}/lib64 ${SLATE_PREFIX}/lib
    ${LAPACKPP_PREFIX}/lib64 ${LAPACKPP_PREFIX}/lib
    ${BLASPP_PREFX}/lib64 ${BLASPP_PREFIX}/lib
)

add_executable(svd_slate ./src/svd_slate.cpp)

target_link_libraries(svd_slate slate lapackpp blaspp)
