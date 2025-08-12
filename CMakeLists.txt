cmake_minimum_required(VERSION 3.18)
project(svd_slate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require Cray compiler wrapper
if (NOT CMAKE_CXX_COMPILER MATCHES "CC$")
    message(FATAL_ERROR "Use: cmake -DCMAKE_CXX_COMPILER=CC ..")
endif()

# Hardcoded Spack install prefixes for Perlmutter E4S v0.22
set(SLATE_PREFIX    /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/slate-2023.11.05-x4dznn44tjb5l3tpl7s7uuogu3o5qfdz)
set(LAPACKPP_PREFIX /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/lapackpp-2023.11.05-pph2uwrfujonebbyueaptvnm3elgz77e)
set(BLASPP_PREFIX   /global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/blaspp-2023.11.05-uffk4lufpj5ne7itkn27wcqcmj42zi5u)

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

if ( NOT DEFINED TAMM_DIR )
    set(TAMM_DIR "/global/homes/d/dogskep/AaronForkNWQ-Sim/tamm_install"
      CACHE PATH "Path to external TAMM installation")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TAMM_DIR}")

message(${TAMM_DIR})
# Enable OpenMP
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TAMM)

# Target
add_executable(svd_slate src/svd_slate.cpp)

target_link_options(svd_slate PRIVATE
    -Wl,--no-as-needed
)

get_target_property(_cudart CUDA::cudart IMPORTED_LOCATION)
get_filename_component(_cuda_libdir ${_cudart} DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH ${_cuda_libdir})

target_link_libraries(svd_slate
    -Wl,--start-group
    ${SLATE_LIB}
    ${LAPACKPP_LIB}
    ${BLASPP_LIB}
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cusolver
    -Wl,--end-group
    OpenMP::OpenMP_CXX
    tamm::tamm
)

target_compile_definitions(svd_slate PUBLIC TAMM_ENABLED)
