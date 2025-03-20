cmake_minimum_required(VERSION 3.10)
project(guac-kalman-filter LANGUAGES C CXX CUDA)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GUAC REQUIRED libguac)
find_package(CUDA QUIET)

# Include directories
include_directories(${GUAC_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic")

# CUDA configuration
if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling GPU acceleration")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2)
    include_directories(${CUDA_INCLUDE_DIRS})
    
    # Add CUDA executable
    cuda_add_executable(guac-kalman-filter
        src/kalman_proxy.c
        src/kalman_cuda.cu
    )
    
    # Link libraries
    target_link_libraries(guac-kalman-filter
        ${GUAC_LIBRARIES}
        ${CUDA_LIBRARIES}
        pthread
    )
else()
    message(STATUS "CUDA not found, disabling GPU acceleration")
    
    # Add CPU-only executable
    add_executable(guac-kalman-filter
        src/kalman_proxy.c
    )
    
    # Link libraries
    target_link_libraries(guac-kalman-filter
        ${GUAC_LIBRARIES}
        pthread
    )
    
    # Define CUDA_DISABLED to disable CUDA code
    add_definitions(-DCUDA_DISABLED)
endif()

# Install target
install(TARGETS guac-kalman-filter
    RUNTIME DESTINATION bin
)