cmake_minimum_required(VERSION 3.10)
project(guac-kalman-filter C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GUAC REQUIRED libguac)

# Include directories
include_directories(${GUAC_INCLUDE_DIRS})

# Add executable
add_executable(guac-kalman-filter src/kalman_proxy.c)

# Link libraries
target_link_libraries(guac-kalman-filter ${GUAC_LIBRARIES})

# Installation
install(TARGETS guac-kalman-filter DESTINATION bin)