cmake_minimum_required(VERSION 3.10)
project(guac-kalman-filter)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GUAC REQUIRED libguac)

# Include directories
include_directories(${GUAC_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

# Create src directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add executable
add_executable(guac-kalman-filter
    src/kalman_proxy.c
)

# Link libraries
target_link_libraries(guac-kalman-filter ${GUAC_LIBRARIES})