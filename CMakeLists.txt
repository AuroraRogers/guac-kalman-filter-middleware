cmake_minimum_required(VERSION 3.10)
project(guac-kalman-filter LANGUAGES C)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GUAC REQUIRED libguac)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -pedantic")

# Include directories
include_directories(
    ${GUAC_INCLUDE_DIRS}
)

# Add executable
add_executable(guac-kalman-filter
    src/kalman_proxy.c
)

# Link libraries
target_link_libraries(guac-kalman-filter
    ${GUAC_LIBRARIES}
    pthread
)

# Install target
install(TARGETS guac-kalman-filter
    RUNTIME DESTINATION bin
)