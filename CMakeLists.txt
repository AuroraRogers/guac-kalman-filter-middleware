cmake_minimum_required(VERSION 3.10)
project(guac-kalman-filter C)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GUAC REQUIRED libguac)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wno-error=implicit-function-declaration")

# Include directories
include_directories(${GUAC_INCLUDE_DIRS})

# Create executable
add_executable(guac-kalman-filter
    src/kalman_proxy.c
)

# Link libraries
target_link_libraries(guac-kalman-filter ${GUAC_LIBRARIES} m pthread)

# Install
install(TARGETS guac-kalman-filter DESTINATION bin)