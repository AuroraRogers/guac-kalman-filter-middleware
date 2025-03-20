# Guacamole Kalman Filter Proxy

A proxy server for Apache Guacamole that applies Kalman filtering to optimize remote desktop performance.

## Overview

This proxy sits between Guacamole clients and the guacd server, applying Kalman filtering to optimize the remote desktop experience. It analyzes the Guacamole protocol instructions and makes predictions to improve performance, especially in bandwidth-constrained environments.

## Features

- Intercepts and analyzes Guacamole protocol instructions
- Applies Kalman filtering to optimize data transmission
- Predicts bandwidth requirements and adapts accordingly
- Prioritizes layers based on user interaction
- Tracks update frequencies for different screen regions

## Requirements

- libguac (Apache Guacamole client library)
- CMake (3.10 or higher)
- C compiler with C11 support

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make

# Install (optional)
sudo make install
```

## Usage

1. Make sure guacd is running on port 4823:
   ```bash
   guacd -b 0.0.0.0 -l 4823
   ```

2. Run the Kalman filter proxy:
   ```bash
   ./guac-kalman-filter
   ```

3. Configure your Guacamole client to connect to the proxy on port 4822 instead of directly to guacd.

## Configuration

The proxy listens on port 4822 by default and connects to guacd on port 4823 on localhost. These values can be modified in the source code if needed.

## License

This project is licensed under the Apache License 2.0.