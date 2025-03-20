# Guacamole Kalman Filter Proxy

A proxy server for Apache Guacamole that applies Kalman filtering to optimize remote desktop performance.

## Overview

This proxy sits between Guacamole clients and the guacd server, analyzing and optimizing the Guacamole protocol instructions to improve performance, especially in bandwidth-constrained environments.

## Features

- Listens on port 4822 (default Guacamole port)
- Connects to guacd on port 4823
- Processes Guacamole protocol instructions
- Applies Kalman filtering for bandwidth prediction and optimization
- Tracks layer priorities and dependencies
- Monitors region update frequencies

## Building

### Prerequisites

- C compiler (gcc or clang)
- CMake (version 3.10 or higher)
- libguac (Apache Guacamole C library)

### Build Steps

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make
```

## Usage

1. Start guacd on port 4823:
   ```bash
   guacd -l 4823
   ```

2. Start the Kalman filter proxy:
   ```bash
   ./guac-kalman-filter
   ```

3. Configure Guacamole clients to connect to the proxy on port 4822

## Configuration

The proxy is currently configured with the following default settings:

- Listens on: 0.0.0.0:4822
- Connects to guacd at: 127.0.0.1:4823
- Log level: DEBUG
- Max layers: 10
- Max regions: 100

## Implementation Details

The proxy implements a Kalman filter to predict bandwidth requirements and optimize the delivery of Guacamole protocol instructions. It tracks:

- Layer priorities based on user selection
- Layer dependencies from copy operations
- Region update frequencies from image operations
- Bandwidth measurements and predictions

## License

This project is licensed under the Apache License 2.0.