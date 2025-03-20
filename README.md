# Guacamole Kalman Filter Proxy

A proxy server that sits between Guacamole clients and the guacd server, applying Kalman filtering to optimize the remote desktop experience.

## Prerequisites

- libguac (Guacamole server library)
- CUDA toolkit
- CMake (3.10 or higher)
- C compiler (gcc/clang)

## Building

```bash
# Create a build directory
mkdir build
cd build

# Configure and build
cmake ..
make

# Install (optional)
sudo make install
```

## Usage

1. Start the guacd server on port 4823 (default is 4822)
   ```bash
   guacd -l 4823
   ```

2. Start the Kalman filter proxy
   ```bash
   guac-kalman-filter
   ```

3. Configure your Guacamole client to connect to the proxy server (default port 4822)

## Configuration

The proxy is configured to listen on 0.0.0.0:4822 and connect to guacd at 127.0.0.1:4823.

## How It Works

The Kalman filter proxy:

1. Intercepts Guacamole protocol instructions between client and server
2. Analyzes drawing patterns and layer usage
3. Uses a Kalman filter to predict and optimize bandwidth usage
4. Prioritizes important screen regions based on user interaction

## License

Apache License 2.0