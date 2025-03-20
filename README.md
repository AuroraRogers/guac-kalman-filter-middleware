# Guacamole Kalman Filter Proxy

A proxy server that sits between Guacamole clients and the guacd server, applying Kalman filtering to optimize the remote desktop experience.

## Prerequisites

- libguac (Guacamole client library)
- CMake (version 3.10 or higher)
- C compiler with C11 support

## Building

```bash
# Create a build directory
mkdir build && cd build

# Configure the build
cmake ..

# Build the project
make
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

3. Connect your Guacamole client to the proxy on port 4822.

## Configuration

The proxy is configured to:
- Listen on port 4822 for client connections
- Connect to guacd on localhost:4823
- Apply Kalman filtering to optimize the remote desktop experience

## How It Works

The proxy:
1. Accepts connections from Guacamole clients
2. Establishes a connection to guacd
3. Processes Guacamole protocol instructions
4. Applies Kalman filtering to optimize the experience
5. Forwards instructions between the client and guacd

## Troubleshooting

If you encounter issues:
- Make sure guacd is running on port 4823
- Check that libguac is properly installed
- Verify that the proxy has permission to bind to port 4822
- Check the logs for error messages