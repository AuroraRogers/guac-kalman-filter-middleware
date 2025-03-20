# Guacamole RDP Kalman Filter Middleware with CUDA Acceleration

This module implements a Kalman filter middleware for the RDP protocol in the Apache Guacamole server with CUDA acceleration. The filter smooths cursor movements by applying statistical prediction and correction techniques using parallel processing on NVIDIA GPUs.

## Features

- CUDA-accelerated Kalman filtering for RDP mouse movements
- Socket middleware design that integrates with Guacamole's protocol handling
- Support for statistics collection to CSV format
- Runtime configuration of filter parameters
- GPU acceleration for matrix operations
- **Video Optimization**: Enhances video quality using adaptive filtering techniques
- **Image Quality Metrics**: Calculates and logs video quality metrics

## Video Optimization Features

The video optimization module provides the following capabilities:

- **Dynamic Quality Adjustment**: Automatically adjusts quality parameters based on network conditions and video complexity
- **Advanced Quality Metrics**: Measures and tracks multiple video quality metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - MS-SSIM (Multi-Scale Structural Similarity)
  - VMAF (Video Multi-method Assessment Fusion)
  - VQM (Video Quality Metric)
- **Bandwidth Management**: Targets specific bandwidth constraints while maximizing visual quality
- **Performance Monitoring**: Tracks frame rates and processing overhead

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 10.0 or higher
- Apache Guacamole Server (1.5.5 or compatible)
- CMake 3.10 or higher
- FFmpeg (optional, for enhanced video metrics)
- OpenCV (optional, for enhanced image processing)

## Building

The middleware can be built using CMake:

```bash
mkdir build
cd build
cmake ..
make
```

## Installation

After building, you can install the middleware:

```bash
sudo make install
```

This will install the library to your system's library directory.

## Usage

To use the Kalman filter in your Guacamole server, you need to:

1. Load the middleware in your Guacamole server configuration
2. Wrap your RDP socket with the Kalman filter

Example code:

```c
#include <guacamole/socket.h>
#include <kalman_filter.h>

/* Create a Kalman filter-wrapped socket */
guac_socket* filtered_socket = guac_socket_kalman_filter_alloc(original_socket);

/* Configure the filter (optional) */
guac_kalman_filter* filter = (guac_kalman_filter*) filtered_socket->data;
guac_kalman_filter_configure(filter, 0.01, 1.0, 1.0);

/* Enable statistics logging (optional) */
guac_kalman_filter_enable_stats(filter, "/var/log/guacamole/kalman-stats.csv");

/* Enable video optimization */
guac_kalman_filter_enable_video_optimization(filter, true, 80, 2000);
```

## Configuration Parameters

- **Process Noise**: Controls how much the filter trusts its internal model versus new measurements
- **Measurement Noise X/Y**: Controls how much the filter trusts new measurements in each axis
- **Video Optimization**: Enable or disable video optimization
- **Target Quality**: Target video quality from 0-100
- **Target Bandwidth**: Target bandwidth in kbps, 0 for unlimited

## Performance Considerations

The CUDA acceleration provides significant performance improvements for Kalman filter operations, but requires:

- A compatible NVIDIA GPU
- Proper CUDA driver installation
- Sufficient GPU memory for matrix operations

For systems without CUDA support, a fallback CPU implementation could be provided.

## Statistics

The Kalman filter middleware can generate comprehensive statistics in CSV format, which can be used for analysis and visualization. The statistics include:

- **Mouse Position**: Original and filtered mouse positions
- **Processing Time**: Time taken for filtering operations
- **Quality Metrics**: Various video quality metrics for each frame
- **Bandwidth Usage**: Data size before and after optimization

## License

This software is licensed under the Apache License, Version 2.0.
