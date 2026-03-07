# TriangleVision TODO 📝

This document tracks planned improvements, optimizations, and new features for the TriangleVision project.

## ⚡ Performance Optimizations
- [ ] **GPU-Based Triangulation**: Move the Delaunay triangulation from SciPy (CPU) to a GPU-based implementation (e.g., CUDA or specialized Compute Shaders) to remove the CPU bottleneck entirely.
- [ ] **Fully Asynchronous Pipeline**: Refactor the architecture into a multi-stage pipeline where `Capture`, `Feature Detection`, `Triangulation`, and `GPU Rendering` run in separate threads/processes with zero-copy shared memory.
- [ ] **C++/Cython Core**: Port the remaining Python bottlenecks in `triangulate.py` to a C++ extension using `pybind11` for maximum execution speed.
- [ ] **SIMD Color Sampling**: Optimize the Numba color sampler to utilize AVX-512 or NEON instructions for faster pixel averaging.

## 🛠️ Usability Features
- [ ] **Standalone GUI**: Build a modern interface (using PyQt or Dear PyGui) featuring real-time sliders for triangle count, quality, and style parameters.
- [ ] **Live Parameter Tweaking**: Allow users to toggle modes (`human`, `rotoscope`, `heatmap`) and adjust sensitivity via hotkeys or UI without restarting the stream.
- [ ] **Interactive Detail Brushes**: Let users "paint" areas of the screen to manually increase triangle density during real-time processing.
- [ ] **Config System**: Support `.yaml` or `.json` config files for persistent user preferences and quality presets.

## 💎 Quality & Style Enhancements
- [ ] **Temporal Mesh Smoothing**: Implement point-to-point tracking between frames to reduce "mesh popping" and flickering in static areas.
- [ ] **Advanced Shading**: Add support for Gouraud or Phong shading within the triangles to create smoother gradients or 3D-like lighting effects.
- [ ] **Texture Mapping**: Support mapping the original video texture onto the triangles with adjustable transparency for a "stained glass" effect.
- [x] **AI-Powered Saliency**: Replaced Haar Cascades with OpenCV's Spectral Residual Saliency for broader, faster point distribution on interesting objects.

## 📦 File Format & Speed (.triv v2)
- [ ] **Delta Encoding**: Only store the movement (vectors) of points between frames to drastically reduce the file size of the `.triv` format.
- [ ] **Modern Compression**: Switch from `zlib` to `Zstd` or `LZ4` for significantly faster decompression during high-speed playback.
- [ ] **Bit-Packing**: Pack point coordinates and color data into bit-aligned structures to minimize the binary footprint.
- [ ] **Headless Batch Encoder**: Create a high-performance CLI tool for batch-converting directories of MP4 files to `.triv` without UI overhead.
