# GEMINI.md - TriangleVision Context

## Project Overview
**TriangleVision** is a high-performance video processing tool that converts standard video (e.g., MP4) into a stylized mesh of triangles in real-time. It uses Delaunay triangulation, adaptive point sampling, and motion tracking to create a unique aesthetic.

- **Status:** Functional Prototype (Optimized for 30FPS+)
- **Core Technologies:** Python, OpenCV, SciPy (Delaunay), NumPy, ModernGL (GPU Acceleration), Numba.
- **Architecture:** 
  - `src/triangulate.py`: Geometry, point sampling (Edge density, Human detection, Motion).
  - `src/gpu_renderer.py`: OpenGL/ModernGL-based fast rendering.
  - `src/codec.py`: Custom `.triv` encoder/decoder with I/P frame motion tracking.
  - `main.py`: CLI entry point for realtime, encoding, playing, and exporting.

## Building and Running
- **Setup:** `pip install -r requirements.txt`
- **Realtime (Webcam):** `python3 main.py realtime --quality medium --human`
- **Encode Video:** `python3 main.py encode input.mp4 output.triv --quality high --human`
- **Play .triv File:** `python3 main.py play output.triv`

## Performance & Optimization ⚡
- **Target:** 30FPS+ at 720p with ~3000-5000 triangles.
- **GPU Acceleration:** `GPURenderer` is the primary path for realtime display.
- **Throttled Detection:** Human detection is throttled (every 3 frames) with 320px downscaling for massive speedups.
- **Profiling:** Use `benchmark.py` for component-level performance analysis.

## Development Conventions
- **Style:** Follow PEP 8 standards.
- **Complexity Management:** Use adaptive triangle counts based on edge density.
- **Resource Management:** Ensure GPU contexts and file handles are correctly closed.
- **Testing:** Always verify changes with `benchmark.py` to ensure real-time performance is maintained.
