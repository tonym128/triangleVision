# AI Agents Guide 🦾

This document provides instructions for AI agents (like Gemini CLI) to effectively interact with, debug, and enhance the TriangleVision codebase.

## Codebase Architecture 🏛️

- **`main.py`**: Entry point for all CLI commands. Handles threading and high-level execution loops.
- **`src/triangulate.py`**: Core logic for point generation, triangulation (Delaunay), and color sampling.
- **`src/gpu_renderer.py`**: OpenGL/ModernGL implementation for triangle rendering.
- **`src/codec.py`**: Implementation of the custom `.triv` encoder/decoder.

## Performance Considerations ⚡

- **Point Generation**: The most performance-critical part. For 30FPS+, human detection is throttled (every 3 frames) and detection images are downscaled to 320px width.
- **GPU Acceleration**: Always prefer `GPURenderer` for standard rendering. It is roughly 4-5x faster than the CPU fallback.
- **NumPy/Numba**: Use vectorized NumPy operations or Numba-jitted functions for heavy pixel processing.

## Testing & Benchmarking 📊

- Use `benchmark.py` to measure the performance of each major component (complexity, point generation, triangulation, color sampling, and rendering).
- When optimizing, always check performance with both `detect_human=True` and `detect_human=False`.

## Development Principles 🛠️

- **Preserve Real-time Capability**: Any change that increases frame processing time beyond 33ms (for 30FPS) should include a fast-path or optimization.
- **Idiomatic Python**: Follow PEP 8 and maintain clear separation of concerns between geometry logic and rendering.
- **No Side Effects**: Ensure `GPURenderer` is correctly closed and its context released to avoid resource leaks.
