# GEMINI.md - TriangleVision Context

## Project Overview
**TriangleVision** is a video processing tool that converts standard video (e.g., MP4) into a stylized mesh of triangles in real-time. It uses Delaunay triangulation and optical flow to create a unique aesthetic while maintaining image detail and screen ratio.

- **Status:** Functional Prototype
- **Core Technologies:** Python, OpenCV, SciPy (Delaunay), NumPy.
- **Architecture:** 
  - `src/triangulate.py`: Core geometry and point sampling logic.
  - `src/codec.py`: Custom `.triv` encoder/decoder with I/P frame motion tracking.
  - `main.py`: CLI entry point for realtime processing, encoding, and playing.

## Building and Running
- **Setup:** `pip install -r requirements.txt`
- **Realtime (Webcam):** `python3 main.py realtime --quality medium`
- **Encode Video:** `python3 main.py encode input.mp4 output.triv --quality high`
- **Play .triv File:** `python3 main.py play output.triv`

## Development Conventions
- **Style:** Follow PEP 8 standards.
- **Complexity Management:** Use adaptive triangle counts based on edge density.
- **Performance:** Prioritize OpenCV vectorized operations over manual loops where possible.

## Key Files
- `main.py`: CLI interface.
- `requirements.txt`: Project dependencies.
- `src/`: Core implementation modules.
- `GEMINI.md`: Project context and instructions.
