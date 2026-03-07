# TriangleVision 📐✨

TriangleVision is a high-performance video processing tool that converts standard video (e.g., MP4) into a stylized, real-time mesh of triangles. It blends geometry and aesthetics to create a unique visual style while maintaining image detail and screen ratio.

## Features 🚀

- **Real-time Processing**: Process webcam or video files at 30+ FPS (on modern hardware).
- **GPU-Accelerated Rendering**: Utilizes ModernGL for high-performance triangle rendering.
- **Intelligent Point Sampling**:
  - **Delaunay Triangulation**: Creates a smooth, connected mesh.
  - **Edge Density Adaptation**: Automatically adds more detail to high-complexity areas.
  - **Human/Face Detection**: Enhanced detail in areas containing people.
  - **Motion Tracking**: Increased triangle density in regions with significant movement.
- **Custom `.triv` Codec**: A custom binary format for storing triangle-based video data with high efficiency.
- **Multiple Modes**:
  - **Realtime**: Live camera or video file playback with real-time triangulation.
  - **Encode**: Convert standard videos into optimized `.triv` files.
  - **Player**: High-speed playback of `.triv` files.
  - **Export**: Export triangulated videos back to standard formats (MP4, MKV).
- **Visual Styles**: Toggle between standard triangulation, **Rotoscope** (ink edges + cell shading), and **Heatmap** modes.

## How it's Implemented 🛠️

TriangleVision is built with Python and leverages several powerful libraries for performance and geometric calculations:

- **OpenCV**: Core image processing, feature detection (Shi-Tomasi), and human detection (HOG + Haar Cascades).
- **SciPy (Delaunay)**: High-speed triangulation of sampled points.
- **ModernGL**: GPU-accelerated triangle rendering with custom GLSL shaders.
- **NumPy & Numba**: Vectorized operations and JIT compilation for color sampling and data processing.
- **Threaded Capture**: Multi-threaded video grabbing to ensure smooth input regardless of processing load.

## Getting Started 🏁

### Prerequisites

- Python 3.8+
- A GPU supporting OpenGL 3.3+ (for hardware acceleration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/triangleVision.git
   cd triangleVision
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Real-time Webcam Mode
Run the tool on your webcam with default settings:
```bash
python3 main.py realtime --quality medium --human
```

#### 2. Encode a Video File
Convert an MP4 to a `.triv` file at high quality:
```bash
python3 main.py encode input.mp4 output.triv --quality high --human
```

#### 3. Play a `.triv` File
Enjoy the triangulated version of your video:
```bash
python3 main.py play output.triv
```

#### 4. Export to MP4
Export a stylized version for sharing:
```bash
python3 main.py export output.triv finished_video.mp4 --rotoscope
```

## CLI Arguments ⌨️

- `--source`: Camera ID or video file path (for `realtime`).
- `--triangles`: Manually set the target number of triangles.
- `--quality`: Adaptive quality presets (`low`, `medium`, `high`).
- `--rotoscope`: Enable ink edges and cell-shading aesthetics.
- `--human`: Enable smart detail enhancement on humans.
- `--output`: Save a `realtime` session directly to a `.triv` file.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
