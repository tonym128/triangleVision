import cv2
import numpy as np
import time
from src.triangulate import get_triangles_and_colors, draw_triangles, generate_points, compute_complexity, determine_triangle_count

def benchmark():
    # Load a sample image or create a dummy one
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    quality = 'high' # Using high for heavier load
    num_triangles = determine_triangle_count(0.08, quality)
    
    print(f"Benchmarking with {num_triangles} triangles on 720p frame...")

    # Warmup
    print("Warming up...")
    points, h_pts, _ = generate_points(frame, num_triangles, detect_human=True)
    simplices, colors = get_triangles_and_colors(frame, points)
    _ = draw_triangles(frame.shape, points, simplices, colors, human_points=h_pts, heatmap=True)

    # Time compute_complexity
    start = time.time()
    for _ in range(30):
        comp = compute_complexity(frame)
    print(f"compute_complexity: {(time.time() - start) / 30:.4f}s")

    # Time generate_points (with human detection)
    start = time.time()
    for _ in range(30):
        points, h_pts, _ = generate_points(frame, num_triangles, detect_human=True)
    print(f"generate_points (+human): {(time.time() - start) / 30:.4f}s")
    
    # Time generate_points (WITHOUT human detection)
    start = time.time()
    for _ in range(30):
        points, _, _ = generate_points(frame, num_triangles, detect_human=False)
    print(f"generate_points (-human): {(time.time() - start) / 30:.4f}s")
    
    points, h_pts, _ = generate_points(frame, num_triangles, detect_human=True)

    # Time get_triangles_and_colors
    start = time.time()
    for _ in range(30):
        simplices, colors = get_triangles_and_colors(frame, points)
    print(f"get_triangles_and_colors: {(time.time() - start) / 30:.4f}s")
    
    simplices, colors = get_triangles_and_colors(frame, points)

    # Time draw_triangles
    start = time.time()
    for _ in range(30):
        out = draw_triangles(frame.shape, points, simplices, colors)
    print(f"draw_triangles: {(time.time() - start) / 30:.4f}s")
    
    # Time draw_triangles with rotoscope
    start = time.time()
    for _ in range(30):
        out = draw_triangles(frame.shape, points, simplices, colors, rotoscope=True)
    print(f"draw_triangles (rotoscope): {(time.time() - start) / 30:.4f}s")
    
    # Time draw_triangles (heatmap+human)
    start = time.time()
    for _ in range(30):
        out = draw_triangles(frame.shape, points, simplices, colors, heatmap=True, human_points=h_pts)
    print(f"draw_triangles (heatmap+human): {(time.time() - start) / 30:.4f}s")

    # Time GPURenderer
    from src.gpu_renderer import GPURenderer
    try:
        renderer = GPURenderer(1280, 720)
        start = time.time()
        for _ in range(30):
            out = renderer.render(points, simplices, colors)
        print(f"GPURenderer: {(time.time() - start) / 30:.4f}s")
        renderer.close()
    except Exception as e:
        print(f"GPURenderer failed: {e}")

if __name__ == "__main__":
    benchmark()
