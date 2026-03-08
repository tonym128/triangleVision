import cv2
import time
from src.codec import TriangleDecoder
from src.triangulate import draw_triangles
import numpy as np

def test_nik9c_integrity():
    print("Testing nik9c.triv integrity...")
    try:
        decoder = TriangleDecoder('nik9c.triv')
        print(f"Header: {decoder.width}x{decoder.height} @ {decoder.fps}fps")
        
        frames_to_test = 5
        for i in range(frames_to_test):
            start = time.time()
            frame_data = decoder.read_frame()
            if frame_data is None:
                print(f"Unexpected EOF at frame {i}")
                break
                
            frame_type, points, colors, simplices, comp_size = frame_data
            # Verify the shapes
            assert points.ndim == 2, "Points should be 2D"
            assert simplices.ndim == 2, "Simplices should be 2D"
            assert colors.ndim == 2, "Colors should be 2D"
            assert simplices.shape[0] == colors.shape[0], "Simplex and Color count mismatch"
            
            # Optionally render a test frame to verify no visual corruption
            _ = draw_triangles((decoder.height, decoder.width), points, simplices, colors)
            
            elapsed = (time.time() - start) * 1000
            print(f"Frame {i}: {len(colors)} triangles, {comp_size:,} bytes compressed, decoded in {elapsed:.2f}ms")
            
        print("Integrity check PASSED!")
        decoder.close()
    except Exception as e:
        print(f"Integrity check FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nik9c_integrity()
