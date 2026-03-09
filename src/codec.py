import struct
import zlib
import cv2
import numpy as np
from src.triangulate import generate_points, get_triangles_and_colors, determine_triangle_count, compute_complexity

class TriangleEncoder:
    def __init__(self, output_path, width, height, fps, target_triangles=None, quality='medium', detect_human=False):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = int(fps)
        self.target_triangles = target_triangles
        self.quality = quality
        self.detect_human = detect_human
        
        self.file = open(self.output_path, 'wb')
        
        # Write Header: TRIV + W + H + FPS
        header = struct.pack('<4sHHH', b'TRIV', self.width, self.height, self.fps)
        self.file.write(header)
        
        self.prev_colors = None
        self.prev_gray = None

    def add_frame(self, frame, manual_points=None, manual_colors=None, manual_simplices=None):
        # Resize if necessary
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if manual_points is not None and manual_colors is not None and manual_simplices is not None:
            # Use pre-calculated data
            points = manual_points.astype(np.float32)
            colors = manual_colors
            simplices = manual_simplices
        else:
            # Standard calculation
            if self.target_triangles is not None:
                num_triangles = self.target_triangles
            else:
                complexity = compute_complexity(frame)
                num_triangles = determine_triangle_count(complexity, self.quality)

            points, _, _ = generate_points(frame, num_triangles, self.prev_gray, detect_human=self.detect_human)
            simplices, colors = get_triangles_and_colors(frame, points, self.prev_colors)
            self.prev_colors = colors

        # Serialize data
        num_points = len(points)
        num_simplices = len(simplices)

        # Quantize points to uint16 for storage
        pts_data = points.astype(np.uint16).tobytes()
        # Simplices are indices, usually small enough for uint16 if points < 65535
        simplices_data = simplices.astype(np.uint16).tobytes()
        colors_data = colors.tobytes()

        raw_data = struct.pack('<HHH', num_points, num_simplices, len(colors))
        raw_data += pts_data + simplices_data + colors_data
        
        compressed_data = zlib.compress(raw_data, level=4)

        # Write frame header
        frame_header = struct.pack('<cI', b'I', len(compressed_data))
        self.file.write(frame_header)
        self.file.write(compressed_data)
        self.prev_gray = gray
        
    def close(self):
        self.file.close()

class TriangleDecoder:
    def __init__(self, input_path):
        self.input_path = input_path
        self.file = open(self.input_path, 'rb')
        
        # Read Header
        header = self.file.read(10)
        magic, self.width, self.height, self.fps = struct.unpack('<4sHHH', header)
        
        if magic != b'TRIV':
            raise ValueError("Not a valid TRIV file")
            
    def read_frame(self):
        # Read Frame Header
        header = self.file.read(5)
        if not header or len(header) < 5:
            return None # EOF
            
        frame_type, data_len = struct.unpack('<cI', header)
        
        # Read Compressed Data
        compressed_data = self.file.read(data_len)
        raw_data = zlib.decompress(compressed_data)
        
        # Parse Raw Data (Updated format: n_pts, n_simplices, n_colors)
        offset = 0
        num_points, num_simplices, num_colors = struct.unpack('<HHH', raw_data[offset:offset+6])
        offset += 6
        
        pts_size = num_points * 4 # 2 uint16s per point
        pts_data = raw_data[offset:offset+pts_size]
        points = np.frombuffer(pts_data, dtype=np.uint16).reshape(-1, 2).astype(np.float32)
        offset += pts_size
        
        simplices_size = num_simplices * 6 # 3 uint16s per simplex
        simplices_data = raw_data[offset:offset+simplices_size]
        simplices = np.frombuffer(simplices_data, dtype=np.uint16).reshape(-1, 3).astype(np.int32)
        offset += simplices_size
        
        colors_size = num_colors * 3
        colors_data = raw_data[offset:offset+colors_size]
        colors = np.frombuffer(colors_data, dtype=np.uint8).reshape(-1, 3)
        
        return frame_type, points, colors, simplices, data_len
        
    def close(self):
        self.file.close()
