import struct
import zlib
import cv2
import numpy as np
from src.triangulate import generate_points, get_triangles_and_colors, determine_triangle_count, compute_complexity

class TriangleEncoder:
    def __init__(self, output_path, width, height, fps, target_triangles=None, quality='medium'):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = int(fps)
        self.target_triangles = target_triangles
        self.quality = quality
        
        self.file = open(self.output_path, 'wb')
        
        # Write Header: TRIV + W + H + FPS
        header = struct.pack('<4sHHH', b'TRIV', self.width, self.height, self.fps)
        self.file.write(header)
        
        self.prev_colors = None

    def add_frame(self, frame, manual_points=None, manual_colors=None):
        # Resize if necessary
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        if manual_points is not None and manual_colors is not None:
            # Use pre-calculated data (quantized to match storage format)
            points = np.floor(manual_points).astype(np.float32)
            colors = manual_colors
        else:
            # Standard calculation
            if self.target_triangles is not None:
                num_triangles = self.target_triangles
            else:
                complexity = compute_complexity(frame)
                num_triangles = determine_triangle_count(complexity, self.quality)

            points = generate_points(frame, num_triangles)
            # This internal call will also quantize inside src/triangulate.py
            simplices, colors = get_triangles_and_colors(frame, points, self.prev_colors)
            self.prev_colors = colors

        # Serialize data
        num_points = len(points)
        num_triangles = len(colors)

        
        pts_data = points.astype(np.uint16).tobytes()
        colors_data = colors.tobytes()
        
        raw_data = struct.pack('<H', num_points) + pts_data + struct.pack('<H', num_triangles) + colors_data
        compressed_data = zlib.compress(raw_data, level=4)
        
        # Write frame header
        frame_header = struct.pack('<cI', b'I', len(compressed_data))
        self.file.write(frame_header)
        self.file.write(compressed_data)
        
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
        
        # Parse Raw Data
        offset = 0
        num_points = struct.unpack('<H', raw_data[offset:offset+2])[0]
        offset += 2
        
        pts_size = num_points * 4 # 2 uint16s per point
        pts_data = raw_data[offset:offset+pts_size]
        points = np.frombuffer(pts_data, dtype=np.uint16).reshape(-1, 2).astype(np.float32)
        offset += pts_size
        
        num_triangles = struct.unpack('<H', raw_data[offset:offset+2])[0]
        offset += 2
        
        colors_size = num_triangles * 3
        colors_data = raw_data[offset:offset+colors_size]
        colors = np.frombuffer(colors_data, dtype=np.uint8).reshape(-1, 3)
        
        return frame_type, points, colors, data_len
        
    def close(self):
        self.file.close()
