import argparse
import cv2
import time
import numpy as np
import struct
import zlib
from src.triangulate import get_triangles_and_colors, draw_triangles, generate_points, compute_complexity, determine_triangle_count
from src.codec import TriangleEncoder, TriangleDecoder
import os
import sys

def realtime_mode(source=0, target_triangles=None, quality='medium', rotoscope=False, output_path=None):
    # Try to convert source to int if it's a digit (for webcam IDs)
    try:
        source_input = int(source)
    except (ValueError, TypeError):
        source_input = source

    print(f"Starting processing on source: {source_input}...")
    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"Error: Could not open source {source_input}")
        return

    # Get video properties for encoder if needed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback for webcams

    encoder = None
    if output_path:
        print(f"Recording to {output_path}...")
        encoder = TriangleEncoder(output_path, width, height, fps, target_triangles, quality)

    prev_colors = None
    frame_count = 0
    show_heatmap = False
    
    cv2.namedWindow('TriangleVision - Realtime', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Every frame is independent now for maximum speed and zero snapping
        if target_triangles is not None:
            num_triangles = target_triangles
        else:
            complexity = compute_complexity(frame)
            num_triangles = determine_triangle_count(complexity, quality)
            
        points = generate_points(frame, num_triangles)
        simplices, colors = get_triangles_and_colors(frame, points, prev_colors)
        out_frame = draw_triangles(frame.shape, points, simplices, colors, rotoscope=rotoscope, heatmap=show_heatmap)
        
        # Save exact points and colors to file
        if encoder:
            encoder.add_frame(frame, manual_points=points, manual_colors=colors)

        # Calculate Mock Byte Size for the frame for display
        num_points = len(points)
        num_triangles = len(colors)
        pts_data = points.astype(np.uint16).tobytes()
        colors_data = colors.tobytes()
        raw_data = struct.pack('<H', num_points) + pts_data + struct.pack('<H', num_triangles) + colors_data
        compressed_size = len(zlib.compress(raw_data, level=4))

        fps_val = 1.0 / (time.time() - start_time)
        info_text = f"FPS: {fps_val:.1f} Triangles: {len(colors)} Size: {compressed_size:,}"
        if encoder:
            info_text += " [RECORDING]"
            
        cv2.putText(out_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('TriangleVision - Realtime', out_frame)
        
        # Pause on the first frame and wait for user input
        if frame_count == 0:
            print("First frame rendered. Press SPACE to start playback, P for heatmap, or 'q' to quit.")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('p'):
                    show_heatmap = not show_heatmap
                    out_frame = draw_triangles(frame.shape, points, simplices, colors, rotoscope=rotoscope, heatmap=show_heatmap)
                    cv2.putText(out_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('TriangleVision - Realtime', out_frame)
                if key == ord('q'):
                    if encoder: encoder.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        prev_colors = colors
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_heatmap = not show_heatmap

    if encoder:
        encoder.close()
    cap.release()
    cv2.destroyAllWindows()


def encode_video(input_path, output_path, target_triangles=None, quality='medium'):
    print(f"Encoding {input_path} to {output_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Optional scaling down for very large videos
    max_dim = 1280
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
    
    encoder = TriangleEncoder(output_path, width, height, fps, target_triangles, quality)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        encoder.add_frame(frame)
        frame_count += 1
        sys.stdout.write(f"\rProcessed {frame_count}/{total_frames} frames")
        sys.stdout.flush()
        
    print("\nEncoding complete!")
    encoder.close()
    cap.release()

def play_video(input_path, rotoscope=False):
    from scipy.spatial import Delaunay
    
    print(f"Playing {input_path}...")
    try:
        decoder = TriangleDecoder(input_path)
    except Exception as e:
        print(f"Failed to load video: {e}")
        return
        
    cv2.namedWindow('TriangleVision - Player', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TriangleVision - Player', decoder.width, decoder.height)
    
    delay = int(1000 / decoder.fps)
    show_heatmap = False
    
    while True:
        start_time = time.time()
        frame_data = decoder.read_frame()
        if frame_data is None:
            break
            
        frame_type, points, colors, compressed_size = frame_data
        
        # We need to triangulate on the fly since we didn't save simplices
        try:
            tri = Delaunay(points)
            simplices = tri.simplices
        except Exception as e:
            print(f"Triangulation error: {e}")
            break
            
        out_frame = draw_triangles((decoder.height, decoder.width), points, simplices, colors, rotoscope=rotoscope, heatmap=show_heatmap)
        
        info_text = f"Triangles: {len(colors)} Size: {compressed_size:,}"
        cv2.putText(out_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('TriangleVision - Player', out_frame)
        
        elapsed = (time.time() - start_time) * 1000
        wait_time = max(1, int(delay - elapsed))
        
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_heatmap = not show_heatmap
            
    decoder.close()
    cv2.destroyAllWindows()

def export_video(input_path, output_path, rotoscope=False):
    from scipy.spatial import Delaunay
    
    print(f"Exporting {input_path} to standard video {output_path}...")
    try:
        decoder = TriangleDecoder(input_path)
    except Exception as e:
        print(f"Failed to load video: {e}")
        return
        
    # Set up VideoWriter with modern H.264 compression
    # Try 'avc1' (H.264) first as it is the most efficient
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # Check if we can set quality (supported in newer OpenCV versions)
    # 0-100, where 23-30 is usually a good sweet spot for size/quality
    params = [cv2.VIDEOWRITER_PROP_QUALITY, 80] 
    
    out = cv2.VideoWriter(output_path, fourcc, decoder.fps, (decoder.width, decoder.height), params)
    
    if not out.isOpened():
        # Fallback for systems without H.264 support
        print("Warning: H.264 (avc1) not supported. Falling back to mp4v (larger file size).")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, decoder.fps, (decoder.width, decoder.height))
    
    frame_count = 0
    while True:
        frame_data = decoder.read_frame()
        if frame_data is None:
            break
            
        frame_type, points, colors, _ = frame_data
        
        try:
            tri = Delaunay(points)
            simplices = tri.simplices
        except Exception as e:
            print(f"Triangulation error at frame {frame_count}: {e}")
            break
            
        out_frame = draw_triangles((decoder.height, decoder.width), points, simplices, colors, rotoscope=rotoscope)
        out.write(out_frame)
        
        frame_count += 1
        sys.stdout.write(f"\rExporting frame {frame_count}")
        sys.stdout.flush()
            
    print("\nExport complete!")
    out.release()
    decoder.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TriangleVision - Convert video to triangles")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Realtime Command
    parser_rt = subparsers.add_parser("realtime", help="Run realtime webcam or file conversion")
    parser_rt.add_argument("--source", type=str, default="0", help="Camera source ID or video file path")
    parser_rt.add_argument("--triangles", type=int, default=None, help="Target number of triangles (overrides quality)")
    parser_rt.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="Adaptive quality level")
    parser_rt.add_argument("--rotoscope", action="store_true", help="Enable rotoscoping mode (ink edges and cell shading)")
    parser_rt.add_argument("--output", type=str, default=None, help="Optionally save the realtime stream to a .triv file")
    
    # Encode Command
    parser_enc = subparsers.add_parser("encode", help="Encode an MP4/video to .triv format")
    parser_enc.add_argument("input", help="Input video path")
    parser_enc.add_argument("output", help="Output .triv file path")
    parser_enc.add_argument("--triangles", type=int, default=None, help="Target number of triangles (overrides quality)")
    parser_enc.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="Adaptive quality level")
    
    # Play Command
    parser_play = subparsers.add_parser("play", help="Play a .triv format video")
    parser_play.add_argument("input", help="Input .triv file path")
    parser_play.add_argument("--rotoscope", action="store_true", help="Enable rotoscoping mode (ink edges and cell shading)")

    # Export Command
    parser_exp = subparsers.add_parser("export", help="Export a .triv file back to standard MP4/MKV")
    parser_exp.add_argument("input", help="Input .triv file path")
    parser_exp.add_argument("output", help="Output standard video path (.mp4 or .mkv)")
    parser_exp.add_argument("--rotoscope", action="store_true", help="Enable rotoscoping mode for export")
    
    args = parser.parse_args()
    
    if args.command == "realtime":
        realtime_mode(args.source, args.triangles, args.quality, args.rotoscope, output_path=args.output)
    elif args.command == "encode":
        encode_video(args.input, args.output, args.triangles, args.quality)
    elif args.command == "play":
        play_video(args.input, args.rotoscope)
    elif args.command == "export":
        export_video(args.input, args.output, args.rotoscope)
    else:
        parser.print_help()
