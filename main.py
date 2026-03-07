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
import threading
from queue import Queue

class ThreadedVideoGetter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread and stores them in a queue.
    """
    def __init__(self, src=0, queue_size=128):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):    
        t = threading.Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):
        while not self.stopped:
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                # If queue is full, wait a bit to avoid CPU pegging
                time.sleep(0.001)

    def read(self):
        # Returns (True, frame) if available, (False, None) otherwise
        if not self.Q.empty():
            return True, self.Q.get()
        return False, None

    def more(self):
        # Check if there are frames left in the queue
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        time.sleep(0.1) # Give thread time to finish
        self.stream.release()

def realtime_mode(source=0, target_triangles=None, quality='medium', rotoscope=False, output_path=None, detect_human=False):
    # Try to convert source to int if it's a digit (for webcam IDs)
    try:
        source_input = int(source)
    except (ValueError, TypeError):
        source_input = source

    print(f"Starting processing on source: {source_input}...")
    
    # Increase queue size to handle potential lag spikes
    video_getter = ThreadedVideoGetter(source_input, queue_size=128).start()
    
    # Wait for the first frame to determine video properties
    while not video_getter.more() and not video_getter.stopped:
        time.sleep(0.01)

    if video_getter.stopped:
        print(f"Error: Could not open source {source_input}")
        return

    # Get properties from the stream before processing
    width = int(video_getter.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_getter.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_getter.stream.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback for webcams

    encoder = None
    if output_path:
        print(f"Recording to {output_path}...")
        encoder = TriangleEncoder(output_path, width, height, fps, target_triangles, quality, detect_human=detect_human)

    prev_colors = None
    prev_gray = None
    frame_count = 0
    show_heatmap = False
    
    cv2.namedWindow('TriangleVision - Realtime', cv2.WINDOW_NORMAL)

    while True:
        # If recording, we MUST process every frame.
        # If NOT recording, we can skip ahead to the latest frame if the queue is backing up
        if not encoder:
            # Skip logic to stay "live"
            while video_getter.Q.qsize() > 1:
                video_getter.read() 
                
        ret, frame = video_getter.read()
        if not ret:
            if video_getter.stopped:
                break
            time.sleep(0.001)
            continue
            
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Every frame is independent for maximum speed and zero snapping
        if target_triangles is not None:
            num_triangles = target_triangles
        else:
            complexity = compute_complexity(frame)
            num_triangles = determine_triangle_count(complexity, quality)
            
        points, h_pts = generate_points(frame, num_triangles, prev_gray, detect_human=detect_human)
        simplices, colors = get_triangles_and_colors(frame, points, prev_colors)
        out_frame = draw_triangles(frame.shape, points, simplices, colors, rotoscope=rotoscope, heatmap=show_heatmap, human_points=h_pts)
        
        # Save exact points and colors to file
        if encoder:
            encoder.add_frame(frame, manual_points=points, manual_colors=colors, manual_simplices=simplices)

        # UI Info and display
        q_size = video_getter.Q.qsize()
        fps_val = 1.0 / (time.time() - start_time)
        info_text = f"FPS: {fps_val:.1f} Triangles: {len(colors)} Q:{q_size}"
        if encoder:
            info_text += " [RECORDING]"
        if q_size > 100:
            info_text += " [LAGGING]"
            
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
                    out_frame = draw_triangles(frame.shape, points, simplices, colors, rotoscope=rotoscope, heatmap=show_heatmap, human_points=h_pts)
                    cv2.putText(out_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('TriangleVision - Realtime', out_frame)
                if key == ord('q'):
                    if encoder: encoder.close()
                    video_getter.stop()
                    cv2.destroyAllWindows()
                    return

        prev_colors = colors
        prev_gray = gray
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_heatmap = not show_heatmap

    if encoder:
        encoder.close()
    video_getter.stop()
    cv2.destroyAllWindows()


def encode_video(input_path, output_path, target_triangles=None, quality='medium', detect_human=False):
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
    
    encoder = TriangleEncoder(output_path, width, height, fps, target_triangles, quality, detect_human=detect_human)
    
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
            
        frame_type, points, colors, simplices, compressed_size = frame_data
        
        out_frame = draw_triangles((decoder.height, decoder.width), points, simplices, colors, rotoscope=rotoscope, heatmap=show_heatmap, human_points=None)
        
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
            
        frame_type, points, colors, simplices, _ = frame_data
        
        out_frame = draw_triangles((decoder.height, decoder.width), points, simplices, colors, rotoscope=rotoscope, human_points=None)
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
    parser_rt.add_argument("--human", action="store_true", help="Enable human detection and enhance detail on people")
    parser_rt.add_argument("--output", type=str, default=None, help="Optionally save the realtime stream to a .triv file")
    
    # Encode Command
    parser_enc = subparsers.add_parser("encode", help="Encode an MP4/video to .triv format")
    parser_enc.add_argument("input", help="Input video path")
    parser_enc.add_argument("output", help="Output .triv file path")
    parser_enc.add_argument("--triangles", type=int, default=None, help="Target number of triangles (overrides quality)")
    parser_enc.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="Adaptive quality level")
    parser_enc.add_argument("--human", action="store_true", help="Enable human detection and enhance detail on people")
    
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
        realtime_mode(args.source, args.triangles, args.quality, args.rotoscope, output_path=args.output, detect_human=args.human)
    elif args.command == "encode":
        encode_video(args.input, args.output, args.triangles, args.quality, detect_human=args.human)
    elif args.command == "play":
        play_video(args.input, args.rotoscope)
    elif args.command == "export":
        export_video(args.input, args.output, args.rotoscope)
    else:
        parser.print_help()
