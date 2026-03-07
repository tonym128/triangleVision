import cv2
import numpy as np
from scipy.spatial import Delaunay
from numba import njit, prange

@njit(parallel=True)
def sample_colors_numba(processed_frame, centroids, h, w):
    num_triangles = centroids.shape[0]
    colors = np.empty((num_triangles, 3), dtype=np.uint8)
    for i in prange(num_triangles):
        cx = centroids[i, 0]
        cy = centroids[i, 1]
        if cx < 0: cx = 0
        if cx >= w: cx = w - 1
        if cy < 0: cy = 0
        if cy >= h: cy = h - 1
        colors[i, 0] = processed_frame[cy, cx, 0]
        colors[i, 1] = processed_frame[cy, cx, 1]
        colors[i, 2] = processed_frame[cy, cx, 2]
    return colors

def compute_complexity(frame):
    """Fast edge-based complexity."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Using a smaller scale for complexity check to speed up
    small_gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    edges = cv2.Canny(small_gray, 100, 200)
    return np.mean(edges > 0)

def determine_triangle_count(complexity, quality):
    max_triangles = {'low': 800, 'medium': 2500, 'high': 6000}
    min_triangles = {'low': 150, 'medium': 500, 'high': 1500}
    q_max = max_triangles.get(quality, 2500)
    q_min = min_triangles.get(quality, 500)
    norm = min(complexity / 0.1, 1.0)
    return int(q_min + (q_max - q_min) * norm)

_hog = None
_face_cascade = None
_body_cascade = None
_upper_cascade = None

def get_hog():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog

def get_cascade(name):
    import os
    cascade_path = os.path.join(cv2.data.haarcascades, name)
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    return None

def generate_points(frame, num_triangles, prev_gray=None, detect_human=False, human_mask=None):
    """Optimized point generation with multi-cascade human/face/body detection."""
    num_points = max(int(num_triangles / 2) + 3, 10)
    h, w = frame.shape[:2]
    
    # 1. DOWN-SCALE for feature detection (huge speedup)
    detect_w = 320
    scale = detect_w / w
    sh, sw = int(h * scale), detect_w
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray_full, (sw, sh))

    # 2. Primary Features (Shi-Tomasi)
    feat_pts = cv2.goodFeaturesToTrack(gray, maxCorners=int(num_points * 0.4), qualityLevel=0.01, minDistance=int(4 * scale))
    
    # 3. MULTI-LAYER Human Detection
    human_pts = None
    if detect_human or human_mask is not None:
        if human_mask is None:
            human_mask = np.zeros_like(gray)
            found_anything = False
            
            # A. Body Detection (HOG - tuned for speed)
            hog = get_hog()
            (rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
            rects_list = rects.tolist() if isinstance(rects, np.ndarray) else list(rects)
            if len(rects_list) > 0:
                rects_list, _ = cv2.groupRectangles(rects_list, 1, 0.2)
                for (rx, ry, rw, rh) in rects_list:
                    cv2.rectangle(human_mask, (rx, ry), (rx + rw, ry + rh), 255, -1)
                    found_anything = True

            # B. Haar Cascades
            cascades = [
                ('haarcascade_frontalface_default.xml', 1.1, 5, 0.2), 
                ('haarcascade_fullbody.xml', 1.1, 3, 0.05)
            ]
            
            for cascade_name, scale_factor, min_neighbors, pad_ratio in cascades:
                cascade = get_cascade(cascade_name)
                if cascade:
                    found_rects = cascade.detectMultiScale(gray, scale_factor, min_neighbors)
                    for (fx, fy, fw, fh) in found_rects:
                        pad = int(fw * pad_ratio)
                        cv2.rectangle(human_mask, (fx-pad, fy-pad), (fx + fw + pad, fy + fh + pad), 255, -1)
                        found_anything = True
        
        # If we have a mask (either just created or passed in), sample points from it
        if human_mask is not None and np.any(human_mask > 0):
            # Allocate 30% of points specifically to human areas
            human_pts = cv2.goodFeaturesToTrack(gray, maxCorners=int(num_points * 0.30), qualityLevel=0.005, minDistance=int(2 * scale), mask=human_mask)

    # 4. Motion (10%)
    motion_pts = None
    if prev_gray is not None and prev_gray.shape == gray_full.shape:
        prev_gray_scaled = cv2.resize(prev_gray, (sw, sh))
        diff = cv2.absdiff(prev_gray_scaled, gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_pts = cv2.goodFeaturesToTrack(gray, maxCorners=int(num_points * 0.1), qualityLevel=0.01, minDistance=int(4 * scale), mask=motion_mask)

    # 4. Center biased (5%) - Reduced to make room for human/face
    center_mask = np.zeros_like(gray)
    ch, cw = sh // 2, sw // 2
    rh, rw = sh // 4, sw // 4
    center_mask[ch-rh:ch+rh, cw-rw:cw+rw] = 255
    center_pts = cv2.goodFeaturesToTrack(gray, maxCorners=int(num_points * 0.05), qualityLevel=0.01, minDistance=int(4 * scale), mask=center_mask)

    points_list = []
    if feat_pts is not None: points_list.append(feat_pts.reshape(-1, 2) / scale)
    if human_pts is not None: points_list.append(human_pts.reshape(-1, 2) / scale)
    if motion_pts is not None: points_list.append(motion_pts.reshape(-1, 2) / scale)
    if center_pts is not None: points_list.append(center_pts.reshape(-1, 2) / scale)

    if len(points_list) > 0:
        points = np.vstack(points_list)
    else:
        points = np.empty((0, 2), dtype=np.float32)

    # 5. Fill remaining points with a fast jittered grid
    remaining = num_points - len(points) - 4
    if remaining > 0:
        rows = int(np.sqrt(remaining * (h/w)))
        cols = int(remaining / rows) if rows > 0 else 1
        yy, xx = np.mgrid[0:h:complex(0, rows), 0:w:complex(0, cols)]
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        grid_points += np.random.uniform(-5, 5, grid_points.shape)
        points = np.vstack([points, grid_points]) if len(points) > 0 else grid_points

    # Clamp and add corners
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    points = np.clip(points, 0, [w-1, h-1])
    points = np.vstack([points, corners])
    
    # Return all points, specific human points for visualization, and the mask for persistence
    return points.astype(np.float32), (human_pts / scale if human_pts is not None else None), human_mask

def get_triangles_and_colors(frame, points, prev_colors=None, smoothing=0.3):
    """Vectorized color sampling with optimized NumPy indexing."""
    h, w = frame.shape[:2]
    # Fast blur and quantization
    processed = cv2.GaussianBlur(frame, (3, 3), 0)
    processed &= 0xF0 
    
    tri = Delaunay(points)
    simplices = tri.simplices
    triangle_pts = points[simplices]
    centroids = np.mean(triangle_pts, axis=1).astype(np.int32)
    
    # Clip centroids to ensure they are within bounds
    cx = np.clip(centroids[:, 0], 0, w - 1)
    cy = np.clip(centroids[:, 1], 0, h - 1)
    
    # Fast NumPy indexing
    colors = processed[cy, cx]
    
    if prev_colors is not None and len(prev_colors) == len(colors):
        # Vectorized smoothing
        colors = (colors.astype(np.float32) * smoothing + 
                  prev_colors.astype(np.float32) * (1.0 - smoothing)).astype(np.uint8)
    return simplices, colors

def draw_heatmap(frame_shape, points):
    """Generates a semi-transparent heatmap showing triangle density."""
    h, w = frame_shape[:2]
    res_h, res_w = 40, 40
    heatmap, xedges, yedges = np.histogram2d(
        points[:, 1], points[:, 0], bins=[res_h, res_w], range=[[0, h], [0, w]]
    )
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    cv2.normalize(heatmap, heatmap, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color_heatmap = cv2.resize(color_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    return color_heatmap

def draw_triangles(frame_shape, points, simplices, colors, rotoscope=False, heatmap=False, human_points=None):
    """Optimized rendering using fillConvexPoly and fewer Python overheads."""
    h, w = frame_shape[:2]
    out_frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Pre-convert points to int32 once
    pts_int = points.astype(np.int32)
    
    # Use fillConvexPoly which is generally faster than fillPoly for single triangles
    # and disable anti-aliasing for maximum speed
    for i in range(len(simplices)):
        pts = pts_int[simplices[i]]
        cv2.fillConvexPoly(out_frame, pts, colors[i].tolist())
        
    if rotoscope:
        all_pts = pts_int[simplices]
        # polylines can take a list of contours
        cv2.polylines(out_frame, all_pts, isClosed=True, color=(15, 15, 15), thickness=1)
        
    if heatmap:
        h_map = draw_heatmap(frame_shape, points)
        cv2.addWeighted(h_map, 0.4, out_frame, 0.6, 0, out_frame)
        
        # ADD GREEN DOTS for human detection points
        if human_points is not None:
            # Reshape from (N, 1, 2) to (N, 2) to avoid indexing errors
            for pt in human_points.reshape(-1, 2).astype(np.int32):
                cv2.circle(out_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                
    return out_frame
