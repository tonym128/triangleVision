import cv2
import numpy as np
from scipy.spatial import Delaunay

def compute_complexity(frame):
    """Fast edge-based complexity."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges > 0)

def determine_triangle_count(complexity, quality):
    max_triangles = {'low': 800, 'medium': 2500, 'high': 6000}
    min_triangles = {'low': 150, 'medium': 500, 'high': 1500}
    q_max = max_triangles.get(quality, 2500)
    q_min = min_triangles.get(quality, 500)
    norm = min(complexity / 0.1, 1.0)
    return int(q_min + (q_max - q_min) * norm)

def generate_points(frame, num_triangles):
    """High-performance point generation with Emboss-based feature detection."""
    num_points = max(int(num_triangles / 2) + 3, 10)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. NEW FEATURE: Emboss-based detail extraction
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    embossed = cv2.filter2D(gray, -1, kernel)

    # 2. Extract anchor points from standard features
    feat_pts = cv2.goodFeaturesToTrack(gray, maxCorners=int(num_points * 0.7), qualityLevel=0.01, minDistance=4)

    # 3. Extract anchor points from the EMBOSSED detail map
    emboss_pts = cv2.goodFeaturesToTrack(embossed, maxCorners=int(num_points * 0.3), qualityLevel=0.02, minDistance=4)

    points_list = []
    if feat_pts is not None:
        points_list.append(feat_pts.reshape(-1, 2))
    if emboss_pts is not None:
        points_list.append(emboss_pts.reshape(-1, 2))

    if len(points_list) > 0:
        points = np.vstack(points_list)
    else:
        points = np.empty((0, 2), dtype=np.float32)

    # 4. Fill remaining points with a fast jittered grid
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
    return points.astype(np.float32)

def get_triangles_and_colors(frame, points, prev_colors=None, smoothing=0.3):
    """Vectorized color sampling with parity quantization."""
    h, w = frame.shape[:2]
    points = np.floor(points).astype(np.float32)
    processed = cv2.GaussianBlur(frame, (3, 3), 0)
    processed = (processed >> 4) << 4
    tri = Delaunay(points)
    simplices = tri.simplices
    triangle_pts = points[simplices]
    centroids = np.mean(triangle_pts, axis=1).astype(np.int32)
    centroids[:, 0] = np.clip(centroids[:, 0], 0, w - 1)
    centroids[:, 1] = np.clip(centroids[:, 1], 0, h - 1)
    colors = processed[centroids[:, 1], centroids[:, 0]]
    if prev_colors is not None and len(prev_colors) == len(colors):
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

def draw_triangles(frame_shape, points, simplices, colors, rotoscope=False, heatmap=False):
    """Optimized rendering with optional heatmap overlay."""
    h, w = frame_shape[:2]
    out_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(simplices)):
        pts = points[simplices[i]].astype(np.int32)
        cv2.fillPoly(out_frame, [pts], colors[i].tolist(), lineType=cv2.LINE_AA)
    if rotoscope:
        all_pts = points[simplices].astype(np.int32)
        cv2.polylines(out_frame, all_pts, isClosed=True, color=(15, 15, 15), thickness=1, lineType=cv2.LINE_AA)
    if heatmap:
        h_map = draw_heatmap(frame_shape, points)
        cv2.addWeighted(h_map, 0.4, out_frame, 0.6, 0, out_frame)
    return out_frame
