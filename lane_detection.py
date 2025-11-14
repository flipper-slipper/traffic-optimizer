import cv2
from ultralytics import YOLO
import numpy as np

def create_tracker(frame, coords):
    # convert center format to top-left for opencv tracker
    x_center, y_center, w, h = coords
    x = int(x_center - w / 2)
    y = int(y_center - h / 2)
    tracker = cv2.TrackerMIL.create()
    tracker.init(frame, (x, y, int(w), int(h)))
    return tracker

def update_tracker(tracker, frame):
    ret, bbox = tracker.update(frame)
    if not ret:
        return None
    # get center point
    cx = int(bbox[0] + bbox[2] / 2)
    cy = int(bbox[1] + bbox[3] / 2)
    return (cx, cy)

def x_at_y(line, y):
    p1, p2 = line
    # handle horizontal lines
    if abs(p2[1] - p1[1]) < 0.000001:
        return (p1[0] + p2[0]) / 2.0
    t = (y - p1[1]) / (p2[1] - p1[1])
    return p1[0] + t * (p2[0] - p1[0])

def line_dist(line1, line2, y_coord):
    x1 = x_at_y(line1, y_coord)
    x2 = x_at_y(line2, y_coord)
    return abs(x1 - x2)

def avg_lines(lines):
    if len(lines) == 0:
        return None, None
    if len(lines) == 1:
        return lines[0]
    
    all_ys = [line[0][1] for line in lines if line[0] is not None and line[1] is not None]
    all_ys.extend([line[1][1] for line in lines if line[0] is not None and line[1] is not None])
    
    if len(all_ys) == 0:
        return None, None
    
    y_min = int(min(all_ys))
    y_max = int(max(all_ys))
    y_samples = np.linspace(y_min, y_max, 10)
    
    x_values = []
    for y in y_samples:
        x_list = []
        for line in lines:
            if line[0] is not None and line[1] is not None:
                p1, p2 = line
                line_y_min = min(p1[1], p2[1])
                line_y_max = max(p1[1], p2[1])
                if line_y_min <= y <= line_y_max:
                    x_list.append(x_at_y(line, y))
        if len(x_list) > 0:
            x_values.append(np.mean(x_list))
        else:
            x_values.append(None)
    
    valid_points = [(x, y_samples[i]) for i, x in enumerate(x_values) if x is not None]
    
    if len(valid_points) < 2:
        return None, None
    
    return fit_line(np.array(valid_points))

def get_line_length(line):
    if line[0] is None or line[1] is None:
        return 0
    p1, p2 = line
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_line_x_at_midpoint(line):
    if line[0] is None or line[1] is None:
        return None
    p1, p2 = line
    mid_y = (p1[1] + p2[1]) / 2.0
    return x_at_y(line, mid_y)

def select_lines(lane_lines_with_count, n=4):
    if len(lane_lines_with_count) <= n:
        return [line for line, _ in lane_lines_with_count]
    
    car_counts = [count for _, count in lane_lines_with_count]
    lengths = [get_line_length(line) for line, _ in lane_lines_with_count]
    
    if len(car_counts) > 0:
        max_cars = max(car_counts)
    else:
        max_cars = 1
    if len(lengths) > 0:
        max_length = max(lengths)
    else:
        max_length = 1
    
    scores = []
    for i, (line, count) in enumerate(lane_lines_with_count):
        if max_cars > 0:
            car_score = count / max_cars
        else:
            car_score = 0
        if max_length > 0:
            length_score = lengths[i] / max_length
        else:
            length_score = 0
        score = 0.4 * car_score + 0.6 * length_score
        scores.append((score, line))
    
    scores.sort(reverse=True)
    return [line for _, line in scores[:n]]

def cluster(lane_lines, distance_threshold=30, num_test_points=5):
    valid_lines = []
    valid_indices = []
    for idx, line in enumerate(lane_lines):
        if line[0] is not None and line[1] is not None:
            valid_lines.append(line)
            valid_indices.append(idx)
    
    if len(valid_lines) == 0:
        return []
    
    all_ys = []
    for line in valid_lines:
        all_ys.append(line[0][1])
        all_ys.append(line[1][1])
    y_min = min(all_ys)
    y_max = max(all_ys)
    test_ys = np.linspace(y_min, y_max, num_test_points)
    
    n = len(valid_lines)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dists = [line_dist(valid_lines[i], valid_lines[j], y) for y in test_ys]
            avg_dist = np.mean(dists)
            distances[i, j] = avg_dist
            distances[j, i] = avg_dist
    
    clusters = []
    used = set()
    for i in range(n):
        if i in used:
            continue
        cluster = [valid_indices[i]]
        used.add(i)
        for j in range(n):
            if j not in used and distances[i, j] <= distance_threshold:
                cluster.append(valid_indices[j])
                used.add(j)
        clusters.append(cluster)
    return clusters

def fit_line(points):
    if len(points) < 2:
        return None, None
    
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    coeffs = np.polyfit(x, y, 1)
    m = coeffs[0]
    b = coeffs[1]
    x_min = int(np.min(x))
    x_max = int(np.max(x))
    y_min = int(m * x_min + b)
    y_max = int(m * x_max + b)
    return (x_min, y_min), (x_max, y_max)

cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

model = YOLO('yolo11n.pt')
model.to('cuda')
print(f"Using {model.device}")

video = cv2.VideoCapture('dataset/example_recording.webm')

fps = video.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
print(f"FPS: {fps}")

ret, frame = video.read()
if not ret:
    print("Can't read video")
    exit(1)

print("\nLooking for cars...")
results = model.predict(frame, conf=0.1, max_det=50, verbose=False, half=True, imgsz=640)
bboxes = results[0].boxes.xywh.cpu().numpy()

if len(bboxes) == 0:
    print("No cars found")
    exit(1)

num_cars = len(bboxes)
print(f"Found {num_cars} cars")
for i in range(num_cars):
    bbox = bboxes[i]
    area = bbox[2] * bbox[3]
    print(f"  {i+1}: area={area:.0f}")

# initialize trackers
trackers = []
for i in range(num_cars):
    coords = bboxes[i].tolist()
    tracker = create_tracker(frame, coords)
    trackers.append(tracker)

TRACKING_DURATION = 3.0
SAMPLE_INTERVAL = 1.0

car_tracks = [[] for _ in range(num_cars)]

# colors for visualization
car_colors = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 192, 203), (128, 255, 0), (255, 165, 0),
]

print(f"\nTracking {num_cars} cars...")

frame_num = 0
frames_per_sample = int(fps * SAMPLE_INTERVAL)
total_frames = int(fps * TRACKING_DURATION)

print("Processing...")
last_progress = -1
active_trackers = list(range(num_cars))

while frame_num < total_frames:
    ret, frame = video.read()
    if not ret:
        print("Video ended")
        break
    
    progress = int((frame_num / total_frames) * 100)
    if progress != last_progress and progress % 10 == 0:
        print(f"{progress}% ({len(active_trackers)} active)")
        last_progress = progress
    
    should_sample = (frame_num % frames_per_sample == 0)
    new_active = []
    for idx in active_trackers:
        center = update_tracker(trackers[idx], frame)
        if center is not None:
            new_active.append(idx)
            if should_sample:
                car_tracks[idx].append(center)
    
    active_trackers = new_active
    
    if not active_trackers:
        print("Trackers stopped")
        break

    frame_num += 1

total_samples = sum(len(track) for track in car_tracks)
print(f"\nGot {total_samples} samples")

print("\nFitting lines...")
lane_lines = []

for i in range(len(car_tracks)):
    positions = car_tracks[i]
    if len(positions) < 2:
        lane_lines.append((None, None))
        continue
    
    line_start, line_end = fit_line(positions)
    lane_lines.append((line_start, line_end))

print("\nClustering...")
clusters = cluster(lane_lines, distance_threshold=30)

lane_lines_with_count = []
for cluster in clusters:
    lines_in_cluster = [lane_lines[idx] for idx in cluster]
    avg_line = avg_lines(lines_in_cluster)
    if avg_line[0] is not None:
        lane_lines_with_count.append((avg_line, len(cluster)))

NUM_LANES = 4
if len(lane_lines_with_count) > NUM_LANES:
    final_lane_lines = select_lines(lane_lines_with_count, n=NUM_LANES)
else:
    final_lane_lines = [line for line, _ in lane_lines_with_count]

print("\nShowing results...")
print("Press q to quit")

video.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    for i in range(len(car_tracks)):
        color = car_colors[i % len(car_colors)]
        track = car_tracks[i]
        for pos in track:
            cv2.circle(frame, pos, 5, color, -1)
            cv2.circle(frame, pos, 6, (255, 255, 255), 1)
    
    for line in final_lane_lines:
        if line[0] is not None and line[1] is not None:
            cv2.line(frame, line[0], line[1], (0, 255, 255), 4)
    
    text = f"Lane Lines: {len(final_lane_lines)} lanes (from {num_cars} cars)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Automated Lane Detection", frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("\nDone")
