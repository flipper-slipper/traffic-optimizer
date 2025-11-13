import cv2
from ultralytics import YOLO
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# x (Center), y (Center), w, h
def create_tracker(frame, coords):
    top_left_coords = [coords[0] - (coords[2] / 2), coords[1] - (coords[3] / 2), coords[2], coords[3]]
    tracker = cv2.TrackerMIL.create()
    tracker.init(frame, list(map(int, top_left_coords)))
    return tracker

def update_tracker(tracker, frame):
    ret, bbox = tracker.update(frame)
    if not ret:
        return None
    center = (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2)))
    return [-1, center]


def tracker_bbox_to_center_bbox(top_left_coords):
    center_coords = [top_left_coords[0] + (top_left_coords[2] / 2), top_left_coords[1] + (top_left_coords[3] / 2), top_left_coords[2], top_left_coords[3]]
    return center_coords

def calculate_lane_speeds(prev_car_centers, car_centers):
    prev_car_centers = np.array(prev_car_centers)
    car_centers = np.array(car_centers)
    
    delta = abs(car_centers - prev_car_centers)
    return delta

colors = [
    (0, 255, 0),   
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]

cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

model = YOLO('yolo11n.pt')
model.to('cuda')
print("Device in use:", model.device)
video = cv2.VideoCapture('dataset/example_recording.webm')
# video = cv2.VideoCapture('dataset/Rush_Hour_Traffic_Stop_and_Go_Video.mp4')

trackers = []

ret, frame = video.read()

def draw_lines_callback(event, x, y, flags, param):
    points = param['points']
    frame = param['frame']
    window_name = param['window_name']
    
    if(event == cv2.EVENT_LBUTTONDOWN):
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        print(f"Point {len(points)} added: {(x, y)}")
        
        if len(points) > 0 and len(points) % 2 == 0:
            # get the last two points
            p1 = points[-2] 
            p2 = points[-1] 
            
            # draw the actual line
            cv2.line(frame, p1, p2, (0, 0, 255), 2) 
            print(f"Line {len(points) // 2} drawn.")
            
        cv2.imshow(window_name, frame)    

lane_annotation_name = "Draw Lanes"
lane_annotation_frame = frame.copy()

callback_data = {
    'points': [],
    'frame': lane_annotation_frame,
    'window_name': lane_annotation_name
}

cv2.namedWindow(lane_annotation_name) 
cv2.setMouseCallback(lane_annotation_name, draw_lines_callback, callback_data) 

while True:
    cv2.imshow(lane_annotation_name, lane_annotation_frame)
    
    if(cv2.waitKey(1) == ord('q')):
        cv2.destroyWindow(lane_annotation_name)
        break

points = callback_data['points']
if(len(points) < 2):
    points = [(470, 195), (448, 51), (349, 198), (435, 45), (228, 200), (408, 60), (91, 203), (360, 65), (3, 196), (351, 49)]
lane_polygons = []

for i in range(0, len(points), 2):
    if i + 3 < len(points):
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]
        p4 = points[i + 3]
        
        lane_polygon = np.array([p1, p2, p4, p3], dtype=np.int32)
        lane_polygons.append(lane_polygon)
        
print(points)

bboxes = list(model.predict(frame, conf=0.1, max_det=100))[0].boxes.xywh

for idx, bbox in enumerate(bboxes):
    coords = bbox.tolist()
    
    tracker = create_tracker(frame, coords)
    trackers.append(tracker)
 
YOLO_RERUN_FRAMES = 5

frame_num = 0

last_car_centers = []
car_centers = []

lane_speeds = [[] for _ in range(len(lane_polygons))]

fps = video.get(cv2.CAP_PROP_FPS) or 30

while True:
    last_car_centers = car_centers.copy()
    car_centers = []
    ret, frame = video.read()
    if not ret:
        break
    
    for lane_polygon in lane_polygons:
        cv2.polylines(frame, [lane_polygon], True, (255, 0, 0), 2)
    
    if(frame_num % YOLO_RERUN_FRAMES == 0):
        trackers = []
        bboxes = list(model.predict(frame, conf=0.1, max_det=20, verbose=False, half=True, imgsz=640))[0].boxes.xywh

        for idx, bbox in enumerate(bboxes):
            coords = bbox.tolist()
            
            tracker = create_tracker(frame, coords)
            trackers.append(tracker)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda t: update_tracker(t, frame), trackers))
        car_centers = [r for r in results if r is not None]

    for l, center in enumerate(car_centers):
        for i, lane in enumerate(lane_polygons):
            in_lane = cv2.pointPolygonTest(lane, center[1], False)
            if(in_lane >= 0):
                car_centers[l][0] = i
                cv2.circle(frame, center[1], radius=2, color=colors[i], thickness=2)
                break
                
    if len(last_car_centers) > 0 and len(car_centers) > 0:
        used_prev = set()
        used_curr = set()
        for pi, (prev_lane, prev_center) in enumerate(last_car_centers):
            if prev_lane == -1:
                continue
            best_j = None
            best_dist = None
            for cj, (curr_lane, curr_center) in enumerate(car_centers):
                if cj in used_curr:
                    continue
                if curr_lane != prev_lane:
                    continue

                dist = abs(curr_center[0] - prev_center[0]) + abs(curr_center[1] - prev_center[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_j = cj

            if best_j is not None:
                used_prev.add(pi)
                used_curr.add(best_j)
                curr_center = car_centers[best_j][1]
                dy = abs(curr_center[1] - prev_center[1])
                speed = dy
                lane_speeds[prev_lane].append(speed)

    avg_lane_speeds = np.array([float(np.mean(s)) if len(s) > 0 else 0.0 for s in lane_speeds])
    print(np.round(avg_lane_speeds, 2))

    cv2.imshow("Multi Object Tracking", frame)
    
    frame_num += 1
    
    if(cv2.waitKey(10) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break