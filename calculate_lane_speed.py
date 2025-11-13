import cv2
from ultralytics import YOLO

# x (Center), y (Center), w, h
def create_tracker(frame, coords):
    top_left_coords = [coords[0] - (coords[2] / 2), coords[1] - (coords[3] / 2), coords[2], coords[3]]
    tracker = cv2.TrackerMIL.create()
    tracker.init(frame, list(map(int, top_left_coords)))
    return tracker

def tracker_bbox_to_center_bbox(top_left_coords):
    center_coords = [coords[0] + (coords[2] / 2), coords[1] + (coords[3] / 2), coords[2], coords[3]]
    return center_coords

model = YOLO('yolo11n.pt')
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
print(points)
bboxes = list(model.predict(frame))[0].boxes.xywh

for idx, bbox in enumerate(bboxes):
    coords = bbox.tolist()
    
    tracker = create_tracker(frame, coords)
    trackers.append(tracker)
 
YOLO_RERUN_FRAMES = 7

frame_num = 0
car_coords = []
last_car_coords = []

while True:
    last_car_coords = car_coords
    car_coords = []
    ret, frame = video.read()
    
    if(frame_num % YOLO_RERUN_FRAMES == 0):
        trackers = []
        bboxes = list(model.predict(frame))[0].boxes.xywh

        for idx, bbox in enumerate(bboxes):
            coords = bbox.tolist()
            
            tracker = create_tracker(frame, coords)
            trackers.append(tracker)
    
    for tracker in trackers:
        ret, bbox = tracker.update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        
        car_coords.append(tracker_bbox_to_center_bbox(bbox))
        
    
    cv2.imshow("Multi Object Tracking", frame)
    
    frame_num += 1
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break