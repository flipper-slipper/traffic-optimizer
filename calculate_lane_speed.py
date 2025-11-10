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

multi_tracker = cv2.MultiTracker_create()
ret, frame = video.read()

bboxes = list(model.predict(frame))[0].boxes.xywh

for idx, bbox in enumerate(bboxes):
    coords = bbox.tolist()
    
    tracker = create_tracker(frame, coords)
    trackers.append(tracker)
 
YOLO_RERUN_FRAMES = 100

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