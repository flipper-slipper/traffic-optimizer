import cv2
from ultralytics import YOLO

# x (Center), y (Center), w, h
def create_tracker(frame, coords):
    top_left_coords = [coords[0] - (coords[2] / 2), coords[1] - (coords[3] / 2), coords[2], coords[3]]
    tracker = cv2.TrackerMIL.create()
    tracker.init(frame, list(map(int, top_left_coords)))

model = YOLO('yolo11n.pt')
video = cv2.VideoCapture('dataset/example_recording.webm')

trackers = []
ret, frame = video.read()

bboxes = list(model.predict(frame))[0].boxes.xywh

for idx, bbox in enumerate(bboxes):
    coords = bbox.tolist()
    
    tracker = create_tracker(frame, coords)
    trackers.append(tracker)
 
YOLO_RERUN_FRAMES = 5

frame = 0
while True:
    ret, frame = video.read()
    
    new_bboxes = list(model.predict(frame))[0].boxes.xywh
    # print(len(new_bboxes))
    tracker_bboxes = []
    unique_bboxes = []
    
    for tracker in trackers:
        ret, bbox = tracker.update(frame)
        tracker_bboxes.append(bbox)
        
    for new_bbox in new_bboxes:
        xn_center, yn_center, wn, hn = new_bbox.tolist()
        for tracker_bbox in tracker_bboxes:
            xt_center, yt_center, wt, ht = tracker_bbox
            
            if(abs(xt_center - xn_center) > 10 and abs(yt_center - yn_center) > 10):
                tracker = cv2.TrackerMIL.create()
                tracker.init(frame, tracker_bbox)
                trackers.append(tracker)
                
    for tracker in trackers:
        ret, bbox = tracker.update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            
    cv2.imshow('Tracking', frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
