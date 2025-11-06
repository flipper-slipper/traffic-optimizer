import cv2

tracker = cv2.TrackerMIL.create()

video = cv2.VideoCapture('dataset/example_recording.webm')

ret, frame = video.read()

bbox = cv2.selectROI('Select ROI', frame, True)

tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    
    if(ret):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        
    cv2.imshow('Tracking', frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break