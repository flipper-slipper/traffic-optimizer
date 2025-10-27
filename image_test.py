from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')
results = model.predict("dataset/example_recording.webm", stream=True)

for idx, result in enumerate(results):
    img = result.plot()
    
    cv2.imshow('test', img)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break