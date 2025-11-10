from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')
# results = list(model.predict("dataset/example_recording.webm", stream=True))
results = list(model.predict("dataset/example_image.png"))

first_result = results[0]
print(first_result.boxes.xywhn)



# for idx, result in enumerate(results):
#     img = result.plot()
    
#     print(result.boxes)
    
#     cv2.imshow('test', img)
    
#     if cv2.waitKey(20000) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
    
#     # break