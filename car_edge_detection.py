import cv2


def apply_canny_edge_detection(image):
    # convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # apply blur and canny edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def outline_car_with_canny(video_path=None, image_path=None):
    if image_path:
        frame = cv2.imread(image_path)
        is_video = False
    elif video_path:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        is_video = True
    else:
        cap = cv2.VideoCapture("dataset/example_recording.webm")
        ret, frame = cap.read()
        is_video = True

    # select ROI for the car
    bbox = cv2.selectROI("Select Car ROI", frame, False)
    cv2.destroyWindow("Select Car ROI")

    if is_video:
        # initialize tracker
        original_bbox = tuple(bbox)
        tracker = cv2.TrackerMIL.create()
        tracker.init(frame, bbox)

        window_name = "Car Edge Detection"
        cv2.namedWindow(window_name)

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                tracker = cv2.TrackerMIL.create()
                tracker.init(frame, original_bbox)
                bbox = original_bbox
                continue

            # update tracker position
            success, bbox = tracker.update(frame)

            if success:
                # get bbox coordinates and ensure they're within frame bounds
                x, y, w, h = [int(v) for v in bbox]
                x = max(0, min(x, frame.shape[1] - 1))
                y = max(0, min(y, frame.shape[0] - 1))
                w = max(1, min(w, frame.shape[1] - x))
                h = max(1, min(h, frame.shape[0] - y))

                # apply edge detection to tracked ROI
                roi = frame[y : y + h, x : x + w]
                edges = apply_canny_edge_detection(roi)

                # overlay edges on original frame
                output_frame = frame.copy()
                mask = edges > 0
                output_frame[y : y + h, x : x + w][mask] = [0, 255, 0]

                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow(window_name, output_frame)
                cv2.imshow("Edge Detection ROI", edges)

            # controls: 'q' to quit, 'r' to reselect ROI
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                new_bbox = cv2.selectROI("Select Car ROI", frame, False)
                cv2.destroyWindow("Select Car ROI")
                bbox = new_bbox
                original_bbox = tuple(new_bbox)
                tracker = cv2.TrackerMIL.create()
                tracker.init(frame, bbox)

        cap.release()
        cv2.destroyAllWindows()
    else:
        # process single image
        x, y, w, h = bbox
        roi = frame[y : y + h, x : x + w]
        edges = apply_canny_edge_detection(roi)

        # overlay edges on original image
        output_frame = frame.copy()
        mask = edges > 0
        output_frame[y : y + h, x : x + w][mask] = [0, 255, 0]

        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Original with Car Outline", output_frame)
        cv2.imshow("Edge Detection ROI", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    outline_car_with_canny()
