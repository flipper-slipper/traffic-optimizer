import cv2
import numpy as np

# storage for clicks
points = []  
# a copy for drawing
drawing_frame_copy = None 
drawing_window_name = 'Draw Lines (click in pairs). Press "q" to finish.'

# handles mouse events
def draw_lines_callback(event, x, y, flags, param):

    # use global variables
    global points, drawing_frame_copy 

    # check for left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # save the new point
        points.append((x, y))
        
        # draw a dot for feedback
        cv2.circle(drawing_frame_copy, (x, y), 5, (0, 255, 0), -1)
        print(f"Point {len(points)} added: {(x, y)}")

        # check if we have a pair of points
        if len(points) > 0 and len(points) % 2 == 0:
            # get the last two points
            p1 = points[-2] 
            p2 = points[-1] 
            
            # draw the actual line
            cv2.line(drawing_frame_copy, p1, p2, (0, 0, 255), 2) 
            print(f"Line {len(points) // 2} drawn.")
        
        # update the display
        cv2.imshow(drawing_window_name, drawing_frame_copy)

# set up the tracker
tracker = cv2.TrackerMIL.create() 
# open the video file
video = cv2.VideoCapture('dataset/example_recording.webm')

# read the first frame
ret, frame = video.read()
# exit if video fails
if not ret:
    print("Error: Could not read video file.")
    exit()

print("First, select the ROI for the *tracker* and press ENTER.")
# get the tracker's box
bbox = cv2.selectROI('Select ROI for Tracker', frame, True)
# close the roi window
cv2.destroyWindow('Select ROI for Tracker') 

# start the tracker
tracker.init(frame, bbox)

# make a copy for drawing lines
drawing_frame_copy = frame.copy() 
# create the drawing window
cv2.namedWindow(drawing_window_name) 
# connect the mouse function
cv2.setMouseCallback(drawing_window_name, draw_lines_callback) 

print("\nROI selected. Now, draw your lines in the new window.")
print("Click in pairs (point 1, point 2 = line 1; point 3, point 4 = line 2, etc.)")
print(f"Press 'q' or 'Esc' in the '{drawing_window_name}' window to finish and start tracking.")

# this is the line drawing loop
while True:
    # show the drawing frame
    cv2.imshow(drawing_window_name, drawing_frame_copy) 
    key = cv2.waitKey(1) & 0xFF
    # wait for 'q' or esc to quit
    if key == ord('q') or key == 27: 
        break

# close the drawing window
cv2.destroyWindow(drawing_window_name) 

print("\nLine drawing complete. Starting video playback with tracking.")

# main video processing loop
while True:
    # read a new frame
    ret, frame = video.read()
    # if video ends, stop
    if not ret:
        print("Video finished.")
        break  
    
    timer = cv2.getTickCount()
    # update the tracker position
    ret_tracker, bbox = tracker.update(frame)
    
    # if tracking is successful
    if(ret_tracker):
        # draw the tracker box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1) 
    
    # draw all the saved lines
    for i in range(0, len(points), 2):
        # make sure we have a full pair
        if i + 1 < len(points):
            # draw one line
            p1 = points[i]
            p2 = points[i+1]
            cv2.line(frame, p1, p2, (0, 0, 255), 2) 
    
    # show the final frame
    cv2.imshow('Tracking', frame)
    
    # check for quit key
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# clean up
video.release()
cv2.destroyAllWindows()