import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import metrics
from ultralytics import YOLO
from utils import find_ball
import csv

# File Paths
out_fp = './output.csv'
video_fp = './ball_tracking_video.mp4'
out_video_fp = './ball_tracking_output.mp4'

# Open video file
cap = cv2.VideoCapture(video_fp)

# Define the codec and create VideoWriter object
writer = cv2.VideoWriter(out_video_fp, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

# CSV File and Writer
csv_file = open(out_fp, 'w')
csv_writer = csv.writer(csv_file)

# Tracker -- CSRT -- to keep track of the ball
tracker = cv2.TrackerCSRT_create()
# YOLO model -- to redetect the ball if lost
model = YOLO("yolov8s.pt")

# Read the first frame
ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Define the bounding box
bbox = [885, 540, 100, 100]
(x, y, w, h) = [int(v) for v in bbox]
ball_ref = frame[y:y+h, x:x+w]
ball_ref_gray = cv2.cvtColor(ball_ref, cv2.COLOR_BGR2GRAY)

# Initialize tracker with first frame and bounding box
tracker.init(frame, tuple(bbox))

# Warming up the model -- first detection is slow
model(frame, verbose=False)

out_data = []

ball_lost = False
i = 0
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_show = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try to Track the ball    
    if not ball_lost:
        success, bbox_tracker = tracker.update(frame)
        if success:
            bbox = bbox_tracker
        if not success:
            print('Tracker failed')
            ball_lost = True

    # If ball is lost, try to detect it using YOLO
    if ball_lost:
        results = model(frame, verbose=False)
        bboxes = find_ball(results)
        
        if len(bboxes) > 0:
            x, y, x2, y2 = bboxes[0][0]

            padding = 3
            x, y, x2, y2 = x-padding, y-padding, x2+padding, y2+padding
            x, y, x2, y2 = max(0, x), max(0, y), min(frame.shape[1], x2), min(frame.shape[0], y2)
            w, h = x2-x, y2-y
            
            dist = np.sqrt((x - bbox[0])**2 + (y - bbox[1])**2)
            if dist > 50:
                bbox = [x, y, w, h]
                # Reinitialize the tracker                        
                tracker.init(frame, tuple(bbox))
                ball_lost = False
                
            # det_box = [x, y, x2, y2]
            # ball = frame[y:y2, x:x2]
            # ball_gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
    
    (x, y, w, h) = [int(v) for v in bbox]
    x, y = max(0, x), max(0, y)
    ball = frame[y:y+h, x:x+w]
    ball = cv2.resize(ball, ball_ref.shape[:2][::-1])
    ball_gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM and Difference Score
    ssim_score = metrics.structural_similarity(ball_ref_gray, ball_gray, gaussian_weights=True)
    diff = cv2.absdiff(ball_ref_gray, ball_gray)
    _, diff = cv2.threshold(diff, 110, 255, cv2.THRESH_BINARY)
    diff_score = np.sum(diff // 255) / (ball_ref_gray.shape[0] * ball_ref_gray.shape[1])
    
    # Check if ball is lost
    if (ssim_score < 0.11 or diff_score > 0.22) and ball_lost == False:
        ball_lost = True
        print('Ball is lost')

    # Write data if ball is not lost        
    if not ball_lost:
        cv2.rectangle(frame_show, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cx, cy = x+w//2, y+h//2
        frame_data = [i, cx, cy, w, h]
        out_data.append(frame_data)
        csv_writer.writerow(frame_data)
    
    writer.write(frame_show)
    cv2.imshow('Ball Tracking', frame_show)    
    
    k = cv2.waitKey(1)
    if  k == 27:
        break
    i += 1
    
cap.release()
csv_file.close()

