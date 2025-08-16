import cv2 as cv
import mediapipe as mp
import numpy as np
from time import time

# Initialize pose and segmentation
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

video = cv.VideoCapture(0)
cv.namedWindow('Pose Detection', cv.WINDOW_NORMAL)
video.set(3, 1280)
video.set(4, 960)
time1 = 0

def detecPose(frame, pose_video):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose_video.process(frame_rgb)
    return results

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break
    frame = cv.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame = cv.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # Get pose and segmentation
    pose_results = detecPose(frame, pose_video)
    seg_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    seg_results = segmentation.process(seg_rgb)

    # Create a mask: human = 1.0, background = 0.0
    mask = seg_results.segmentation_mask > 0.1  # You can tune threshold

    # New image: original background, human blacked out
    frame_black_human = frame.copy()
    frame_black_human[mask] = (0, 0, 0)  # Set human pixels to black

    # Draw landmarks over black human
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame_black_human,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(225,117,66), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    time2 = time()
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv.putText(frame_black_human, 'FPS: {}'.format(int(frames_per_second)), (10, 30),
                   cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    time1 = time2

    cv.imshow('Pose Detection', frame_black_human)
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

video.release()
cv.destroyAllWindows()
