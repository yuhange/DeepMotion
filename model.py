import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Video and Image
video_path = "train_video.mp4"
image_path = "output_image.png"

video = cv2.VideoCapture(video_path)
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Define keypoint indices (example: nose, shoulders) - adjust based on your needs
keypoint_indices = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]

# Get image keypoints
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Extract image keypoint data
image_keypoints = []
if results.pose_landmarks:
    for idx in keypoint_indices:
        landmark = results.pose_landmarks.landmark[idx]
        image_keypoints.append((int(landmark.x * image_width), int(landmark.y * image_height)))
else:
    print("No image pose detections, exiting.")
    exit()

# Initialize the video keypoints list (will store keypoints for each frame)
video_keypoints = []

# Process Video
while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    # Pose Estimation on Video Frame
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Extract Video keypoint data
    frame_keypoints = []
    if results.pose_landmarks:
        for idx in keypoint_indices:
            landmark = results.pose_landmarks.landmark[idx]
            frame_keypoints.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
    else:
        print("Skipping frame due to no detection")
        video_keypoints.append(video_keypoints[-1]) #reuse keypoints

        continue


    video_keypoints.append(frame_keypoints)  # Store keypoints for this frame

# basic Affine Warp Example (for a single frame)
def affine_warp(img, image_points, video_points):
    M = cv2.getAffineTransform(np.float32(video_points), np.float32(image_points))
    dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return dst

# Create the animated frames (basic warp example)
animated_frames = []
for frame_keypoints in video_keypoints:
   animated_frame = affine_warp(image.copy(), image_keypoints, frame_keypoints)
   animated_frames.append(animated_frame)

# Write output video
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, 30.0, (image_width, image_height))
for frame in animated_frames:
    out.write(frame)

out.release()
video.release()
cv2.destroyAllWindows()