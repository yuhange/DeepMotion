import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Video and Image
video_path = "train_video.mp4"
image_path = "output_image_v2.png"

video = cv2.VideoCapture(video_path)
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Define keypoint indices (Use ALL keypoints for Delaunay triangulation)
keypoint_indices = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_PINKY,
    mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_THUMB,
    mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]

# Get image keypoints
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Extract image keypoint data
image_keypoints = []
if results.pose_landmarks:
    for idx in keypoint_indices:
        landmark = results.pose_landmarks.landmark[idx]
        image_keypoints.append((landmark.x * image_width, landmark.y * image_height))
    image_keypoints = np.array(image_keypoints, dtype=np.float32)
else:
    print("No image pose detections, exiting.")
    exit()

# Create Delaunay triangulation (on the image)
rect = (0, 0, image_width, image_height)  # Define the rectangle for the triangulation
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(image_keypoints)
triangleList = subdiv.getTriangleList()
delaunay_triangles = []
for t in triangleList:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    index_pt1 = np.where((image_keypoints == pt1).all(axis=1))[0][0]
    index_pt2 = np.where((image_keypoints == pt2).all(axis=1))[0][0]
    index_pt3 = np.where((image_keypoints == pt3).all(axis=1))[0][0]

    delaunay_triangles.append([index_pt1, index_pt2, index_pt3]) #save the keypoint ID instead of pixel location

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
            frame_keypoints.append((landmark.x * frame.shape[1], landmark.y * frame.shape[0]))
        frame_keypoints = np.array(frame_keypoints, dtype=np.float32)
    else:
        print("Skipping frame due to no detection")
        if video_keypoints:
            video_keypoints.append(video_keypoints[-1])  # Reuse keypoints
        else:
            print("Error: No keypoints detected in the first frame.")
            continue  # Skip this frame
        continue

    video_keypoints.append(frame_keypoints)  # Store keypoints for this frame

import cv2
import numpy as np

def warp_triangle(img1, img2, t1, t2, feather_amount=5):  # Added feather_amount
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by the top-left corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Create feathering mask
    if feather_amount > 0:
        kernel = np.ones((feather_amount, feather_amount), np.float32) / (feather_amount * feather_amount)
        mask = cv2.filter2D(mask, -1, kernel)  # Apply blurring (feathering)

    # Apply warpImage to small rectangular patches
    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    img2_rect = cv2.warpAffine(img1_rect, M, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

    #Ensuring the sizes match and the assignment location is within bounds
    x_start = r2[0]
    x_end = r2[0] + r2[2]
    y_start = r2[1]
    y_end = r2[1] + r2[3]

    # Check boundaries
    if x_start < 0: x_start = 0
    if y_start < 0: y_start = 0
    if x_end > img2.shape[1]: x_end = img2.shape[1]
    if y_end > img2.shape[0]: y_end = img2.shape[0]

    #Make sure warped image does not over write source image
    w = x_end - x_start
    h = y_end - y_start

    # CHECK FOR ZERO WIDTH OR HEIGHT BEFORE RESIZING
    if w <= 0 or h <= 0:
        print("Skipping triangle warp due to zero width or height.")
        return  # Skip to the next triangle

    #The cropped region should be equal to the warped image
    cropped_img2 = img2[y_start:y_end, x_start:x_end]
    resized_mask = cv2.resize(mask, (w, h)) #resizing mask to correct region
    resized_img2_rect = cv2.resize(img2_rect, (w,h)) #resizing warpped image to correct region

    #Perform replacement
    cropped_img2 = cropped_img2 * ((1.0, 1.0, 1.0) - resized_mask) + resized_img2_rect * resized_mask
    img2[y_start:y_end, x_start:x_end] = cropped_img2

# Create the animated frames
animated_frames = []
for frame_keypoints in video_keypoints:
    # Create a copy of the original image for each frame
    warped_image = image.copy()

    # Warp each triangle
    for triangle in delaunay_triangles:
        # Get the keypoint indices for the triangle
        index_pt1, index_pt2, index_pt3 = triangle

        # Get the corresponding points in the image and video frame
        img_triangle = [image_keypoints[index_pt1], image_keypoints[index_pt2], image_keypoints[index_pt3]]
        video_triangle = [frame_keypoints[index_pt1], frame_keypoints[index_pt2], frame_keypoints[index_pt3]]

        # Warp the triangle in the image (with feathering)
        warp_triangle(image, warped_image, img_triangle, video_triangle, feather_amount=5)

    # Post-processing (Bilateral Filtering)
    warped_image = cv2.bilateralFilter(warped_image, d=9, sigmaColor=75, sigmaSpace=75)

    animated_frames.append(warped_image)

# Write output video
output_path = "output_triangulation.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, 30.0, (image_width, image_height))
for frame in animated_frames:
    out.write(frame)

out.release()
video.release()
cv2.destroyAllWindows()
