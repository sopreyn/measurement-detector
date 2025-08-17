import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF INFO/WARN
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2, mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

img = cv2.imread("person.jpg")   # put a full-body test image in the project root
res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if res.pose_landmarks:
    mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Landmarks", img)
    cv2.waitKey(0); cv2.destroyAllWindows()
else:
    print("No person detected.")