import cv2, mediapipe as mp
mp_pose = mp.solutions.pose

def _pix_dist(p1, p2, w, h):
    dx, dy = (p1.x - p2.x) * w, (p1.y - p2.y) * h
    return (dx*dx + dy*dy) ** 0.5

def _cm_per_px(landmarks, w, h, height_cm: float):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    if heel.visibility < 0.5:
        heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    px_height = _pix_dist(nose, heel, w, h)
    if px_height <= 0:
        raise ValueError("Could not compute pixel height")
    return height_cm / px_height

def estimate_measurements_from_image(image_bgr, height_cm: float):
    pose = mp_pose.Pose(static_image_mode=True)
    res = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None, "No person detected."

    lm = res.pose_landmarks.landmark
    h, w, _ = image_bgr.shape
    cm_per_px = _cm_per_px(lm, w, h, height_cm)

    def horiz_cm(k1, k2, mult=1.0):
        return _pix_dist(lm[k1], lm[k2], w, h) * cm_per_px * mult

    # Simple circumference approximations from 2D spans
    bust_cm  = horiz_cm(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 1.30)
    hip_span = horiz_cm(mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.RIGHT_HIP,      1.25)
    waist_cm = hip_span * 0.85
    hips_cm  = hip_span
    inseam_cm = _pix_dist(lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_ANKLE], w, h) * cm_per_px

    return {
        "bust_cm": bust_cm,
        "waist_cm": waist_cm,
        "hips_cm": hips_cm,
        "inseam_cm": inseam_cm,
        "cm_per_px": cm_per_px,
        "landmarks": res.pose_landmarks,
    }, None