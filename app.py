import os, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2, numpy as np, streamlit as st, mediapipe as mp
from measure_core import estimate_measurements_from_image
from mediapipe import solutions as mp_solutions
mp_pose = mp.solutions.pose

st.title("AI Body Measurement Extractor")
st.caption("Upload a front-facing full-body photo, enter your height, and get approximate measurements.")
height_cm = st.number_input("Your height (cm)", min_value=130, max_value=220, value=170)
unit = st.selectbox("Units", ["cm", "inches"])
file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])

to_unit = (lambda x: x) if unit=="cm" else (lambda x: x/2.54)

if file:
    bytes_ = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(bytes_, cv2.IMREAD_COLOR)
    vals, err = estimate_measurements_from_image(img, float(height_cm))
    if err:
        st.error(err)
    else:
        draw = img.copy()
        mp_solutions.drawing_utils.draw_landmarks(
            draw, vals["landmarks"], mp_pose.POSE_CONNECTIONS,
            mp_solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        st.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), caption="Detected pose", use_container_width=True)

        bust, waist, hips, inseam = [to_unit(vals[k]) for k in ["bust_cm","waist_cm","hips_cm","inseam_cm"]]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Bust", f"{bust:.1f} {unit}")
            st.metric("Waist", f"{waist:.1f} {unit}")
        with c2:
            st.metric("Hips", f"{hips:.1f} {unit}")
            st.metric("Inseam", f"{inseam:.1f} {unit}")

        csv = ("measurement,value,unit\n"
               f"bust,{bust:.1f},{unit}\n"
               f"waist,{waist:.1f},{unit}\n"
               f"hips,{hips:.1f},{unit}\n"
               f"inseam,{inseam:.1f},{unit}\n")
        st.download_button("Download CSV", data=csv, file_name="measurements.csv", mime="text/csv")

with st.expander("Photo tips (for better accuracy)"):
    st.write("- Plain background, full body visible\n- Camera at mid-torso height, minimal tilt\n- Arms relaxed, feet hip-width apart")