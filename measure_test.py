import cv2
from measure_core import estimate_measurements_from_image

img = cv2.imread("person.jpg")
vals, err = estimate_measurements_from_image(img, height_cm=170.0)
if err:
    print(err)
else:
    print({k: round(vals[k],1) for k in ["bust_cm","waist_cm","hips_cm","inseam_cm"]})