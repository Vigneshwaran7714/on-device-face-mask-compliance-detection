"""
Real-time, fully on-device mask compliance detection (webcam).
3 states per face:  MASK (green) | IMPROPER (orange) | NO MASK (red)
Keys:  q = quit    s = save a screenshot (handy for your PPT/report)
"""
import time
import cv2
import numpy as np
import tensorflow as tf

names = open("labels.txt").read().split()
STYLE = {   # label -> (text, BGR colour, counts as violation?)
    "with_mask":             ("MASK",     (0, 200, 0),   False),
    "mask_weared_incorrect": ("IMPROPER", (0, 165, 255), True),
    "without_mask":          ("NO MASK",  (0, 0, 255),   True),
}

interp = tf.lite.Interpreter(model_path="mask_model.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def classify(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32)
    interp.set_tensor(inp["index"], rgb[None, ...])     # scaling is inside the model
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0]
    k = int(np.argmax(probs))
    return names[k], float(probs[k])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Close other apps using it, or try VideoCapture(1).")
prev, shots = time.time(), 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    ok_n = bad_n = 0
    for (x, y, w, h) in faces:
        p = int(0.1 * w)                                 # small context padding
        x1, y1 = max(0, x - p), max(0, y - p)
        x2, y2 = min(frame.shape[1], x + w + p), min(frame.shape[0], y + h + p)
        label, conf = classify(frame[y1:y2, x1:x2])
        text, color, violation = STYLE[label]
        bad_n += violation; ok_n += (not violation)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{text} {conf*100:.0f}%", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    now = time.time(); fps = 1 / max(now - prev, 1e-6); prev = now
    cv2.putText(frame, f"FPS {fps:.0f} | Compliant {ok_n} | Violations {bad_n}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("On-Device Face Mask Compliance (q=quit, s=screenshot)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        shots += 1; cv2.imwrite(f"screenshot_{shots}.png", frame)

cap.release(); cv2.destroyAllWindows()
