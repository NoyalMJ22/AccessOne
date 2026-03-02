"""
find_camera.py — Auto-detect your IR camera and update CAMERA_ID in all files.
Run this once after connecting your IR camera.
"""

import cv2
import re

print("Scanning cameras 0–9...\n")

found = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            shape = frame.shape
            # IR cameras are usually grayscale (2D) or have very low colour variance
            is_gray = len(shape) == 2 or shape[2] == 1
            if not is_gray and len(shape) == 3:
                # Check if it's a colour-wrapped grayscale (all channels equal)
                b, g, r = cv2.split(frame)
                diff = cv2.absdiff(b, g).mean() + cv2.absdiff(g, r).mean()
                is_gray = diff < 3.0
            found.append((i, shape, is_gray))
            kind = "IR/GRAYSCALE" if is_gray else "RGB colour"
            print(f"  Camera {i}: {shape}  → {kind}")
        cap.release()

print()
if not found:
    print("No cameras found. Check your connections.")
    exit(1)

# Pick IR camera automatically (grayscale), fallback to highest index
ir_cam = next((i for i, _, g in found if g), found[-1][0])
rgb_cam = next((i for i, _, g in found if not g), found[0][0])

print(f"  Detected IR camera  : {ir_cam}")
print(f"  Detected RGB camera : {rgb_cam}")

# Show a preview
print(f"\nShowing IR camera ({ir_cam}) preview for 3 seconds...")
cap = cv2.VideoCapture(ir_cam)
for _ in range(90):
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, f"IR Camera {ir_cam} - press any key to close",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("IR Camera Preview", frame)
    if cv2.waitKey(33) != -1:
        break
cap.release()
cv2.destroyAllWindows()

# Ask user to confirm
print(f"\nUpdate CAMERA_ID to {ir_cam} in record_data.py and main.py? (y/n): ", end="")
ans = input().strip().lower()

if ans == "y":
    for fname in ["record_data.py", "main.py"]:
        try:
            txt = open(fname).read()
            new = re.sub(r"CAMERA_ID\s*=\s*\d+", f"CAMERA_ID = {ir_cam}", txt)
            open(fname, "w").write(new)
            print(f"  Updated {fname}")
        except FileNotFoundError:
            print(f"  {fname} not found — update CAMERA_ID manually")
    print(f"\nDone! CAMERA_ID is now {ir_cam} in both files.")
else:
    print(f"\nNo changes made. Manually set CAMERA_ID = {ir_cam} in record_data.py and main.py.")
