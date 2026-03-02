"""
record_data.py — EyeAssist gesture data recorder.

GESTURES TO RECORD (from spec doc):
  Key 1 → LONG_BLINK      Close eyes 1.5–2 s (G-01 Entry Trigger)
  Key 2 → LOOK_UP_BLINK   Look up 20°+ then quick blink (G-02 Light)
  Key 3 → LOOK_RIGHT_BLINK Look right 20°+ then quick blink (G-03 Fan)
  Key 4 → LOOK_DOWN_BLINK  Look down 20°+ then quick blink (G-04 Bed)
  Key 5 → TRIPLE_BLINK     Blink 3 times fast within 2 s (G-05 Emergency)
  Key 6 → LOOK_LEFT_LONG   Look left + hold blink 1.5 s (G-06 Exit)
  Key 7 → IDLE             Normal, no gesture (needed for baseline)

HOW TO RECORD (read this before starting):
  1. Sit in your normal position, IR camera in front of you
  2. Press the key JUST AS you complete the gesture — not before
  3. Record each gesture in blocks — do 10-15 reps, take a break, repeat
  4. TARGET: 100 samples per gesture (document spec shows 100/gesture sessions)
  5. Vary slightly — different blink speeds, head angles, distances
  6. IDLE: just sit naturally and press 7 every few seconds

ACCURACY TIPS:
  - LONG_BLINK: count "one-thousand-one, one-thousand-two" then open
  - LOOK_UP/RIGHT/DOWN: exaggerate the gaze shift, make it obvious
  - TRIPLE_BLINK: blink-blink-blink as fast as you can
  - LOOK_LEFT_LONG: hold the gaze LEFT while keeping eyes closed
"""

import cv2
import csv
import time
import numpy as np
from features import get_features, FEATURE_NAMES

CSV_PATH  = "dataset.csv"
CAMERA_ID = 1   # IR camera

LABELS = {
    ord("1"): "LONG_BLINK",
    ord("2"): "LOOK_UP_BLINK",
    ord("3"): "LOOK_RIGHT_BLINK",
    ord("4"): "LOOK_DOWN_BLINK",
    ord("5"): "TRIPLE_BLINK",
    ord("6"): "LOOK_LEFT_LONG",
    ord("7"): "IDLE",
}

TIPS = {
    "LONG_BLINK":       "Close both eyes and HOLD for ~1.5s",
    "LOOK_UP_BLINK":    "Look UP 20deg+, then blink quickly",
    "LOOK_RIGHT_BLINK": "Look RIGHT 20deg+, then blink quickly",
    "LOOK_DOWN_BLINK":  "Look DOWN 20deg+, then blink quickly",
    "TRIPLE_BLINK":     "Blink 3 times FAST (all within 2 seconds)",
    "LOOK_LEFT_LONG":   "Look LEFT and hold eyes CLOSED for 1.5s",
    "IDLE":             "Look natural, relaxed, no gesture",
}

TARGET = 100   # samples per gesture (per spec)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

file   = open(CSV_PATH, "a", newline="")
writer = csv.writer(file)
counts = {v: 0 for v in LABELS.values()}

FONT = cv2.FONT_HERSHEY_SIMPLEX

print("\n" + "="*55)
print("  EyeAssist Gesture Recorder")
print("="*55)
for k, label in LABELS.items():
    print(f"  [{chr(k)}]  {label:<22} — {TIPS[label]}")
print("  [ESC] Quit")
print("="*55)
print(f"\n  Target: {TARGET} samples per gesture\n")

flash_label = None
flash_until = 0.0

while True:
    ret, frame = cap.read()
    if not ret: continue

    feat_vals, debug = get_features(frame)
    key = cv2.waitKey(1)

    if key in LABELS:
        label = LABELS[key]
        writer.writerow(feat_vals + [label])
        file.flush()
        counts[label] += 1
        flash_label = label
        flash_until = time.time() + 0.5
        pct = int(counts[label]/TARGET*100)
        bar = "#"*int(pct/5) + "."*(20-int(pct/5))
        print(f"  {label:<22} {counts[label]:>3}/{TARGET}  [{bar}] {pct}%")

    # ── Build display ─────────────────────────────────────────────────────────
    h, w = frame.shape[:2]
    if len(frame.shape) == 2:
        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        disp = frame.copy()

    # Draw pupil dots
    pd = debug.get("pupil", {})
    if debug["face_found"] and pd and not pd.get("error"):
        from pupil import draw_pupils
        disp = draw_pupils(disp, pd)

    # Right info panel
    panel_w = 260
    panel   = np.full((h, panel_w, 3), (10,14,22), dtype=np.uint8)

    y = 22
    cv2.putText(panel, "EYEASSIST RECORDER", (8,y), FONT, 0.42, (0,200,170), 1)
    y += 20
    cv2.line(panel, (8,y), (panel_w-8,y), (35,50,65), 1)
    y += 14

    # Key guide
    for k, lbl in LABELS.items():
        cnt  = counts[lbl]
        pct  = min(cnt/TARGET, 1.0)
        done = cnt >= TARGET
        col  = (50,220,80) if done else (160,200,220)
        cv2.putText(panel, f"[{chr(k)}] {lbl}", (8,y), FONT, 0.34, col, 1)
        # mini progress bar
        bw = int(pct * (panel_w-18))
        cv2.rectangle(panel, (8,y+2), (panel_w-8,y+8), (22,30,42), -1)
        if bw > 0:
            cv2.rectangle(panel, (8,y+2), (8+bw,y+8), (50,220,80) if done else (0,160,200), -1)
        cv2.putText(panel, f"{cnt}", (panel_w-30,y), FONT, 0.33, col, 1)
        y += 20

    y += 4
    cv2.line(panel, (8,y), (panel_w-8,y), (35,50,65), 1)
    y += 12

    # Gaze direction indicators
    ud, dd, ld, rd = debug.get("gaze_dirs", (False,False,False,False))
    face_ok = debug["face_found"]
    fc = (50,220,80) if face_ok else (0,60,200)
    cv2.circle(panel, (14,y-3), 5, fc, -1)
    cv2.putText(panel, "FACE OK" if face_ok else "NO FACE", (24,y), FONT, 0.37, fc, 1)
    y += 18

    ear = feat_vals[FEATURE_NAMES.index("ear_avg")]
    bcol = (0,60,230) if ear < 0.20 else (50,210,80)
    cv2.putText(panel, f"EAR {ear:.3f}", (8,y), FONT, 0.38, bcol, 1)
    y += 18

    for lbl2, active in [("UP",ud),("DOWN",dd),("LEFT",ld),("RIGHT",rd)]:
        col2 = (0,200,255) if active else (45,58,72)
        cv2.circle(panel, (14,y-3), 4, col2, -1 if active else 1)
        cv2.putText(panel, lbl2, (24,y), FONT, 0.36, col2, 1)
        y += 14

    y += 4
    # Flash saved label
    now = time.time()
    if flash_label and now < flash_until:
        cv2.putText(panel, f"SAVED!", (8,y+14), FONT, 0.5, (0,255,180), 2)
        cv2.putText(panel, flash_label, (8,y+30), FONT, 0.38, (0,255,180), 1)
        cv2.putText(disp, f"SAVED: {flash_label}", (10, h-20),
                    FONT, 0.65, (0,255,180), 2)
        tip = TIPS.get(flash_label,"")
        cv2.putText(disp, tip, (10, 30), FONT, 0.48, (0,230,255), 1)

    cv2.line(panel, (8,h-22), (panel_w-8,h-22), (35,50,65), 1)
    cv2.putText(panel, "ESC = quit", (8,h-8), FONT, 0.33, (45,58,72), 1)

    display = np.hstack([disp, panel])
    cv2.imshow("EyeAssist Recorder — 1-7 to label, ESC to quit", display)

    if key == 27:
        break

cap.release()
file.close()
cv2.destroyAllWindows()

print("\n" + "="*55)
print("  Session complete. Sample counts:")
total = 0
for label, cnt in counts.items():
    bar  = "#"*int(cnt/TARGET*20) + "."*(20-int(cnt/TARGET*20))
    stat = "DONE" if cnt >= TARGET else f"{TARGET-cnt} more needed"
    print(f"  {label:<22} {cnt:>3}  [{bar}]  {stat}")
    total += cnt
print(f"\n  Total samples this session: {total}")
print(f"  Data saved to: {CSV_PATH}")
print("\n  Next: run  python train_model.py")
print("="*55)