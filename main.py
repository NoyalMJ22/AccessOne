"""
main.py — EyeAssist live detector.
Gesture → Action mapping per document spec.
Red dot at pupil centre. Clean side panel.
"""

import cv2
import time
import numpy as np
import joblib
from features import get_features, FEATURE_NAMES
from pupil import draw_pupils

MODEL_PATH   = "model.pkl"
ENCODER_PATH = "label_encoder.pkl"
CAMERA_ID    = 1
CONFIDENCE   = 0.55
DEBOUNCE_S   = 1.5   # per spec cooldown

model = joblib.load(MODEL_PATH)
le    = joblib.load(ENCODER_PATH)

# Gesture → (display name, BGR colour, action description)
GESTURES = {
    "LONG_BLINK":       ("ENTRY",       ( 50,220, 50), "Beep + LED ON"),
    "LOOK_UP_BLINK":    ("LIGHT",       (200,200,  0), "Toggle Light"),
    "LOOK_RIGHT_BLINK": ("FAN",         (  0,200,230), "Toggle Fan"),
    "LOOK_DOWN_BLINK":  ("BED HEIGHT",  (200,130,  0), "Raise/Lower Bed"),
    "TRIPLE_BLINK":     ("EMERGENCY",   (  0, 50,255), "Alarm + SMS"),
    "LOOK_LEFT_LONG":   ("EXIT",        (180,  0,220), "Deactivate System"),
    "IDLE":             ("IDLE",        ( 45, 55, 65), ""),
}

_last_action = {}
_flash_label = None
_flash_until = 0.0

def debounced(label):
    global _flash_label, _flash_until
    now = time.time()
    if now - _last_action.get(label, 0) >= DEBOUNCE_S:
        _last_action[label] = now
        _flash_label = label
        _flash_until = now + 0.8
        g = GESTURES.get(label, ("?","",""))
        print(f"[GESTURE] {label}  →  {g[2]}")
        return True
    return False

F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX

def put(img,t,x,y,c,sc=0.42,th=1,font=F):
    cv2.putText(img,str(t),(x,y),font,sc,c,th,cv2.LINE_AA)

def hbar(img,x,y,w,h,val,mx,col):
    cv2.rectangle(img,(x,y),(x+w,y+h),(20,28,40),-1)
    fw=int(min(max(val,0)/max(mx,1e-9),1.0)*w)
    if fw>0: cv2.rectangle(img,(x,y),(x+fw,y+h),col,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(38,52,68),1)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

ret, sample = cap.read()
if not ret: raise RuntimeError("Cannot read camera")
CAM_H, CAM_W = sample.shape[:2]

FEED_W  = 700
FEED_H  = int(FEED_W * CAM_H / max(CAM_W,1))
PANEL_W = 310
WIN_W   = FEED_W + PANEL_W
WIN_H   = max(FEED_H, 700)

cv2.namedWindow("EyeAssist", cv2.WINDOW_NORMAL)
cv2.resizeWindow("EyeAssist", WIN_W, WIN_H)
print(f"EyeAssist  |  camera {CAMERA_ID}  |  ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret: continue

    feat_vals, debug = get_features(frame)

    try:
        proba    = model.predict_proba([feat_vals])[0]
        pred_idx = int(np.argmax(proba))
        conf     = float(proba[pred_idx])
        label    = le.inverse_transform([pred_idx])[0]
    except Exception:
        label, conf = "IDLE", 0.0

    if conf >= CONFIDENCE and label != "IDLE":
        debounced(label)

    # Convert IR frame
    if len(frame.shape) == 2:
        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        disp = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2BGR)
    else:
        disp = frame.copy()

    # Red pupil dots
    pd = debug.get("pupil", {})
    if debug["face_found"] and pd and not pd.get("error"):
        disp = draw_pupils(disp, pd)

    feed = cv2.resize(disp, (FEED_W, FEED_H))

    # Canvas
    canvas = np.full((WIN_H, WIN_W, 3), (9,13,20), dtype=np.uint8)
    yo = (WIN_H - FEED_H)//2
    canvas[yo:yo+FEED_H, 0:FEED_W] = feed
    cv2.rectangle(canvas,(0,yo),(FEED_W-1,yo+FEED_H-1),(35,50,65),1)
    cv2.line(canvas,(FEED_W,0),(FEED_W,WIN_H),(38,52,68),1)

    # ── PANEL ─────────────────────────────────────────────────────────────────
    px = FEED_W + 10
    pw = PANEL_W - 18
    y  = 22

    put(canvas,"EYE",px,y,(0,200,170),sc=0.55,th=1,font=FB)
    put(canvas,"ASSIST",px+42,y,(50,220,80),sc=0.55,th=1,font=FB)
    y+=13; put(canvas,"IR GESTURE SYSTEM",px,y,(50,65,82),sc=0.32)
    y+=12; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1); y+=10

    # Action badge
    now   = time.time()
    flash = _flash_label and now < _flash_until
    sl    = _flash_label if flash else label
    gtxt, gcol, gdesc = GESTURES.get(sl, ("IDLE",(45,55,65),""))
    active = conf >= CONFIDENCE and label != "IDLE"

    cv2.rectangle(canvas,(px,y),(px+pw,y+52),gcol,-1 if flash else 1)
    put(canvas,gtxt,px+8,y+34,(255,255,255) if (active or flash) else (70,85,100),
        sc=0.72,th=2,font=FB)
    put(canvas,gdesc,px+8,y+50,(200,220,200) if (active or flash) else (50,62,75),sc=0.32)
    y+=58; put(canvas,f"conf {conf:.0%}  |  {label}",px,y,gcol if active else (50,62,75),sc=0.32)
    y+=14; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1); y+=10

    # Detection
    put(canvas,"DETECTION",px,y,(90,125,155),sc=0.33); y+=14
    face_ok = debug["face_found"]
    fc=(50,220,80) if face_ok else (0,60,200)
    cv2.circle(canvas,(px+5,y-3),4,fc,-1)
    put(canvas,"FACE OK" if face_ok else "NO FACE",px+14,y,fc,sc=0.37)
    lconf=pd.get("left_confidence",0); rconf=pd.get("right_confidence",0)
    pc=(lconf+rconf)/2
    pc_c=(50,220,80) if pc>0.7 else (20,170,200) if pc>0.3 else (0,60,180)
    put(canvas,f"PUPIL  L:{lconf:.2f}  R:{rconf:.2f}",px+14,y+14,pc_c,sc=0.35)
    y+=30; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1); y+=10

    # EAR
    put(canvas,"EAR",px,y,(90,125,155),sc=0.33); y+=14
    el=feat_vals[FEATURE_NAMES.index("ear_left")]
    er=feat_vals[FEATURE_NAMES.index("ear_right")]
    ea=feat_vals[FEATURE_NAMES.index("ear_avg")]
    bc=(0,65,240) if ea<0.20 else (50,210,80)
    for lb2,vv in [("L",el),("R",er),("AVG",ea)]:
        put(canvas,f"{lb2} {vv:.3f}",px,y+10,(145,185,210),sc=0.35)
        hbar(canvas,px+58,y+1,pw-58,10,vv,0.4,bc); y+=15
    y+=6; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1); y+=10

    # Gaze map
    put(canvas,"GAZE",px,y,(90,125,155),sc=0.33); y+=10
    MAP=80; mx=px+pw//2-MAP//2
    cv2.rectangle(canvas,(mx,y),(mx+MAP,y+MAP),(18,26,38),-1)
    cv2.line(canvas,(mx,y+MAP//2),(mx+MAP,y+MAP//2),(35,50,65),1)
    cv2.line(canvas,(mx+MAP//2,y),(mx+MAP//2,y+MAP),(35,50,65),1)
    cv2.rectangle(canvas,(mx,y),(mx+MAP,y+MAP),(40,55,70),1)
    gx=pd.get("pupil_x",feat_vals[FEATURE_NAMES.index("iris_gaze_x")]) if pc>0.3 else feat_vals[FEATURE_NAMES.index("iris_gaze_x")]
    gy=pd.get("pupil_y",feat_vals[FEATURE_NAMES.index("iris_gaze_y")]) if pc>0.3 else feat_vals[FEATURE_NAMES.index("iris_gaze_y")]
    dx=int(mx+max(0.03,min(0.97,gx))*MAP); dy=int(y+max(0.03,min(0.97,gy))*MAP)
    cv2.circle(canvas,(dx,dy),5,(0,215,255),-1)
    cv2.circle(canvas,(dx,dy),9,(0,120,148),1)
    src="PUPIL" if pc>0.3 else "IRIS"
    put(canvas,f"{src} ({gx:.2f},{gy:.2f})",mx,y+MAP+12,(0,185,205),sc=0.30)
    y+=MAP+20; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1); y+=10

    # Head pose
    put(canvas,"HEAD POSE",px,y,(90,125,155),sc=0.33); y+=14
    for lb3,fn in [("Pitch","head_pitch"),("Yaw  ","head_yaw"),("Roll ","head_roll")]:
        v=feat_vals[FEATURE_NAMES.index(fn)]
        put(canvas,f"{lb3} {v:+.1f}",px,y+10,(150,178,208),sc=0.35)
        hbar(canvas,px+82,y+1,pw-82,9,abs(v),45,(18,150,200)); y+=14
    y+=6; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1); y+=10

    # Gaze dirs
    put(canvas,"GAZE DIRECTION",px,y,(90,125,155),sc=0.33); y+=14
    gd = debug.get("gaze_dirs",(False,False,False,False))
    for i,(name,active2) in enumerate([("UP",gd[0]),("DOWN",gd[1]),("LEFT",gd[2]),("RIGHT",gd[3])]):
        col2=(0,200,255) if active2 else (38,50,62)
        xi = px + (i%2)*(pw//2)
        yi = y + (i//2)*16
        cv2.circle(canvas,(xi+5,yi-3),4,col2,-1 if active2 else 1)
        put(canvas,name,xi+13,yi,col2,sc=0.36)
    y+=36; cv2.line(canvas,(px,y),(px+pw,y),(38,52,68),1)

    put(canvas,f"CAM {CAMERA_ID}   ESC = quit",px,WIN_H-8,(42,55,70),sc=0.32)

    cv2.imshow("EyeAssist",canvas)
    if cv2.waitKey(1)==27: break

cap.release()
cv2.destroyAllWindows()