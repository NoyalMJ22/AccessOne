"""
features.py — EyeAssist IR feature extraction.
Built exactly to the EyeAssist gesture spec:

  G-01  LONG_BLINK      EAR < 0.20 held 1.5–2.0 s         → Entry Trigger
  G-02  LOOK_UP_BLINK   Gaze up >20° + short blink <150ms  → Light Toggle
  G-03  LOOK_RIGHT_BLINK Gaze right >20° + short blink      → Fan Control
  G-04  LOOK_DOWN_BLINK Gaze down >20° + short blink        → Bed Height
  G-05  TRIPLE_BLINK    3 blinks within 2 s                 → Emergency
  G-06  LOOK_LEFT_LONG  Gaze left + EAR < 0.20 for 1.5 s   → Exit Mode
  IDLE                  No gesture active
"""

import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pupil import detect_pupils

# ── Camera mode ────────────────────────────────────────────────────────────────
IS_IR = True   # True = IR camera (default). Set False for RGB webcam.

# ── MediaPipe ─────────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# ── Landmark indices ───────────────────────────────────────────────────────────
L_VERT   = [159, 145];  L_HORIZ = [33,  133]
R_VERT   = [386, 374];  R_HORIZ = [362, 263]
L_IRIS   = 468;         R_IRIS  = 473
MOUTH_T  = 13;          MOUTH_B = 14
FACE_L   = 234;         FACE_R  = 454

# ── Spec constants (from document) ────────────────────────────────────────────
EAR_BLINK_TH       = 0.20   # G-01/02/03/04/06 blink threshold
LONG_BLINK_MIN_S   = 1.50   # G-01/G-06 minimum hold time (seconds)
LONG_BLINK_MAX_S   = 2.00   # G-01/G-06 maximum hold time
SHORT_BLINK_MAX_S  = 0.150  # G-02/03/04 confirm blink max duration (150 ms)
SHORT_BLINK_MIN_S  = 0.030  # minimum to not count as noise
TRIPLE_BLINK_WIN_S = 2.0    # G-05 window for 3 blinks
GAZE_ANGLE_TH      = 20.0   # degrees — min gaze shift to register direction
GAZE_CONFIRM_F     = 4      # frames gaze must be sustained before counting

# ── Smoothing ──────────────────────────────────────────────────────────────────
SMOOTH_N = 5
_bufs: dict = {}

def _s(key, val, n=SMOOTH_N):
    if key not in _bufs:
        _bufs[key] = deque(maxlen=n)
    _bufs[key].append(val)
    return float(np.mean(_bufs[key]))

# ── Helpers ────────────────────────────────────────────────────────────────────
def _lm(face, idx, w, h):
    p = face[idx]; return np.array([p.x*w, p.y*h])

def _dist(a, b): return float(np.linalg.norm(a-b))

def _ear(face, v, hz, w, h):
    top = _lm(face,v[0],w,h); bot = _lm(face,v[1],w,h)
    inn = _lm(face,hz[0],w,h); out = _lm(face,hz[1],w,h)
    return _dist(top,bot) / (_dist(inn,out)+1e-6)

# ── Temporal pattern state ─────────────────────────────────────────────────────
_eye_closed_since  = None   # when eye went below EAR_BLINK_TH
_blink_log         = []     # list of (open_time, duration) for completed blinks
_gaze_dir_count    = {"up":0,"down":0,"left":0,"right":0}  # sustained frame count

# Per-gesture cooldown so we don't repeat-fire
_gesture_cooldown  = {}
GESTURE_COOLDOWN_S = 1.5

def _on_cooldown(gid):
    return time.time() - _gesture_cooldown.get(gid, 0) < GESTURE_COOLDOWN_S

def _arm_cooldown(gid):
    _gesture_cooldown[gid] = time.time()

def _update_blink_and_gestures(ear_avg, gaze_up, gaze_down, gaze_left, gaze_right):
    """
    Core temporal engine. Returns dict of active gesture flags.
    Implements exact spec logic for G-01 through G-06.
    """
    global _eye_closed_since, _blink_log

    now = time.time()
    out = {f"G{i:02d}": False for i in range(1,7)}
    out["blink_open"] = False   # eye just opened (for debug)

    # ── Track eye closure ─────────────────────────────────────────────────────
    if ear_avg < EAR_BLINK_TH:
        if _eye_closed_since is None:
            _eye_closed_since = now
        closed_dur = now - _eye_closed_since

        # G-01: Long blink (1.5–2.0 s), no specific gaze required
        if LONG_BLINK_MIN_S <= closed_dur <= LONG_BLINK_MAX_S and not _on_cooldown("G01"):
            out["G01"] = True

        # G-06: Left gaze + long blink (1.5 s+)
        if gaze_left and closed_dur >= LONG_BLINK_MIN_S and not _on_cooldown("G06"):
            out["G06"] = True

    else:
        # Eye just opened — record the blink if eye was closed
        if _eye_closed_since is not None:
            dur = now - _eye_closed_since
            _eye_closed_since = None
            out["blink_open"] = True

            if SHORT_BLINK_MIN_S <= dur <= SHORT_BLINK_MAX_S:
                # Short blink — could be confirm for G-02/03/04
                _blink_log.append(("short", now, gaze_up, gaze_down, gaze_right, gaze_left))
                _arm_cooldown("short")
            elif dur > SHORT_BLINK_MAX_S:
                # Longer blink
                _blink_log.append(("long", now, gaze_up, gaze_down, gaze_right, gaze_left))

    # Clean old blink log entries
    _blink_log = [(t, ts, *r) for (t, ts, *r) in _blink_log if now - ts <= TRIPLE_BLINK_WIN_S]

    # ── G-05: Triple blink — 3 blinks (any type) within 2 s ──────────────────
    recent_blinks = [b for b in _blink_log if now - b[1] <= TRIPLE_BLINK_WIN_S]
    if len(recent_blinks) >= 3 and not _on_cooldown("G05"):
        out["G05"] = True
        _arm_cooldown("G05")
        _blink_log.clear()

    # ── G-02/03/04: Gaze direction + short blink ──────────────────────────────
    recent_short = [b for b in _blink_log if b[0]=="short" and now - b[1] <= 0.8]
    if recent_short:
        b = recent_short[-1]   # most recent short blink
        _, _, bup, bdown, bright, bleft = b
        if bup    and not _on_cooldown("G02"): out["G02"] = True; _arm_cooldown("G02")
        if bright  and not _on_cooldown("G03"): out["G03"] = True; _arm_cooldown("G03")
        if bdown  and not _on_cooldown("G04"): out["G04"] = True; _arm_cooldown("G04")
        if out["G02"] or out["G03"] or out["G04"]:
            _blink_log.clear()

    return out

# ── Gaze direction from head pose + iris ──────────────────────────────────────
def _gaze_dirs(pitch_deg, yaw_deg, iris_gaze_x, iris_gaze_y, pupil_x, pupil_y, pupil_conf):
    """
    Returns (up, down, left, right) booleans.
    Uses GAZE_ANGLE_TH = 20° from the spec.
    For direction: combines head yaw/pitch with iris position.
    """
    # Head-based direction (dominant for big movements)
    head_up    = pitch_deg < -GAZE_ANGLE_TH
    head_down  = pitch_deg >  GAZE_ANGLE_TH
    head_left  = yaw_deg   < -GAZE_ANGLE_TH
    head_right = yaw_deg   >  GAZE_ANGLE_TH

    # Iris/pupil-based direction (more sensitive for subtle gaze shifts)
    src_x = pupil_x if pupil_conf > 0.4 else iris_gaze_x
    src_y = pupil_y if pupil_conf > 0.4 else iris_gaze_y

    iris_left  = src_x < 0.38
    iris_right = src_x > 0.62
    iris_up    = src_y < 0.38
    iris_down  = src_y > 0.62

    # Combine: either head OR iris must trigger, head gives stronger signal
    return (
        head_up    or iris_up,
        head_down  or iris_down,
        head_left  or iris_left,
        head_right or iris_right,
    )

# ── Sustained gaze counter ────────────────────────────────────────────────────
def _sustained(key, flag):
    if flag:
        _gaze_dir_count[key] = min(_gaze_dir_count[key]+1, GAZE_CONFIRM_F+5)
    else:
        _gaze_dir_count[key] = max(_gaze_dir_count[key]-1, 0)
    return _gaze_dir_count[key] >= GAZE_CONFIRM_F

# ── Feature names ──────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # EAR
    "ear_left", "ear_right", "ear_avg", "ear_delta",
    # Iris gaze (MediaPipe)
    "iris_gaze_x", "iris_gaze_y",
    # Pupil gaze (OpenCV IR)
    "pupil_x", "pupil_y", "pupil_x_vel", "pupil_y_vel",
    "pupil_left_conf", "pupil_right_conf",
    # Head pose
    "head_pitch", "head_yaw", "head_roll", "head_pitch_vel",
    # Blink signals
    "blink_left", "blink_right",
    # Sustained gaze direction flags (0/1)
    "gaze_up", "gaze_down", "gaze_left", "gaze_right",
    # Gesture pattern flags (temporal — this is what the model trains on)
    "pat_long_blink",       # G-01 / G-06 component
    "pat_short_blink",      # G-02/03/04 confirm component
    "pat_triple_blink",     # G-05 component
    "pat_gaze_up",          # G-02 gaze component
    "pat_gaze_right",       # G-03 gaze component
    "pat_gaze_down",        # G-04 gaze component
    "pat_gaze_left",        # G-06 gaze component
    # Misc
    "mouth_open", "face_width", "dwell_time",
]

# ── State ──────────────────────────────────────────────────────────────────────
_prev_ear    = 0.0
_prev_gx     = 0.5
_prev_gy     = 0.5
_prev_pitch  = 0.0
_prev_time   = time.time()

def get_features(frame):
    global _prev_ear, _prev_gx, _prev_gy, _prev_pitch, _prev_time

    h, w   = frame.shape[:2]
    feat   = {k: 0.0 for k in FEATURE_NAMES}
    debug  = {
        "face_found": False, "landmarks": None,
        "pupil": {}, "gestures": {}, "gaze_dirs": (False,False,False,False)
    }

    # Convert for MediaPipe
    if len(frame.shape) == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:
        rgb = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_mesh.detect(mp_img)

    now = time.time()
    dt  = max(now - _prev_time, 1e-6)
    feat["dwell_time"] = _s("dw", dt)
    _prev_time = now

    if not results.face_landmarks:
        return [feat[k] for k in FEATURE_NAMES], debug

    face = results.face_landmarks[0]
    debug["face_found"] = True
    debug["landmarks"]  = face

    # ── EAR ──────────────────────────────────────────────────────────────────
    el = _ear(face, L_VERT, L_HORIZ, w, h)
    er = _ear(face, R_VERT, R_HORIZ, w, h)
    feat["ear_left"]  = _s("el", el)
    feat["ear_right"] = _s("er", er)
    feat["ear_avg"]   = (feat["ear_left"] + feat["ear_right"]) / 2
    feat["ear_delta"] = _s("ed", (feat["ear_avg"] - _prev_ear) / dt)
    _prev_ear = feat["ear_avg"]

    feat["blink_left"]  = _s("bl", 1.0 if el < EAR_BLINK_TH else 0.0)
    feat["blink_right"] = _s("br", 1.0 if er < EAR_BLINK_TH else 0.0)

    # ── Pupil detection (IR OpenCV) ───────────────────────────────────────────
    try:
        pd = detect_pupils(frame, face, w, h, is_ir=IS_IR)
        feat["pupil_x"]         = pd["pupil_x"]
        feat["pupil_y"]         = pd["pupil_y"]
        feat["pupil_x_vel"]     = pd["pupil_x_vel"]
        feat["pupil_y_vel"]     = pd["pupil_y_vel"]
        feat["pupil_left_conf"] = pd["left_confidence"]
        feat["pupil_right_conf"]= pd["right_confidence"]
        debug["pupil"] = pd
    except Exception as e:
        debug["pupil"] = {"error": str(e)}

    # ── Iris gaze (MediaPipe) ─────────────────────────────────────────────────
    try:
        li = _lm(face, L_IRIS, w, h); ri = _lm(face, R_IRIS, w, h)
        li_in = _lm(face, L_HORIZ[0], w, h); li_out = _lm(face, L_HORIZ[1], w, h)
        ri_in = _lm(face, R_HORIZ[0], w, h); ri_out = _lm(face, R_HORIZ[1], w, h)
        lt    = _lm(face, L_VERT[0],  w, h); lb     = _lm(face, L_VERT[1],  w, h)
        lew   = _dist(li_in, li_out)+1e-6;   rew = _dist(ri_in, ri_out)+1e-6
        gx    = _s("gx", ((li[0]-li_in[0])/lew + (ri[0]-ri_in[0])/rew)/2)
        gy    = _s("gy", (li[1]-lt[1]) / (_dist(lt,lb)+1e-6))
        feat["iris_gaze_x"] = gx
        feat["iris_gaze_y"] = gy
        _prev_gx, _prev_gy  = gx, gy
    except IndexError:
        gx, gy = _prev_gx, _prev_gy

    # ── Head pose ─────────────────────────────────────────────────────────────
    pitch_deg = yaw_deg = 0.0
    if results.facial_transformation_matrixes:
        mat = np.array(results.facial_transformation_matrixes[0].data).reshape(4,4)
        R   = mat[:3,:3]
        pitch_deg = float(np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+R[2,2]**2))))
        yaw_deg   = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
        roll_deg  = float(np.degrees(np.arctan2(R[2,1], R[2,2])))
        feat["head_pitch"]     = _s("hp", pitch_deg)
        feat["head_yaw"]       = _s("hy", yaw_deg)
        feat["head_roll"]      = _s("hr", roll_deg)
        feat["head_pitch_vel"] = _s("hpv", (pitch_deg - _prev_pitch)/dt)
        _prev_pitch = pitch_deg

    # ── Mouth ─────────────────────────────────────────────────────────────────
    mt = _lm(face, MOUTH_T, w, h); mb = _lm(face, MOUTH_B, w, h)
    fl = _lm(face, FACE_L,  w, h); fr = _lm(face, FACE_R,  w, h)
    fw2 = _dist(fl, fr)+1e-6
    feat["mouth_open"] = _s("mo", _dist(mt,mb)/fw2)
    feat["face_width"] = _s("fw", fw2)

    # ── Gaze direction (spec: 20° angle threshold) ────────────────────────────
    pc = (feat["pupil_left_conf"] + feat["pupil_right_conf"]) / 2
    raw_up, raw_down, raw_left, raw_right = _gaze_dirs(
        pitch_deg, yaw_deg, gx, gy,
        feat["pupil_x"], feat["pupil_y"], pc
    )
    sus_up    = _sustained("up",    raw_up)
    sus_down  = _sustained("down",  raw_down)
    sus_left  = _sustained("left",  raw_left)
    sus_right = _sustained("right", raw_right)

    feat["gaze_up"]    = 1.0 if sus_up    else 0.0
    feat["gaze_down"]  = 1.0 if sus_down  else 0.0
    feat["gaze_left"]  = 1.0 if sus_left  else 0.0
    feat["gaze_right"] = 1.0 if sus_right else 0.0
    debug["gaze_dirs"] = (sus_up, sus_down, sus_left, sus_right)

    # ── Temporal gesture engine ───────────────────────────────────────────────
    gestures = _update_blink_and_gestures(
        feat["ear_avg"], sus_up, sus_down, sus_left, sus_right
    )
    debug["gestures"] = gestures

    feat["pat_long_blink"]  = 1.0 if (gestures["G01"] or gestures["G06"]) else 0.0
    feat["pat_short_blink"] = 1.0 if gestures.get("blink_open") else 0.0
    feat["pat_triple_blink"]= 1.0 if gestures["G05"] else 0.0
    feat["pat_gaze_up"]     = feat["gaze_up"]
    feat["pat_gaze_right"]  = feat["gaze_right"]
    feat["pat_gaze_down"]   = feat["gaze_down"]
    feat["pat_gaze_left"]   = feat["gaze_left"]

    return [feat[k] for k in FEATURE_NAMES], debug