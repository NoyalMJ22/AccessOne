"""
pupil.py — IR pupil detection.
- Crops each eye using MediaPipe landmarks
- Adaptive threshold (auto-adjusts to your IR camera brightness)
- HoughCircles → contour centroid → darkest pixel fallback chain
- Returns absolute pixel coords of each pupil centre for red-dot overlay
"""

import cv2
import numpy as np
from collections import deque

# ── Config — tune PUPIL_PERCENTILE if red dot drifts off pupil ─────────────────
PUPIL_PERCENTILE  = 15    # darken threshold: bottom N% of pixels = pupil
MIN_PUPIL_RADIUS  = 3     # px (in upscaled ROI)
MAX_PUPIL_RADIUS  = 22    # px (in upscaled ROI)
EYE_PADDING       = 10    # extra pixels around eye landmarks when cropping
SMOOTH_N          = 4     # temporal smoothing frames (lower = more responsive)

_bufs: dict = {}

def _s(key, val, n=SMOOTH_N):
    if key not in _bufs:
        _bufs[key] = deque(maxlen=n)
    _bufs[key].append(val)
    return float(np.mean(_bufs[key]))

def _to_gray(frame):
    if len(frame.shape) == 2:            return frame.copy()
    if frame.shape[2] == 1:             return frame[:,:,0].copy()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def _crop_eye(gray, lm_inner, lm_outer, lm_top, lm_bot):
    fh, fw = gray.shape[:2]
    x1 = max(0,  int(min(lm_inner[0], lm_outer[0])) - EYE_PADDING)
    x2 = min(fw, int(max(lm_inner[0], lm_outer[0])) + EYE_PADDING)
    y1 = max(0,  int(min(lm_top[1],   lm_bot[1]))   - EYE_PADDING)
    y2 = min(fh, int(max(lm_top[1],   lm_bot[1]))   + EYE_PADDING)
    if x2 - x1 < 8 or y2 - y1 < 5:
        return None
    return gray[y1:y2, x1:x2], x1, y1, x2-x1, y2-y1

def _find_pupil(roi, is_ir):
    """
    Returns (cx_norm, cy_norm, radius_px, confidence) in upscaled space.
    confidence: 1.0=hough, 0.6=contour, 0.3=darkest pixel
    """
    h, w = roi.shape[:2]
    # Upscale small ROIs so circle detection works
    scale  = max(1, 80 // max(w, 1))
    up     = cv2.resize(roi, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
    uh, uw = up.shape[:2]

    # For RGB cameras invert so pupil is dark
    work = up if is_ir else cv2.bitwise_not(up)

    # Adaptive threshold based on actual pixel distribution
    thresh_val = int(np.percentile(work, PUPIL_PERCENTILE))
    thresh_val = max(5, min(thresh_val, 120))   # sanity clamp

    _, mask = cv2.threshold(work, thresh_val, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # ── HoughCircles ──────────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(work, (5,5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1, minDist=uw//2,
        param1=35, param2=12,
        minRadius=MIN_PUPIL_RADIUS * scale,
        maxRadius=MAX_PUPIL_RADIUS * scale,
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        return float(cx)/uw, float(cy)/uh, float(r)/scale, 1.0

    # ── Contour centroid ──────────────────────────────────────────────────────
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) > 15:
            M = cv2.moments(best)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                r  = np.sqrt(cv2.contourArea(best) / np.pi)
                return float(cx)/uw, float(cy)/uh, float(r)/scale, 0.6

    # ── Darkest pixel ─────────────────────────────────────────────────────────
    _, _, _, min_loc = cv2.minMaxLoc(work)
    # minMaxLoc signature: (minVal, maxVal, minLoc, maxLoc)
    mn, _, mn_loc, _ = cv2.minMaxLoc(work)
    return float(mn_loc[0])/uw, float(mn_loc[1])/uh, 3.0/scale, 0.3


def detect_pupils(frame, face_landmarks, frame_w, frame_h, is_ir=True):
    gray = _to_gray(frame)

    def lm(idx):
        p = face_landmarks[idx]
        return np.array([p.x * frame_w, p.y * frame_h])

    l_inner, l_outer = lm(33),  lm(133)
    l_top,   l_bot   = lm(159), lm(145)
    r_inner, r_outer = lm(362), lm(263)
    r_top,   r_bot   = lm(386), lm(374)

    # ── Left eye ──────────────────────────────────────────────────────────────
    lc = _crop_eye(gray, l_inner, l_outer, l_top, l_bot)
    if lc:
        roi_l, lx, ly, lw, lh = lc
        lpx, lpy, lpr, lconf = _find_pupil(roi_l, is_ir)
        l_abs = (int(lx + _s("lpx", lpx) * lw),
                 int(ly + _s("lpy", lpy) * lh))
        l_rect = (lx, ly, lw, lh)
    else:
        lconf, l_abs, l_rect = 0.0, (0,0), (0,0,0,0)
        lw = lh = 1

    # ── Right eye ─────────────────────────────────────────────────────────────
    rc = _crop_eye(gray, r_inner, r_outer, r_top, r_bot)
    if rc:
        roi_r, rx, ry, rw, rh = rc
        rpx, rpy, rpr, rconf = _find_pupil(roi_r, is_ir)
        r_abs = (int(rx + _s("rpx", rpx) * rw),
                 int(ry + _s("rpy", rpy) * rh))
        r_rect = (rx, ry, rw, rh)
    else:
        rconf, r_abs, r_rect = 0.0, (0,0), (0,0,0,0)
        rw = rh = 1

    # Normalised combined gaze (0-1 within frame)
    if lconf > 0 and rconf > 0:
        comb_x = (l_abs[0]/frame_w + r_abs[0]/frame_w) / 2
        comb_y = (l_abs[1]/frame_h + r_abs[1]/frame_h) / 2
    elif lconf > 0:
        comb_x, comb_y = l_abs[0]/frame_w, l_abs[1]/frame_h
    elif rconf > 0:
        comb_x, comb_y = r_abs[0]/frame_w, r_abs[1]/frame_h
    else:
        comb_x, comb_y = 0.5, 0.5

    px_vel = _s("pxv", comb_x - _bufs.get("_ppx", deque([comb_x]))[-1])
    py_vel = _s("pyv", comb_y - _bufs.get("_ppy", deque([comb_y]))[-1])
    _s("_ppx", comb_x); _s("_ppy", comb_y)

    return {
        "left_pupil_abs":   l_abs,
        "right_pupil_abs":  r_abs,
        "left_confidence":  lconf,
        "right_confidence": rconf,
        "left_roi_rect":    l_rect,
        "right_roi_rect":   r_rect,
        "pupil_x":          _s("px", comb_x),
        "pupil_y":          _s("py", comb_y),
        "pupil_x_vel":      px_vel,
        "pupil_y_vel":      py_vel,
        # per-eye normalised within eye ROI (kept for feature compatibility)
        "left_pupil_x":     _s("lpx_n", l_abs[0]/frame_w),
        "left_pupil_y":     _s("lpy_n", l_abs[1]/frame_h),
        "right_pupil_x":    _s("rpx_n", r_abs[0]/frame_w),
        "right_pupil_y":    _s("rpy_n", r_abs[1]/frame_h),
    }


def draw_pupils(frame, pupil_data):
    """
    Draw a red dot at each pupil centre. Nothing else.
    """
    out = frame.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    for key, conf_key in [("left_pupil_abs","left_confidence"),
                           ("right_pupil_abs","right_confidence")]:
        pt   = pupil_data.get(key, (0,0))
        conf = pupil_data.get(conf_key, 0)
        if conf > 0 and pt != (0,0):
            cv2.circle(out, pt, 4, (0, 0, 255), -1)   # solid red dot

    return out