"""
generate_dataset.py — EyeAssist Synthetic Dataset Generator
Calibrated from REAL IR samples provided by user.
Run: python generate_dataset.py
Then: python train_model.py
"""

import csv, os, random

random.seed(42)

def r(mu, s): return random.gauss(mu, s)
def rc(mu, s, lo=None, hi=None):
    v = random.gauss(mu, s)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v
def rb(p): return 1.0 if random.random() < p else 0.0

FEATURE_NAMES = [
    "ear_left","ear_right","ear_avg","ear_delta",
    "iris_gaze_x","iris_gaze_y","pupil_x","pupil_y","pupil_x_vel","pupil_y_vel",
    "pupil_left_conf","pupil_right_conf",
    "head_pitch","head_yaw","head_roll","head_pitch_vel",
    "blink_left","blink_right",
    "gaze_up","gaze_down","gaze_left","gaze_right",
    "pat_long_blink","pat_short_blink","pat_triple_blink",
    "pat_gaze_up","pat_gaze_right","pat_gaze_down","pat_gaze_left",
    "mouth_open","face_width","dwell_time",
]
N = len(FEATURE_NAMES)  # 32

def mk(el,er,ix,iy,px,py,pxv,pyv,lc,rc2,pitch,yaw,roll,pv,bl,br,gu,gd,gl,gr,plo,psh,ptr,pgu,pgr,pgd,pgl,mo,fw,dw):
    ea=(el+er)/2
    ed=r(-0.1,0.2)
    return [el,er,ea,ed,ix,iy,px,py,pxv,pyv,lc,rc2,pitch,yaw,roll,pv,bl,br,gu,gd,gl,gr,plo,psh,ptr,pgu,pgr,pgd,pgl,mo,fw,dw]

def LONG_BLINK():
    el=rc(0.07,0.03,0.01,0.14); er=rc(el,0.01,0.01,0.15)
    return mk(el,er, rc(0.49,0.03,0.42,0.56),rc(0.48,0.08,0.30,0.65),
              rc(0.49,0.03,0.42,0.56),rc(0.47,0.03,0.40,0.54),
              r(0,0.003),r(0,0.003),
              rc(0.6,0.15,0.3,0.85),rc(0.6,0.15,0.3,0.85),
              rc(3.0,2.5,-2,8),rc(-3.0,1.5,-6,0.5),rc(5.0,3.0,0,10),r(0,1),
              1.0,1.0, 0,0,0,0, 1.0,0,0, 0,0,0,0,
              rc(0.007,0.003,0.001,0.015),rc(325,8,305,345),rc(0.040,0.005,0.028,0.055))

def LOOK_UP_BLINK():
    el=rc(0.30,0.05,0.10,0.42); er=rc(el,0.02,0.08,0.44)
    iy=rc(0.32,0.09,0.13,0.46)
    gu=1.0 if iy<0.38 else rb(0.5)
    psh=rb(0.4)
    return mk(el,er, rc(0.48,0.06,0.35,0.52),iy,
              rc(0.47,0.02,0.43,0.52),rc(0.40,0.01,0.38,0.43),
              r(0.002,0.003),r(0.011,0.005),
              rc(0.8,0.2,0.6,1.0),rc(0.8,0.2,0.6,1.0),
              rc(5.0,0.3,4.5,5.8),rc(-5.0,0.6,-5.8,-4.3),rc(0.8,0.3,0,1.3),r(2.5,1.5),
              rc(0.2,0.1,0,0.4),rc(0.2,0.1,0,0.4), gu,0,0,0, 0,psh,0, gu,0,0,0,
              rc(0.012,0.005,0.001,0.025),rc(335,5,320,345),rc(0.042,0.005,0.028,0.055))

def LOOK_RIGHT_BLINK():
    el=rc(0.25,0.08,0.09,0.39); er=rc(el,0.03,0.07,0.42)
    ix=rc(0.36,0.06,0.24,0.48)
    gr=1.0 if ix<0.40 else rb(0.3)
    psh=rb(0.5)
    return mk(el,er, ix,rc(0.08,0.12,-0.32,0.22),
              rc(0.33,0.06,0.20,0.44),rc(0.45,0.02,0.41,0.50),
              r(-0.001,0.005),r(0.004,0.007),
              rc(0.75,0.2,0.3,1.0),rc(0.75,0.2,0.3,1.0),
              rc(2.0,8.0,-13,17),rc(-4.7,0.5,-5.6,-3.8),rc(6.8,2.0,4,10.5),r(-3,3),
              rc(0.4,0.3,0,1.0),rc(0.4,0.3,0,1.0), 0,0,0,gr, 0,psh,0, 0,gr,0,gr,
              rc(0.010,0.005,0.001,0.022),rc(322,10,305,340),rc(0.038,0.007,0.025,0.052))

def LOOK_DOWN_BLINK():
    el=rc(0.060,0.015,0.030,0.100); er=rc(el,0.010,0.025,0.110)
    return mk(el,er, rc(0.490,0.005,0.475,0.505),rc(-0.45,0.25,-0.82,-0.04),
              rc(0.400,0.007,0.385,0.415),rc(0.515,0.020,0.475,0.560),
              r(-0.001,0.002),r(0.002,0.003),
              rc(0.8,0.2,0.6,1.0),rc(0.8,0.2,0.6,1.0),
              rc(3.5,0.5,2.7,4.4),rc(-3.5,0.8,-4.5,-2.2),rc(8.0,1.5,6,10.5),r(-2.5,1.5),
              1.0,1.0, 0,0,0,0, 0,0,0, 0,0,0,0,
              rc(0.011,0.001,0.009,0.013),rc(327,4,316,336),rc(0.040,0.003,0.033,0.048))

def TRIPLE_BLINK():
    el=rc(0.350,0.025,0.295,0.415); er=rc(el,0.015,0.290,0.435)
    return mk(el,er, rc(0.505,0.030,0.445,0.565),rc(0.395,0.015,0.360,0.430),
              rc(0.410,0.010,0.390,0.435),rc(0.472,0.012,0.445,0.500),
              r(0,0.003),r(-0.001,0.002),
              1.0,1.0,
              rc(3.3,0.4,2.4,4.2),rc(-3.5,0.3,-4.2,-2.8),rc(10.7,0.8,9,12.2),r(0,1.5),
              0.0,0.0, 0,0,0,0, 0,0,0, 0,0,0,0,
              rc(0.021,0.004,0.013,0.030),rc(332,3,325,338),rc(0.040,0.004,0.033,0.052))

def LOOK_LEFT_LONG():
    el=rc(0.28,0.09,0.07,0.43); er=rc(el,0.03,0.05,0.46)
    gu=rb(0.3); gd=rb(0.5)
    return mk(el,er, rc(0.60,0.10,0.45,0.74),rc(0.43,0.04,0.34,0.50),
              rc(0.56,0.06,0.43,0.70),rc(0.43,0.06,0.32,0.55),
              r(0.004,0.006),r(0.003,0.006),
              rc(0.9,0.1,0.6,1.0),rc(0.9,0.1,0.6,1.0),
              rc(7.0,4.0,3,18),rc(-1.2,0.5,-2.0,-0.3),rc(7.8,1.5,5.5,10.5),r(3.5,4.5),
              rc(0.05,0.1,0,0.4),rc(0.05,0.1,0,0.4), gu,gd,0,1.0, rb(0.3),0,0, gu,1.0,gd,0,
              rc(0.012,0.007,0.002,0.030),rc(300,45,223,370),rc(0.042,0.006,0.030,0.055))

def IDLE():
    is_blink = random.random() < 0.10
    el = rc(0.10,0.05,0.02,0.18) if is_blink else rc(0.33,0.05,0.18,0.47)
    er = rc(el,0.02,0.02,0.49)
    ix=rc(0.49,0.10,0.27,0.73); iy=rc(0.43,0.08,0.11,0.49)
    px=rc(0.55,0.12,0.36,0.74); py=rc(0.47,0.08,0.33,0.64)
    gu=rb(0.15) if iy<0.35 else 0.0
    gd=rb(0.50) if py>0.55 else 0.0
    gr=rb(0.40) if px>0.58 else 0.0
    bl=1.0 if is_blink else 0.0
    psh=1.0 if is_blink else 0.0
    return mk(el,er, ix,iy,px,py,
              r(0.002,0.010),r(0.001,0.005),
              rc(0.88,0.15,0.6,1.0),rc(0.88,0.15,0.6,1.0),
              rc(7.0,5.0,-4,17),rc(-2.5,4.0,-7,11),rc(5.5,5.0,-2,13.5),r(1.5,8),
              bl,bl, gu,gd,0,gr, 0,psh,0, gu,gr,gd,0,
              rc(0.013,0.010,0.001,0.045),rc(310,40,223,380),rc(0.040,0.006,0.025,0.060))

GENERATORS = {
    "LONG_BLINK": LONG_BLINK,
    "LOOK_UP_BLINK": LOOK_UP_BLINK,
    "LOOK_RIGHT_BLINK": LOOK_RIGHT_BLINK,
    "LOOK_DOWN_BLINK": LOOK_DOWN_BLINK,
    "TRIPLE_BLINK": TRIPLE_BLINK,
    "LOOK_LEFT_LONG": LOOK_LEFT_LONG,
    "IDLE": IDLE,
}
SAMPLES_PER_CLASS = 300

print(f"\nGenerating {SAMPLES_PER_CLASS * len(GENERATORS)} synthetic samples...")
all_rows = []
for label, fn in GENERATORS.items():
    for _ in range(SAMPLES_PER_CLASS):
        row = fn()
        assert len(row)==N, f"{label}: got {len(row)}, expected {N}"
        all_rows.append(row + [label])
    print(f"  {label:<22} {SAMPLES_PER_CLASS} ✓")

random.shuffle(all_rows)
merged = list(all_rows)
real_count = 0
REAL_CSV = "dataset.csv"

if os.path.exists(REAL_CSV) and os.path.getsize(REAL_CSV) > 0:
    with open(REAL_CSV, newline="") as f:
        for line in csv.reader(f):
            if len(line) == N+1:
                try:
                    if sum(abs(float(x)) for x in line[:-1]) > 0.01:
                        merged.append(line); real_count += 1
                except: pass
    print(f"\n+ {real_count} real samples merged from existing dataset.csv")

random.shuffle(merged)
with open(REAL_CSV,"w",newline="") as f:
    w=csv.writer(f)
    for row in merged:
        w.writerow([f"{v:.8f}" if isinstance(v,float) else v for v in row])

print(f"\nSaved {len(merged)} rows → {REAL_CSV}")
print("Next: python train_model.py\n")