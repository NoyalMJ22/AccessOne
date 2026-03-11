# AccessOne 👁️ — Gaze-Controlled Smart Room for Paralysed Patients

> *"Your eyes are your hands."*
> A fully hands-free smart environment system — control your entire room using only your gaze.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square)
![GazeTracking](https://img.shields.io/badge/GazeTracking-antoinelame-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🎯 What is AccessOne?

**AccessOne** is an assistive technology system designed for patients with paralysis (ALS, locked-in syndrome, spinal cord injuries, etc.) who have **no voluntary movement except eye control**. Using a standard webcam and computer vision, AccessOne tracks a patient's gaze in real time and maps it to smart home controls — giving complete environmental autonomy with zero physical interaction.

**The patient controls everything. With only their eyes.**

---

## 🌟 Core Features

### 👁️ Gaze Zone Control
The screen is divided into a **3×3 grid of control zones**. The patient looks at a zone and holds their gaze for a configurable dwell time (default: 2 seconds) to activate it.

| Zone | Action |
|---|---|
| Top-Left | 💡 Toggle Room Light |
| Top-Center | 🌀 Fan Speed Up |
| Top-Right | 📺 Toggle TV |
| Mid-Left | 🛏️↑ Raise Bed Height |
| **Center** | **🆘 Emergency SOS** |
| Mid-Right | 🛏️↓ Lower Bed Height |
| Bottom-Left | 🪟 Toggle Curtains |
| Bottom-Center | 🌀 Fan Speed Down |
| Bottom-Right | 🎵 Toggle Music |

### 🛏️ Smart Room Devices Controlled
- **Room Light** — On/Off + brightness dimming
- **Ceiling Fan** — 4-speed control (Off / Low / Medium / High)
- **Bed Height** — Motorized adjustment (0–100%)
- **Air Conditioning** — On/Off + temperature setting (16–30°C)
- **TV** — On/Off
- **Curtains** — Open/Close
- **Music** — On/Off

### 🚨 Emergency SOS
- **Single gaze-hold on the center zone** triggers an immediate alert
- Sends notification to the nurse station / caregiver app
- Visual + audio alert in the room
- Auto-logs event with timestamp

### 🎯 Dwell Selection (No Blink Needed)
- Gaze-dwell based activation — no forced blinking required
- Configurable dwell time (1s – 5s) based on patient preference
- Visual progress bar shows dwell fill
- Cooldown prevents accidental re-triggers

### 📐 Calibration System
- 9-point calibration grid on startup
- Maps gaze coordinates to screen zones accurately
- Re-calibration available at any time

---

## 🧠 How It Works — Technical Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     AccessOne Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Webcam → Face Detection → Pupil Tracking → Gaze Vector     │
│       → Zone Mapping → Dwell Timer → Action Trigger         │
│       → Smart Room Command → Feedback to Patient            │
└─────────────────────────────────────────────────────────────┘
```

1. **Face Detection** — OpenCV + dlib detects the face and extracts 68 facial landmarks
2. **Pupil Extraction** — `pupil.py` isolates each eye region and finds the pupil center
3. **Gaze Ratio Calculation** — `features.py` calculates horizontal and vertical gaze ratios (0.0–1.0)
4. **Direction Classification** — Maps gaze ratio to one of 9 directional zones
5. **Dwell Timer** — Tracks how long gaze stays in a zone; fires action on threshold
6. **Device Control** — Triggers the corresponding smart room command
7. **Feedback** — On-screen confirmation + activity log entry

---

## 📁 Project Structure

```
AccessOne/
├── main.py                  # 🚀 Entry point — live gaze recognition & room control
├── features.py              # 👁️  Facial feature & gaze ratio extraction
├── pupil.py                 # 🔵 Pupil center detection & eye tracking
├── find_camera.py           # 📷 Auto-detects available camera devices
├── record_data.py           # 💾 Records user face data for enrollment
├── generate_dataset.py      # 🗂️  Builds training dataset from captured images
├── train_model.py           # 🤖 Trains identity classifier
├── label_encoder.pkl        # 🏷️  Saved user identity label encoder
├── requirements.txt         # 📦 Python dependencies
├── gaze_smart_room.html     # 🖥️  Demo UI (works in browser, no backend needed)
└── .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Webcam (any USB or built-in)
- Good lighting (important for pupil tracking accuracy)

### 1. Clone the repository
```bash
git clone https://github.com/NoyalMJ22/AccessOne.git
cd AccessOne
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
```
opencv-python
dlib
numpy
scikit-learn
imutils
```

> **dlib tip**: If dlib install fails, use `pip install dlib --no-build-isolation` or download a prebuilt wheel for your OS.

### 3. (Optional) GazeTracking Integration
AccessOne is designed to work with [Antoine Lamé's GazeTracking library](https://github.com/antoinelame/GazeTracking):
```bash
git clone https://github.com/antoinelame/GazeTracking.git
cp -r GazeTracking/gaze_tracking ./gaze_tracking
```

---

## 🚀 Running AccessOne

### Step 1 — Find your camera index
```bash
python find_camera.py
```

### Step 2 — (First time only) Enroll the patient
```bash
python record_data.py
```
Enter the patient's name when prompted. Captures ~200 face samples.

### Step 3 — Build dataset & train
```bash
python generate_dataset.py
python train_model.py
```

### Step 4 — Launch the smart room controller
```bash
python main.py
```

A window opens showing the live camera feed with gaze direction overlay, active zone highlight, dwell progress bar, and room status panel.

---

## 🖥️ Browser Demo (Hackathon Ready)

Open `gaze_smart_room.html` in any modern browser for a **full interactive demo** — no Python or backend required. Simulates gaze tracking so you can demonstrate all features live.

**Demo features:**
- Live simulated gaze dot moving across zones
- All 6 devices controllable by clicking or gaze-dwell
- Emergency SOS button with alert animation
- Bed/fan/light sliders
- Real-time activity log
- 9-point calibration flow

---

## 🏥 Patient Experience Flow

```
Patient wakes up
    → System auto-starts on boot
    → Caregiver assists with one-time 9-point calibration
    → Patient is in full control

Patient looks at LIGHT zone (top-left) for 2 seconds
    → 💡 Light turns ON — confirmed in activity log

Patient looks at FAN UP zone (top-center) for 2 seconds
    → 🌀 Fan starts at LOW — speed shown on screen

Patient feels unwell → looks at CENTER zone for 2 seconds
    → 🚨 Emergency SOS sent to nurse station immediately
```

The UI uses **high contrast, large zones, and zero cognitive load** — everything is visible at a glance, with no buttons, no speech, no touch required.

---

## 🔧 Configuration

Edit constants in `main.py`:

```python
DWELL_TIME = 2.0        # Seconds to hold gaze before activation
COOLDOWN = 1.5          # Seconds before re-activation allowed
CAMERA_INDEX = 0        # Change if using external webcam
SHOW_LANDMARKS = True   # Show face landmark overlay
EMERGENCY_ZONE = True   # Enable center-zone emergency trigger
```

---

## 🗺️ Roadmap

- [ ] Voice feedback for confirmed actions
- [ ] MQTT / Home Assistant integration for real smart devices
- [ ] Night mode with IR camera support
- [ ] Caregiver mobile alerts via SMS
- [ ] Raspberry Pi deployment for bedside unit
- [ ] Multi-patient profile support
- [ ] Blink-code secondary input (Morse-like)

---

## 🤝 Built With

- [GazeTracking by antoinelame](https://github.com/antoinelame/GazeTracking) — core gaze estimation
- [OpenCV](https://opencv.org/) — real-time video processing
- [dlib](http://dlib.net/) — 68-point facial landmark detection
- [scikit-learn](https://scikit-learn.org/) — identity classification

---

## 📄 License

MIT License — free to use, modify, and deploy for assistive technology purposes.

---

<p align="center">
  Built with ❤️ for patients who deserve independence — one gaze at a time.
</p>
