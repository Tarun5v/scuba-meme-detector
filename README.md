# scooba-meme-detector

A local, open-source Python app that watches your webcam in real time, tracks
your body with computer-vision pose landmarks, and plays the viral **Scooba
Scooba / Scuba Dance** meme video the moment you strike the pose.

The Scuba Dance, for context, is the move where you pinch your nose with one
hand and wave the other hand back and forth in front of you while your knees
bounce — like you're swimming underwater. Hold that pose long enough and this
app fires the meme at you.

![pose](https://img.shields.io/badge/pose%20tracking-mediapipe-green)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-brightgreen)

---

## Features

- **Real-time body tracking** at 30+ FPS on a 720p webcam feed (selfie view).
- **Heuristic pose scoring** that recognizes the Scuba signature: a hand on the
  nose, the opposite hand waving side to side, and a knee bounce.
- **Debounced triggering** so a brief flinch doesn't fire the meme — you have to
  hold the pose.
- **Seamless meme playback** in its own window, then it drops you back into the
  live feed. Press `Q`/`ESC`/`Space` in the meme window to skip early.
- **No cloud calls** — everything runs locally on your machine.

---

## Requirements

- Python **3.10+**
- A working webcam
- [`pip`](https://pip.pypa.io/en/stable/) to install dependencies

---

## Setup

1. **Clone the repo:**

   ```bash
   git clone https://github.com/Tarun5v/scooba-meme-detector.git
   cd scooba-meme-detector
   ```

2. **(Recommended) Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs `opencv-python` (webcam + video playback), `mediapipe`
   (pose landmark tracking), and `numpy`.

4. **Add your meme video:**

   Drop a clip of the Scooba / Scuba meme into the `assets/` folder and name it
   exactly:

   ```
   assets/scuba_meme.mp4
   ```

   Any `.mp4` of the trend works. A short 3–10 second clip loops best, but
   longer clips also play fine. See `assets/README.md` for details.

   > Video files are deliberately excluded from version control (they're large
   > and typically copyrighted), so this step is always yours.

---

## How to run

```bash
python3 src/main.py
```

A window titled **SCUBA** opens showing your mirrored camera feed with
skeleton landmarks overlaid.

Then just do the move:

1. Pinch/cover your nose with one hand.
2. Wave the other hand back and forth in front of you.
3. Bounce your knees a little.

Hold it for about a quarter of a second and the meme video plays. When it
finishes (or you skip it), you're straight back in the live feed.

### Controls

| Key         | Action                            |
| ----------- | --------------------------------- |
| `Q` / `ESC` | Quit the app                      |
| `R`         | (reserved) reset the video loop   |
| `Space`     | Skip the meme video while playing |

---

## How the pose detection works

Under the hood this is straightforward landmark geometry — no cloud services.
MediaPipe Pose returns 33 body landmarks per frame. `src/detector.py` scores how
strongly the current frame matches the Scuba pose:

1. **Nose plug** — one wrist sitting close to the nose landmark (relative to the
   torso width, so it works at any distance).
2. **Free-hand wave** — the opposite wrist is held up and oscillates side to
   side across recent frames.
3. **Knee bounce** — the knees spread and dip rhythmically as a secondary check.

Each condition adds to a running score. `src/main.py` requires the score to stay
high for `HOLD_FRAMES` consecutive frames (debounce) before it triggers, then
enforces a short cooldown so it doesn't spam the meme.

All thresholds live at the top of each file and are easy to tune if the pose
isn't registering for you.

---

## Project layout

```
scooba-meme-detector/
├── assets/
│   ├── README.md           # where to drop scuba_meme.mp4
│   └── scuba_meme.mp4      # (you add this — git ignored)
├── src/
│   ├── detector.py         # scuba pose scoring heuristics
│   └── main.py             # webcam loop, debounce, meme playback
├── .gitignore
├── LICENSE                 # MIT
├── README.md
└── requirements.txt
```

---

## Troubleshooting

- **"Could not open webcam"** — make sure a camera is connected and no other app
  (Zoom, your browser) is hogging it. Try plugging it into a different port.
- **The pose never triggers** — make sure you're in frame with decent lighting,
  front and center. If you're far from the camera, step closer so your full
  torso is visible. Adjust the thresholds in `detector.py` if needed.
- **Window renders at low FPS** — close other GPU-heavy apps. The loop already
  targets 720p/30fps; model complexity is set to `1` for speed.

---

## License

This project is open source and available under the [MIT License](LICENSE).
