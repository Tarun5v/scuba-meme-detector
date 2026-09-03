"""Heuristic pose scoring for the Scuba Dance.

This module turns raw MediaPipe pose landmarks into a single "scuba score".

The move (verified against the real trend and its reference clips) has three
signature elements:

  1. The dancer holds their nose with ONE hand (a sustained pinch, with the
     nose landmark sitting right at that wrist).
  2. The FREE hand waves side to side out in front of the face.
  3. The knees open and close rhythmically to the beat (the "juke").

Each element contributes to the score, and the opposite-hand pair (nose pinch
on one hand while the other waves) gives a bonus because that combination is
the most distinctive framing. Distances are measured relative to a per-frame
torso scale so they work at any camera distance.
"""

from collections import deque

# MediaPipe Pose landmark indices we care about.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

# Tunable thresholds.
RAISED_MARGIN = 0.18       # wrist may sit this far below shoulder and still count
NOSE_PLUG_DISTANCE = 0.55  # wrist-to-nose distance (in scale units) = a pinch
WAVE_SPREAD = 0.10         # required side-to-side travel (normalized units)
WAVE_WINDOW = 6            # frames of history used to detect a wave
KNEE_SPREAD = 0.10         # knee-spread travel that counts as the "juke"
KNEE_WINDOW = 6            # frames used to detect the open/close juke


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _frame_scale(landmarks):
    """Per-frame size reference so thresholds survive any camera distance."""
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]
    shoulder_vis = min(ls.visibility, rs.visibility)

    # Shoulder width is the most reliable measure and survives cropping.
    if shoulder_vis >= 0.4:
        width = _distance((ls.x, ls.y), (rs.x, rs.y))
        if width > 0.01:
            return width

    # Fall back to a torso side length when only one side is trusted.
    for a, b in ((ls, lh), (rs, rh)):
        if a.visibility >= 0.5 and b.visibility >= 0.5:
            return _distance((a.x, a.y), (b.x, b.y))
    return 0.0


class _Oscillate:
    """Rolling window that detects a back-and-forth (waving) motion.

    Feed it a scalar signal on every frame; it reports True once the signal
    has travelled side to side by at least `spread` and reversed direction.
    """

    def __init__(self, window, spread):
        self.buffer = deque(maxlen=window)
        self.spread = spread

    def update(self, value):
        self.buffer.append(value)

    def clear(self):
        self.buffer.clear()

    @property
    def ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def oscillating(self):
        vals = list(self.buffer)
        if max(vals) - min(vals) < self.spread:
            return False
        # A real oscillation reverses direction at least once within the window.
        reversals = sum(
            1 for i in range(1, len(vals) - 1)
            if (vals[i] - vals[i - 1]) * (vals[i + 1] - vals[i]) < 0
        )
        return reversals >= 1


class ScubaDetector:
    def __init__(self, raised_margin=RAISED_MARGIN,
                 nose_plug_distance=NOSE_PLUG_DISTANCE,
                 wave_spread=WAVE_SPREAD, wave_window=WAVE_WINDOW,
                 knee_spread=KNEE_SPREAD, knee_window=KNEE_WINDOW):
        self.raised_margin = raised_margin
        self.nose_plug_distance = nose_plug_distance
        self.wave_spread = wave_spread
        self._left = _Oscillate(wave_window, wave_spread)
        self._right = _Oscillate(wave_window, wave_spread)
        self._knees = _Oscillate(knee_window, knee_spread)

    def reset(self):
        self._left.clear()
        self._right.clear()
        self._knees.clear()

    def score(self, landmarks, image_height):
        """Return 0.0-4.0 describing how strongly the scuba pose is present.

        A value >= SCUBA_THRESHOLD (in main.py) means the user is doing the move.
        """
        nose = landmarks[NOSE]
        if nose.visibility < 0.5:
            self._left.clear()
            self._right.clear()
            self._knees.clear()
            return 0.0

        scale = _frame_scale(landmarks)
        if scale <= 0.0:
            self._left.clear()
            self._right.clear()
            self._knees.clear()
            return 0.0

        nose_p = (nose.x, nose.y)
        lw_p = (landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y)
        rw_p = (landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y)
        lw_vis = landmarks[LEFT_WRIST].visibility
        rw_vis = landmarks[RIGHT_WRIST].visibility
        shoulder_y = (landmarks[LEFT_SHOULDER].y
                      + landmarks[RIGHT_SHOULDER].y) / 2.0

        score = 0.0

        # --- Track each hand: is it waving, and is it near the nose? ----------
        # A hand waving side to side should not also count as a nose pinch;
        # the pinch is the still hand held on the nose while the other waves.
        lw_nose = _distance(nose_p, lw_p) / scale
        rw_nose = _distance(nose_p, rw_p) / scale

        left_waving = False
        right_waving = False
        if lw_vis >= 0.4 and lw_p[1] < shoulder_y + self.raised_margin:
            self._left.update(lw_p[0])
            if self._left.oscillating():
                left_waving = True
        else:
            self._left.clear()
        if rw_vis >= 0.4 and rw_p[1] < shoulder_y + self.raised_margin:
            self._right.update(rw_p[0])
            if self._right.oscillating():
                right_waving = True
        else:
            self._right.clear()

        wave_count = int(left_waving) + int(right_waving)

        # --- 1. Nose pinch (held, distinct from the waving hand). ------------
        # A hand counts as pinching only if it is near the nose AND not waving.
        left_pinch = (not left_waving) and lw_nose < self.nose_plug_distance
        right_pinch = (not right_waving) and rw_nose < self.nose_plug_distance
        if left_pinch or right_pinch:
            score += 1.0

        # --- 2. Free-hand wave: one/both wrists oscillating side to side. ----
        if wave_count >= 1:
            score += 1.0
        if wave_count >= 2:
            score += 0.5

        # --- 3. Knee juke: knees rhythmically opening and closing. ------------
        lk_vis = landmarks[LEFT_KNEE].visibility
        rk_vis = landmarks[RIGHT_KNEE].visibility
        if lk_vis >= 0.4 and rk_vis >= 0.4:
            spread = abs(landmarks[LEFT_KNEE].x - landmarks[RIGHT_KNEE].x)
            self._knees.update(spread)
            if self._knees.oscillating():
                score += 0.5
        else:
            self._knees.clear()

        # --- Bonus: one hand on the nose while the other waves. ---------------
        if (left_pinch or right_pinch) and wave_count >= 1:
            score += 0.5

        return min(score, 4.0)
