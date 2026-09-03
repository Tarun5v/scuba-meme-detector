"""Heuristic pose scoring for the Scuba Dance.

This module turns raw MediaPipe pose landmarks into a single "scuba score".

What actually defines the move (verified against a reference clip of the trend):

  1. Both hands are raised up around the face/torso and wave side to side --
     this is the core, sustained signature of the dance.
  2. Briefly, one hand is brought right up to the nose (the nose-pinch) --
     a quick, diagnostic accent rather than a held pose.
  3. The knees open and close on the beat as a secondary confirmation.

Distances are measured relative to a per-frame torso scale so they behave the
same regardless of camera distance. The scale falls back to shoulder width so
detection still works when the lower body is cropped out of frame.
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
RAISED_MARGIN = 0.12       # wrist may sit this far below shoulder and still count
NOSE_PLUG_DISTANCE = 0.60  # wrist-to-nose distance (in scale units) = a pinch
WAVE_SPREAD = 0.10         # required side-to-side travel (normalized units)
WAVE_WINDOW = 6            # frames of history used to detect the wave


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
    pairs = ((ls, lh), (rs, rh))
    for a, b in pairs:
        if a.visibility >= 0.5 and b.visibility >= 0.5:
            return _distance((a.x, a.y), (b.x, b.y))
    return 0.0


class _HandTrack:
    """Rolling window of a single hand's horizontal position."""

    def __init__(self, window):
        self.buffer = deque(maxlen=window)

    def update(self, x):
        self.buffer.append(x)

    def clear(self):
        self.buffer.clear()

    @property
    def ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def waving(self, spread):
        """True when the buffered x-positions swing side to side."""
        vals = list(self.buffer)
        travel = max(vals) - min(vals)
        if travel < spread:
            return False
        # A real wave reverses direction at least once within the window.
        reversals = sum(
            1 for i in range(1, len(vals) - 1)
            if (vals[i] - vals[i - 1]) * (vals[i + 1] - vals[i]) < 0
        )
        return reversals >= 1


class ScubaDetector:
    def __init__(self, raised_margin=RAISED_MARGIN,
                 nose_plug_distance=NOSE_PLUG_DISTANCE,
                 wave_spread=WAVE_SPREAD,
                 wave_window=WAVE_WINDOW):
        self.raised_margin = raised_margin
        self.nose_plug_distance = nose_plug_distance
        self.wave_spread = wave_spread
        self._left = _HandTrack(wave_window)
        self._right = _HandTrack(wave_window)

    def reset(self):
        self._left.clear()
        self._right.clear()

    def score(self, landmarks, image_height):
        """Return 0.0-4.0 describing how strongly the scuba pose is present.

        A value >= 2.0 means the user is doing the move.
        """
        nose = landmarks[NOSE]
        if nose.visibility < 0.5:
            self._left.clear()
            self._right.clear()
            return 0.0

        scale = _frame_scale(landmarks)
        if scale <= 0.0:
            return 0.0

        nose_p = (nose.x, nose.y)
        lw = (landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y)
        rw = (landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y)
        shoulder_y = (landmarks[LEFT_SHOULDER].y
                      + landmarks[RIGHT_SHOULDER].y) / 2.0

        lw_vis = landmarks[LEFT_WRIST].visibility
        rw_vis = landmarks[RIGHT_WRIST].visibility

        score = 0.0

        # --- 1. Waving hands (the core signature). ---------------------------
        wave_count = 0
        # A hand counts as "raised" for waving when it clears shoulder level.
        if lw_vis >= 0.4 and lw[1] < shoulder_y + self.raised_margin:
            self._left.update(lw[0])
            if self._left.ready and self._left.waving(self.wave_spread):
                wave_count += 1
        else:
            self._left.clear()
        if rw_vis >= 0.4 and rw[1] < shoulder_y + self.raised_margin:
            self._right.update(rw[0])
            if self._right.ready and self._right.waving(self.wave_spread):
                wave_count += 1
        else:
            self._right.clear()

        # The wave is the defining motion; both hands just make it stronger.
        if wave_count >= 2:
            score += 1.5
        elif wave_count == 1:
            score += 1.0

        # --- 2. Nose pinch (a quick accent, adds confidence). ----------------
        best = min(_distance(nose_p, lw), _distance(nose_p, rw))
        if best / scale < self.nose_plug_distance:
            score += 1.0

        # --- 3. Knee open-close (secondary confirmation). --------------------
        lk = (landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y)
        rk = (landmarks[RIGHT_KNEE].x, landmarks[RIGHT_KNEE].y)
        if (landmarks[LEFT_KNEE].visibility >= 0.5
                and landmarks[RIGHT_KNEE].visibility >= 0.5):
            spread = abs(lk[0] - rk[0])
            drop = abs(lk[1] - rk[1])
            # Legs clearly apart and roughly level = a scuba stance.
            if spread > 0.30 and drop < 0.25:
                score += 1.0

        return min(score, 4.0)
