"""Heuristic pose scoring for the Scuba Dance.

This module turns raw MediaPipe pose landmarks into a single "scuba score".

The signature (calibrated against a set of real webcam snapshots) is driven by
face + hand gestures, not the lower body:

  1. One hand's FINGERTIPS come up to the nose (the pinch / face gesture). We
     measure from the fingertip, not the wrist joint, because the fingertips
     are what actually reach the nose.
  2. A hand waves side to side out in front of the face.

Because the dance is repetitive, the detector accumulates evidence over a short
rolling window rather than demanding that both gestures fire on the exact same
frame. The score climbs when there has been recent nose-touch activity AND
recent hand-waving, which is what doing the dance actually looks like.

Distances are measured relative to a per-frame shoulder scale so they behave
the same regardless of camera distance.
"""

from collections import deque

# MediaPipe Pose landmark indices we care about.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_MIDDLE = 21
RIGHT_MIDDLE = 22

# Tunable thresholds. These are tuned to recognize the dance naturally -- a
# hand doesn't have to be held awkwardly above the shoulder to count as waving,
# and a short near-nose pass is enough -- while still telling a real scuba from
# just waving a hand around (that distinction lives in nose_plug_distance).
RAISED_MARGIN = 0.30     # wrist may sit this far below shoulder and still wave
NOSE_PLUG_DISTANCE = 0.40  # fingertip-to-nose (scale units) = the face gesture
WAVE_SPREAD = 0.10       # required side-to-side travel (normalized units)
WAVE_WINDOW = 6          # frames of history used to detect a wave
NOSE_WINDOW = 10         # frames over which nose-touch evidence is gathered
WAVE_RATIO = 0.18        # fraction of wave window with motion to count a wave
NOSE_RATIO = 0.12        # fraction of nose window with a pinch to count it


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _shoulder_scale(landmarks):
    """Per-frame shoulder width so thresholds survive any camera distance."""
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    if min(ls.visibility, rs.visibility) < 0.4:
        return 0.0
    width = _distance((ls.x, ls.y), (rs.x, rs.y))
    return width if width > 0.01 else 0.0


class _Oscillate:
    """Rolling window that detects a back-and-forth (waving) motion."""

    def __init__(self, window, spread):
        self.buffer = deque(maxlen=window)
        self.spread = spread

    def update(self, value):
        self.buffer.append(value)

    def clear(self):
        self.buffer.clear()

    def oscillating(self):
        vals = list(self.buffer)
        if len(vals) < self.buffer.maxlen:
            return False
        if max(vals) - min(vals) < self.spread:
            return False
        reversals = sum(
            1 for i in range(1, len(vals) - 1)
            if (vals[i] - vals[i - 1]) * (vals[i + 1] - vals[i]) < 0
        )
        return reversals >= 1


class ScubaDetector:
    def __init__(self, raised_margin=RAISED_MARGIN,
                 nose_plug_distance=NOSE_PLUG_DISTANCE,
                 wave_spread=WAVE_SPREAD, wave_window=WAVE_WINDOW,
                 nose_window=NOSE_WINDOW, wave_ratio=WAVE_RATIO,
                 nose_ratio=NOSE_RATIO):
        self.raised_margin = raised_margin
        self.nose_plug_distance = nose_plug_distance
        self.wave_ratio = wave_ratio
        self.nose_ratio = nose_ratio
        self._left = _Oscillate(wave_window, wave_spread)
        self._right = _Oscillate(wave_window, wave_spread)
        self._nose = deque(maxlen=nose_window)
        self._waved = deque(maxlen=wave_window)

    def reset(self):
        self._left.clear()
        self._right.clear()
        self._nose.clear()
        self._waved.clear()

    def score(self, landmarks, image_height):
        """Return 0.0-4.0 describing how strongly the scuba pose is present.

        A value >= SCUBA_THRESHOLD (in main.py) means the user is doing the move.
        """
        nose = landmarks[NOSE]
        if nose.visibility < 0.5:
            self.reset()
            return 0.0

        scale = _shoulder_scale(landmarks)
        if scale <= 0.0:
            self.reset()
            return 0.0

        nose_p = (nose.x, nose.y)

        # ---- 1. Nose-touch: a fingertip reaching the nose. ------------------
        tips = (
            landmarks[LEFT_INDEX], landmarks[RIGHT_INDEX],
            landmarks[LEFT_MIDDLE], landmarks[RIGHT_MIDDLE],
        )
        distances = [
            _distance(nose_p, (t.x, t.y)) for t in tips
            if t.visibility >= 0.4
        ]
        nose_touch = (
            min(distances) / scale < self.nose_plug_distance
            if distances else False
        )
        self._nose.append(nose_touch)
        nose_ratio = sum(self._nose) / len(self._nose)

        # ---- 2. Free-hand wave: a wrist oscillating side to side. -----------
        shoulder_y = (landmarks[LEFT_SHOULDER].y
                      + landmarks[RIGHT_SHOULDER].y) / 2.0
        lw = landmarks[LEFT_WRIST]
        rw = landmarks[RIGHT_WRIST]
        if lw.visibility >= 0.4 and lw.y < shoulder_y + self.raised_margin:
            self._left.update(lw.x)
        else:
            self._left.clear()
        if rw.visibility >= 0.4 and rw.y < shoulder_y + self.raised_margin:
            self._right.update(rw.x)
        else:
            self._right.clear()

        waving = self._left.oscillating() or self._right.oscillating()
        self._waved.append(waving)
        wave_ratio = sum(self._waved) / len(self._waved)

        # ---- Combine recent evidence into a score. --------------------------
        score = 0.0
        if wave_ratio > self.wave_ratio:
            score += 1.0
            if nose_ratio > self.nose_ratio:
                score += 0.5       # both gestures close together = the move
        if nose_ratio > self.nose_ratio:
            score += 1.0

        return min(score, 4.0)
