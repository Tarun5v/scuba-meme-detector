"""Heuristic pose scoring for the Scuba Dance.

This module turns raw MediaPipe pose landmarks into a single "scuba score".

The signature (calibrated against real webcam recordings of the dance) is a
face + hand gesture, not the lower body. In the actual dance the person has
BOTH arms raised up around the face, one hand's fingertips reaching the nose,
and some small natural motion. A casual one-handed face touch is NOT the move.

So the score requires:

  1. Both hands raised up (wrists above / near the shoulders) -- this is the
     "hands up around your face" posture that singles out the scuba from just
     touching your face with one hand.
  2. At least one hand's FINGERTIPS reaching the nose (the pinch / face
     gesture), held over a short window. We measure from the fingertip, not
     the wrist joint, because the fingertips are what actually reach the nose.
  3. A small motion bonus -- the free hand sways a little while doing it.

Because the pose is repetitive, evidence is accumulated over a short rolling
window instead of demanding every condition on the exact same frame.

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

# Tunable thresholds. The face gesture is the main thing: both hands raised
# with a fingertip reaching the nose, held over a short window. The wave rule
# is a loose motion check (small travel counts) used only as a score bonus.
RAISED_MARGIN = 0.30      # wrist may sit this far below shoulder and still be "up"
NOSE_PLUG_DISTANCE = 0.42 # fingertip-to-nose (scale units) = the face gesture
NOSE_WINDOW = 8           # frames over which nose-touch evidence is gathered
NOSE_RATIO = 0.40         # fraction of nose window with a pinch to count it
WAVE_SPREAD = 0.025       # small side-to-side travel counts as motion
WAVE_WINDOW = 6           # frames of history used to detect a small wave
WAVE_RATIO = 0.30         # fraction of wave window with motion to count it


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2) ** 0.5


def _shoulder_scale(landmarks):
    """Per-frame shoulder width so thresholds survive any camera distance.

    Uses the 3D landmark positions (x, y, z) so the measured width stays the
    same even when the person turns sideways or stands at an angle to the
    camera. A side-on pose used to shrink the 2D on-screen shoulder width and
    silently inflate every normalized distance.
    """
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    if min(ls.visibility, rs.visibility) < 0.4:
        return 0.0
    width = _distance((ls.x, ls.y, ls.z), (rs.x, rs.y, rs.z))
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

    def _score(self, nose_ratio, hand_up, waving):
        """Pure scoring given the current signals; separate for testability."""
        if not hand_up:
            return 0.0
        score = 0.0
        if nose_ratio > self.nose_ratio:
            score += 2.0
        if waving and nose_ratio > self.nose_ratio:
            score += 1.0
        return min(score, 4.0)

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

        nose_p = (nose.x, nose.y, nose.z)

        # ---- 1. Nose-touch: a fingertip reaching the nose. ------------------
        tips = (
            landmarks[LEFT_INDEX], landmarks[RIGHT_INDEX],
            landmarks[LEFT_MIDDLE], landmarks[RIGHT_MIDDLE],
        )
        distances = [
            _distance(nose_p, (t.x, t.y, t.z)) for t in tips
            if t.visibility >= 0.4
        ]
        nose_touch = (
            min(distances) / scale < self.nose_plug_distance
            if distances else False
        )
        self._nose.append(nose_touch)
        nose_ratio = sum(self._nose) / len(self._nose)

        # ---- 2. Both hands raised + small motion. ---------------------------
        shoulder_y = (landmarks[LEFT_SHOULDER].y
                      + landmarks[RIGHT_SHOULDER].y) / 2.0
        lw = landmarks[LEFT_WRIST]
        rw = landmarks[RIGHT_WRIST]

        left_up = lw.visibility >= 0.4 and lw.y < shoulder_y + self.raised_margin
        right_up = rw.visibility >= 0.4 and rw.y < shoulder_y + self.raised_margin
        both_up = left_up and right_up

        # Feed the wave detector the raised hand(s) so a small sway counts.
        if left_up:
            self._left.update(lw.x)
        else:
            self._left.clear()
        if right_up:
            self._right.update(rw.x)
        else:
            self._right.clear()

        waving = self._left.oscillating() or self._right.oscillating()
        self._waved.append(waving)
        wave_ratio = sum(self._waved) / len(self._waved)

        return self._score(nose_ratio, both_up, waving)
