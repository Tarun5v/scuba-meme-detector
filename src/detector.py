"""Heuristic pose scoring for the Scuba Dance.

This module turns raw MediaPipe pose landmarks into a single "scuba score".
The choreography we key on is:

  1. One wrist held near the nose (the nose-plug / pinch move).
  2. The opposite wrist waving back and forth horizontally in front of the
     body (the underwater "fan" motion).
  3. A rhythmic knee open-close, which serves as a secondary confirmation.

All measurements are computed relative to a per-frame scale so they hold up at
any camera distance.
"""

from collections import deque

# MediaPipe Pose landmark indices we care about.
NOSE = 0
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_KNEE = 25
RIGHT_KNEE = 26

# Tunable thresholds.
NOSE_PLUG_DISTANCE = 0.18      # max wrist-to-nose distance (relative units)
ARM_RAISE_Y = 0.28             # free wrist must be at least this high in frame
WAVE_AMPLITUDE = 0.085         # free wrist horizontal travel to count as waving
KNEE_BOUNCE_AMPLITUDE = 0.02   # knee vertical travel to reinforce the pose
FRAMES_TO_WAVE = 4             # consecutive frames that must show waving


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _frame_scale(landmarks):
    """Return a size reference so thresholds work at any distance."""
    shoulder = landmarks[11]
    hip = landmarks[23]
    if shoulder.visibility < 0.5 or hip.visibility < 0.5:
        return 0.0
    return _distance((shoulder.x, shoulder.y), (hip.x, hip.y))


class ScubaDetector:
    def __init__(self, nose_plug_distance=NOSE_PLUG_DISTANCE,
                 arm_raise_y=ARM_RAISE_Y,
                 wave_amplitude=WAVE_AMPLITUDE,
                 knee_bounce_amplitude=KNEE_BOUNCE_AMPLITUDE):
        self.nose_plug_distance = nose_plug_distance
        self.arm_raise_y = arm_raise_y
        self.wave_amplitude = wave_amplitude
        self.knee_bounce_amplitude = knee_bounce_amplitude
        # Rolling buffer of free-wrist horizontal positions for waving detection.
        self._wave_history = deque(maxlen=FRAMES_TO_WAVE)

    def reset(self):
        self._wave_history.clear()

    def score(self, landmarks, image_height):
        """Return a float 0.0-3.0 describing how strongly the pose is present.

        0.0 means the pose is definitely not present; a value >= 2.0 means the
        full signature (nose plug + waving + knee bounce) is being held.
        """
        nose = landmarks[NOSE]
        if nose.visibility < 0.5:
            self._wave_history.clear()
            return 0.0

        scale = _frame_scale(landmarks)
        if scale <= 0.0:
            return 0.0

        points = {
            "nose": (nose.x, nose.y),
            "lw": (landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y),
            "rw": (landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y),
            "lk": (landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y),
            "rk": (landmarks[RIGHT_KNEE].x, landmarks[RIGHT_KNEE].y),
        }

        wrist_vis = landmarks[LEFT_WRIST].visibility, landmarks[RIGHT_WRIST].visibility

        score = 0.0

        # --- 1. Nose plug: nearest wrist must sit on the nose. ---------------
        plug_candidates = []
        for label, vis in (("lw", wrist_vis[0]), ("rw", wrist_vis[1])):
            if vis < 0.5:
                continue
            d = _distance(points["nose"], points[label]) / scale
            plug_candidates.append((d, label))
        if plug_candidates:
            best_dist, best_label = min(plug_candidates)
            if best_dist < self.nose_plug_distance:
                score += 1.0
                free_hand = "rw" if best_label == "lw" else "lw"
            else:
                free_hand = None
        else:
            free_hand = None

        # --- 2. Free hand waving back and forth horizontally. ---------------
        if free_hand:
            free_vis = wrist_vis[1] if free_hand == "rw" else wrist_vis[0]
            if free_vis >= 0.5 and points[free_hand][1] < self.arm_raise_y:
                self._wave_history.append(points[free_hand][0])
                if (len(self._wave_history) == FRAMES_TO_WAVE
                        and self._is_waving()):
                    score += 1.0
            else:
                self._wave_history.clear()
        else:
            self._wave_history.clear()

        # --- 3. Knee open-close (secondary confirmation). --------------------
        lk_vis = landmarks[LEFT_KNEE].visibility
        rk_vis = landmarks[RIGHT_KNEE].visibility
        if lk_vis >= 0.5 and rk_vis >= 0.5:
            knee_spread = abs(points["lk"][0] - points["rk"][0])
            knee_drop = max(points["lk"][1], points["rk"][1]) - min(
                points["lk"][1], points["rk"][1])
            # Opening the legs spreads them apart; both knees held lowish.
            if knee_spread > self.knee_bounce_amplitude * 6 and knee_drop < 0.3:
                score += 1.0

        return score

    def _is_waving(self):
        """True when the buffered wrist positions swing side to side."""
        if len(self._wave_history) < FRAMES_TO_WAVE:
            return False
        vals = list(self._wave_history)
        spread = max(vals) - min(vals)
        # A convincing wave should sweep back and forth, not just sit still.
        direction_changes = sum(
            1 for i in range(1, len(vals))
            if (vals[i] - vals[i - 1]) * (vals[i + 1] - vals[i]) < 0
        ) if len(vals) >= 3 else 0
        return spread > self.wave_amplitude and direction_changes >= 1
