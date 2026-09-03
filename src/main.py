"""Run the scuba meme detector live from your webcam.

Track your body with MediaPipe pose landmarks in real time. When the app sees
the Signature "Scooba Scooba" / Scuba Dance move -- one hand pinching the nose
while the other waves back and forth and the knees bounce -- it plays a local
meme video, then returns to the live feed.

Controls:
  Q / ESC   quit
  R         reset / re-sync the video loop position

Press any key inside the video window while it plays to skip it and go back
to the camera feed early.
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from detector import ScubaDetector

# --- Tuning -----------------------------------------------------------
CAP_WIDTH = 1280   # 720p
CAP_HEIGHT = 720
TARGET_FPS = 30
HOLD_FRAMES = 8        # how many frames the pose must hold before triggering
COOLDOWN_FRAMES = 60   # minimum gap between triggers (seconds-worth of frames)
SCUBA_THRESHOLD = 1.5  # detector score that counts as "doing the scuba"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")


def find_meme_video():
    """Locate the meme clip, preferring a real uploaded one over a placeholder."""
    for name in ("nick_wilde_scuba.mp4", "scuba_meme.mp4"):
        path = os.path.join(ASSETS, name)
        if os.path.exists(path):
            return path
    return os.path.join(ASSETS, "scuba_meme.mp4")


def normalize_frame(frame):
    """Mirror the feed for a selfie view and convert BGR -> RGB."""
    frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_hud(frame, hold, score):
    """Overlay status text on the camera feed."""
    label = "scuba" if hold >= HOLD_FRAMES else "tracking"
    color = (100, 220, 60) if hold >= HOLD_FRAMES else (230, 230, 230)
    cv2.putText(frame, f"pose: {label}  score: {score:.2f}",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                cv2.LINE_AA)
    cv2.putText(frame, "hold the scuba pose to trigger the meme",
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190),
                1, cv2.LINE_AA)


STOP_GRACE_FRAMES = 6  # keep playing this many frames after the pose ends


def play_meme(video_path, cam, pose, detector, threshold):
    """Play the meme while the user keeps doing the scuba pose.

    The webcam is scored on every frame as the clip plays. Playing continues
    while the pose stays above the threshold, and automatically stops once the
    user stops dancing (after a short grace period). Q / ESC / Space stops it
    immediately.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "error"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0
    frame_delay = max(1, min(int(round(1000.0 / fps)), 100))

    cv2.namedWindow("MEME", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowTitle("MEME", "")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    missing = 0
    result = "ended"
    while True:
        ok_cam, frame_cam = cam.read()
        if ok_cam:
            rgb = normalize_frame(frame_cam)
            rh = rgb.shape[0]
            res = pose.process(rgb)
            s = detector.score(
                res.pose_landmarks.landmark, rh) if res.pose_landmarks else 0.0
            # Keep the live camera feed running in the background while the
            # meme plays, so dancing is still visible.
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 128, 255), thickness=2))
            draw_hud(display, HOLD_FRAMES if s >= threshold else 0, s)
            cv2.imshow("SCUBA", display)
        else:
            s = 0.0

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.imshow("MEME", frame)
        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord("q"), 27, ord(" ")):
            result = "manual"
            break
        if s >= threshold:
            missing = 0
        else:
            missing += 1
            if missing >= STOP_GRACE_FRAMES:
                result = "ended"
                break

    cap.release()
    cv2.destroyWindow("MEME")
    return result


def main():
    meme_video = find_meme_video()
    if not os.path.exists(meme_video):
        print(f"[!] Could not find any meme video in {ASSETS}")
        print("    Add a clip named nick_wilde_scuba.mp4 or scuba_meme.mp4")
        print("    (see assets/README.md).")
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam (index 0). Is one attached?")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = ScubaDetector()
    hold = 0
    cooldown = 0

    cv2.namedWindow("SCUBA", cv2.WINDOW_NORMAL)
    print("[*] Running. Hold the scuba pose to play the meme. Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = normalize_frame(frame)
        rh, rw = rgb.shape[:2]
        results = pose.process(rgb)

        if results.pose_landmarks:
            score = detector.score(results.pose_landmarks.landmark, rh)
        else:
            score = 0.0

        # Debounce: only trigger after the pose is held steadily.
        if cooldown > 0:
            cooldown -= 1

        if score >= SCUBA_THRESHOLD:
            hold += 1
        else:
            hold = 0

        if hold >= HOLD_FRAMES and cooldown == 0:
            print("[*] Scuba pose detected! Playing meme (stops when you stop)...")
            end_reason = play_meme(
                meme_video, cap, pose, detector, SCUBA_THRESHOLD)
            print(f"[*] Meme stopped ({end_reason}).")
            detector.reset()
            hold = 0
            cooldown = COOLDOWN_FRAMES
            # Re-show the camera window after the meme closes.

        # Draw landmarks for a bit of visual feedback.
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 128, 255), thickness=2))

        draw_hud(display, hold, score)
        cv2.imshow("SCUBA", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n[*] Interrupted.")
    except Exception as exc:  # pragma: no cover
        print(f"[!] Unexpected error: {exc}")
        sys.exit(1)
