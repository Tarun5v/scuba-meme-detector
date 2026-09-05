"""Run the scuba meme detector live from your webcam.

Track your body with MediaPipe pose landmarks in real time. When the app sees
the Scuba Dance move -- both arms raised up around the face with a fingertip
reaching the nose -- it plays a local meme video, then returns to the live feed.

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
    """Locate the meme clip, preferring the bundled reaction clip."""
    for name in ("nick_wilde_scuba.mp4", "scuba_meme.mp4"):
        path = os.path.join(ASSETS, name)
        if os.path.exists(path):
            return path
    return os.path.join(ASSETS, "scuba_meme.mp4")


def normalize_frame(frame):
    """Mirror the feed for a selfie view and convert BGR -> RGB."""
    frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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
            # Keep the clean live camera feed running in the background while
            # the meme plays (no overlays, same as the normal view).
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
    # macOS/OpenCV: the window close only takes effect after we pump the GUI
    # event loop, and the last video frame can linger as a ghost where the
    # video window was. Re-render a fresh camera frame over that region and
    # pump the event loop a few times so macOS fully clears the video frame.
    if ok_cam:
        cv2.imshow("SCUBA", display)
    for _ in range(5):
        cv2.waitKey(1)
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
    cv2.setWindowTitle("SCUBA", "Camera (q to quit)")
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

        # Show the clean camera feed (no overlays) so it just looks like a
        # normal webcam view while detection runs silently in the background.
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
