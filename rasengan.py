import argparse
import math
import time

import cv2
import mediapipe as mp
import numpy as np


PALM = (0, 5, 9, 13, 17)
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
OPEN_HOLD = 0.30
FIST_HOLD = 0.12
TRACKING_GRACE = 0.18
BURST_DURATION = 0.50
TITLE = "RetroLens | Rasengan"
PHASE = np.linspace(0.0, 1.0, 100)


def joint_angle(a, b, c):
    first, second = a - b, c - b
    lengths = np.linalg.norm(first) * np.linalg.norm(second)
    if lengths < 1e-6:
        return 0.0
    cosine = np.dot(first, second) / lengths
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


def read_hand(landmarks, width, height):
    # MediaPipe's depth coordinate uses roughly the same scale as x.
    points = np.array([(p.x * width, p.y * height, p.z * width) for p in landmarks])
    return describe_hand(points, width, height)


def describe_hand(points, width, height):
    # Compare joint angles in three dimensions so rotating the wrist does not
    # turn an open hand into a fist merely through foreshortening.
    palm_size = max(np.linalg.norm(points[0] - points[9]), 1.0)
    supporting = 0
    tucked = 0
    for base, middle, distal, tip in FINGERS:
        pip_angle = joint_angle(points[base], points[middle], points[distal])
        joints = points[[base, middle, distal, tip]]
        finger_length = np.linalg.norm(np.diff(joints, axis=0), axis=1).sum()
        reach = np.linalg.norm(joints[-1] - joints[0]) / max(finger_length, 1e-6)
        from_wrist = np.linalg.norm(points[tip] - points[0])
        knuckle_from_wrist = np.linalg.norm(points[base] - points[0])
        extension = (from_wrist - knuckle_from_wrist) / max(finger_length, 1e-6)

        if pip_angle <= 118 and (reach <= 0.66 or extension <= 0.24):
            tucked += 1
        elif pip_angle >= 105 and reach >= 0.70 and extension >= 0.30:
            supporting += 1

    pose = "OPEN" if supporting >= 3 else "FIST" if tucked == 4 else "NEUTRAL"
    center = points[list(PALM), :2].mean(axis=0)
    radius = float(np.clip(palm_size * 0.83, 22, min(width, height) * 0.17))

    # When the palm plane turns toward the ceiling, lift the effect in screen
    # space. The absolute normal works for both left and right hands.
    across = points[5] - points[17]
    along = (points[5] + points[17]) * 0.5 - points[0]
    normal = np.cross(across, along)
    vertical = abs(normal[1]) / max(np.linalg.norm(normal), 1e-6)
    lift = float(np.clip((vertical - 0.35) / 0.65, 0, 1))
    center[1] -= radius * 1.32 * lift
    return pose, center, radius, lift


def select_hand(readings, effect, now):
    if not readings:
        return None

    # Stick to the hand already charging or holding the effect even when a
    # second hand enters the frame or MediaPipe returns its hands in a new order.
    recent = effect.last_seen is not None and now - effect.last_seen < TRACKING_GRACE
    if effect.center is not None and recent:
        nearest = min(readings, key=lambda hand: np.linalg.norm(hand[1] - effect.center))
        distance = np.linalg.norm(nearest[1] - effect.center)
        if distance <= max(55, effect.radius * 2.4):
            if effect.started is not None or nearest[0] == "OPEN":
                return nearest
        elif effect.started is not None:
            return None

    return max(readings, key=lambda hand: (hand[0] == "OPEN", hand[3], hand[2]))


class EnergyEffect:
    def __init__(self):
        self.center = None
        self.radius = 0.0
        self.pose = "NO HAND"
        self.pose_since = 0.0
        self.last_seen = None
        self.started = None
        self.burst_origin = None
        self.burst_radius = 0.0
        self.burst_time = None

    def update(self, reading, now):
        was_missing = self.pose == "NO HAND"
        pose = "NO HAND" if reading is None else reading[0]
        if pose != self.pose:
            self.pose = pose
            self.pose_since = now

        if reading is None:
            if self.last_seen is not None and now - self.last_seen >= TRACKING_GRACE:
                self.started = None
                self.center = None
            return

        _, center, radius, _ = reading
        gap = 0.0 if self.last_seen is None else now - self.last_seen
        if was_missing and gap >= TRACKING_GRACE:
            self.started = None
            self.center = None

        if self.center is None:
            self.center, self.radius = center, radius
        else:
            blend = 1.0 - math.exp(-gap / 0.055)
            self.center += (center - self.center) * blend
            self.radius += (radius - self.radius) * blend
        self.last_seen = now

        held = now - self.pose_since
        if pose == "OPEN" and held >= OPEN_HOLD and self.started is None:
            self.started = now
        elif pose == "FIST" and held >= FIST_HOLD and self.started is not None:
            self.burst_origin = self.center.copy()
            self.burst_radius = self.radius
            self.burst_time = now
            self.started = None

    def draw(self, frame, now):
        if self.started is not None and self.center is not None:
            age = now - self.started
            charge = min(age / 0.45, 1.0)
            visibility = max(0.0, 1.0 - (now - self.last_seen) / TRACKING_GRACE)
            if visibility > 0:
                radius = self.radius * (0.25 + 0.75 * (1 - (1 - charge) ** 2))
                draw_orb(frame, self.center, radius, age, visibility)

        if self.burst_time is not None:
            progress = (now - self.burst_time) / BURST_DURATION
            if progress < 1.0:
                draw_burst(frame, self.burst_origin, self.burst_radius, progress)
            else:
                self.burst_time = None


def draw_orb(frame, center, radius, age, opacity):
    height, width = frame.shape[:2]
    radius *= 1 + 0.035 * math.sin(age * 19) + 0.025 * math.sin(age * 33)
    cx, cy = np.rint(center).astype(int)
    margin = math.ceil(radius * 2.1)
    left, right = max(0, cx - margin), min(width, cx + margin + 1)
    top = max(0, cy - margin)
    bottom = min(height, cy + math.ceil(radius * 2.8) + 1)
    if left >= right or top >= bottom:
        return

    shape = (bottom - top, right - left, 3)
    origin = np.array((cx - left, cy - top))
    yy, xx = np.ogrid[top:bottom, left:right]
    x, y = (xx - cx) / radius, (yy - cy) / radius
    distance = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    edge = 0.98 + 0.025 * np.sin(9 * angle - age * 23)
    body = np.clip((edge - distance) * 9.0, 0.0, 1.0)
    core = np.exp(-4.5 * ((x + 0.04) ** 2 + (y + 0.06) ** 2))
    swirl = angle - age * 8.6 + 4.2 * distance
    turbulence = 0.5 + 0.5 * np.sin(38 * distance + 4 * swirl + 1.4 * np.sin(3 * swirl))
    rim = np.exp(-((distance - edge) / 0.080) ** 2)
    halo = np.exp(-1.55 * distance ** 2)
    spill = np.exp(-1.6 * x ** 2 - 2.3 * (y - 1.28) ** 2)
    cyan = 140 + 113 * core + 32 * turbulence + 37 * rim
    white = 35 + 211 * core + 39 * turbulence + 43 * rim
    surface = np.stack((np.broadcast_to(246.0, distance.shape), cyan, white), axis=-1)

    # Project curved ribbons on different planes of the same sphere. The near
    # part is brighter and wider, so the streams read as a spinning volume.
    paths = np.zeros(shape, dtype=np.uint8)
    for index in range(13):
        phase = index * 2.39996
        spin = age * (7.0 + 0.21 * index) + phase
        theta = spin + PHASE * (3.6 + 0.46 * (index % 4))
        curve = 0.69 + 0.071 * (index % 5) + 0.08 * np.sin(theta * 2 + phase)
        tilt = 0.22 + 0.20 * (index % 5)
        u = curve * np.cos(theta)
        v = tilt * curve * np.sin(theta) + 0.065 * np.cos(theta * 3 + phase)
        turn = phase + age * (0.58 if index % 2 else -0.53)
        px = math.cos(turn) * u - math.sin(turn) * v
        py = math.sin(turn) * u + math.cos(turn) * v
        points = np.rint(origin + radius * np.column_stack((px, py))).astype(np.int32)
        depth = np.sin(theta)
        for start in range(0, len(PHASE) - 1, 9):
            end = min(start + 9, len(PHASE) - 1)
            mid = (start + end) // 2
            taper = math.sin(math.pi * PHASE[mid])
            light = taper * (0.34 + 0.66 * max(0.0, depth[mid]))
            color = (round(250 * light), round(244 * light), round(228 * light))
            thickness = max(1, round(radius * (0.009 + 0.025 * light)))
            cv2.polylines(paths, [points[start:end + 1]], False, color, thickness, cv2.LINE_AA)

    # Short streaks around the edge imply energy escaping tangentially.
    for index in range(19):
        phase = index * 2.39996
        life = (age * (1.3 + 0.16 * (index % 4)) + index * 0.618) % 1.0
        theta = phase + age * (4.0 + index % 3)
        reach = 0.96 + life * (0.38 + 0.05 * (index % 6))
        tangent = theta - 0.10 - 0.18 * life
        first = origin + radius * reach * np.array((math.cos(theta), math.sin(theta)))
        last = origin + radius * (reach - 0.08) * np.array((math.cos(tangent), math.sin(tangent)))
        light = math.sin(math.pi * life) ** 2
        color = (round(166 * light), round(122 * light), round(49 * light))
        cv2.line(paths, tuple(np.rint(first).astype(int)), tuple(np.rint(last).astype(int)),
                 color, max(1, round(radius * 0.021)), cv2.LINE_AA)

    bloom = cv2.GaussianBlur(paths, (0, 0), max(1.25, radius * 0.055))
    region = frame[top:bottom, left:right]
    alpha = (body * opacity * 0.94)[..., None]
    result = region * (1 - alpha) + surface * alpha
    result += opacity * (paths * 0.38 + bloom * 0.98)
    result += opacity * halo[..., None] * np.array((63, 42, 8))
    result += opacity * spill[..., None] * np.array((44, 32, 6))
    region[:] = np.clip(result, 0, 255).astype(np.uint8)


def draw_burst(frame, center, radius, progress):
    cx, cy = center
    strength = (1.0 - progress) ** 1.5
    color = (round(255 * strength), round(217 * strength), round(165 * strength))
    layer = np.zeros_like(frame)
    if progress < 0.36:
        draw_orb(frame, center, radius * (1 + progress * 2.8), progress,
                 0.92 * (1 - progress / 0.36) ** 2)
    for index in range(38):
        angle = index * 2.39996
        travel = radius * (0.35 + (1.1 + (index % 9) * 0.25) * progress)
        tail = max(0.0, travel - radius * 0.2 * (1.0 - progress))
        dx, dy = math.cos(angle), math.sin(angle)
        head = (round(cx + dx * travel), round(cy + dy * travel))
        end = (round(cx + dx * tail), round(cy + dy * tail))
        cv2.line(layer, end, head, color, 1 + (index % 3 == 0), cv2.LINE_AA)
    cv2.add(frame, layer, dst=frame)


def preview():
    started = time.monotonic()
    center = np.array((320.0, 228.0))
    while True:
        age = (time.monotonic() - started) % 4.0
        frame = np.full((480, 640, 3), (31, 22, 18), dtype=np.uint8)
        if age < 3.35:
            size = 22 + 78 * min(age / 0.5, 1.0)
            draw_orb(frame, center, size, age, 1.0)
        elif age < 3.85:
            draw_burst(frame, center, 100.0, (age - 3.35) / 0.50)
        if show_frame(frame):
            return


def show_frame(frame):
    cv2.imshow(TITLE, frame)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--preview", action="store_true", help="Show the effect without a webcam")
    args = parser.parse_args()

    if args.preview:
        try:
            preview()
        finally:
            cv2.destroyAllWindows()
        return

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        raise SystemExit("This script uses mp.solutions.hands. Run it with MediaPipe 0.10.9.")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f"Cannot open camera {args.camera}. Check camera permissions or try --camera 1.")

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        effect = EnergyEffect()
        with solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.65,
                                   min_tracking_confidence=0.60) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise SystemExit("Cannot read a frame from the camera.")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = tracker.process(rgb)
                height, width = frame.shape[:2]
                now = time.monotonic()
                readings = [read_hand(hand.landmark, width, height)
                            for hand in (result.multi_hand_landmarks or ())]
                reading = select_hand(readings, effect, now)
                effect.update(reading, now)
                effect.draw(frame, now)
                label = f"{effect.pose}  |  Open or cup palm: charge   Fist: burst   Q: quit"
                for thickness, color in ((3, (18, 22, 30)), (1, (240, 245, 255))):
                    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, color, thickness, cv2.LINE_AA)
                if show_frame(frame):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
