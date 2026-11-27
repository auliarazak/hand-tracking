import cv2
import mediapipe as mp
import time
import math
import numpy as np
from pathlib import Path
from preview import fit_size, fixed_preview

MODEL_DIR = Path(__file__).resolve().parent
MAX_PREVIEW_SIZE = (1280, 720)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / 'hand_landmarker.task')),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

segmenter = None
try:
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_DIR / 'selfie_segmenter.tflite')),
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True
    )
    segmenter = ImageSegmenter.create_from_options(seg_options)
except Exception as e:
    print("Segmenter tidak tersedia (pastikan file selfie_segmenter.tflite ada)", e)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    landmarker.close()
    if segmenter is not None:
        segmenter.close()
    raise SystemExit('Kamera tidak dapat dibuka. Periksa perangkat dan izin kamera.')

galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8) 
galaxy_bg[:] = (30, 10, 40) # Space purple
for _ in range(800):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    galaxy_bg[sy, sx] = (255, 255, 255)
for _ in range(100):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    cv2.circle(galaxy_bg, (sx, sy), np.random.randint(2, 6), (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)

print("Membuka Kamera... Tekan tombol 'q' di keyboard untuk keluar.")

filters = ["MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"]
if segmenter is None:
    filters.remove('GALAXY')
filters.extend(["RED", "CARTOON", "RAINBOW", "WATER", "KALEIDOSCOPE", "DIMENSION"])
current_filter = 0
gesture_triggered = False
multi_finger_mode = False
mesh_mode = False
fists_triggered = False
last_timestamp_ms = -1
window_name = 'Hand Tracking Filter'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
preview_size = None

def draw_hud(image, filter_name, mode):
    lines = (
        "Ganti Filter: Jempol & Kelingking",
        f"Filter Aktif: {filter_name}",
        f"Mode: {mode} | M: jari | C: bidang",
        "N/P: filter | S: foto | Q: keluar",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    longest = max(cv2.getTextSize(line, font, 0.6, 2)[0][0] for line in lines)
    factor = min(1.0, image.shape[1] / (longest + 20), image.shape[0] / 130)
    for row, line in enumerate(lines, start=1):
        position = (round(10 * factor), round(30 * row * factor))
        color = (0, 255, 255) if row == 2 else (255, 255, 255)
        cv2.putText(image, line, position, font, 0.6 * factor, color,
                    max(1, round(2 * factor)), cv2.LINE_AA)


def window_is_open():
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False

def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    elif filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)
    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255] # Orange
        filtered[mask_c == 0] = [147, 20, 255]  # Pink
        return filtered
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (w_r//10, h_r//10), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_name == "GLITCH":
        h_r, w_r = roi.shape[:2]
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi
    elif filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)
    elif filter_name == "RED":
        filtered = np.zeros_like(roi)
        filtered[:, :, 2] = roi[:, :, 2]
        return filtered
    elif filter_name == "CARTOON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        return cv2.bitwise_and(cv2.bilateralFilter(roi, 9, 100, 100),
                               cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    elif filter_name == "RAINBOW":
        bh, bw = roi.shape[:2]
        xx, yy = np.meshgrid(np.arange(bw), np.arange(bh))
        wave = (np.sin((xx + yy) * 0.05 + time.monotonic() * 5) * 127 + 128).astype(np.uint8)
        return cv2.addWeighted(roi, 0.3, cv2.applyColorMap(wave, cv2.COLORMAP_HSV), 0.7, 0)
    elif filter_name == "WATER":
        bh, bw = roi.shape[:2]
        xx, yy = np.meshgrid(np.arange(bw, dtype=np.float32), np.arange(bh, dtype=np.float32))
        t = time.monotonic() * 3
        return cv2.remap(roi, xx + 5 * np.sin(yy * 0.06 + t),
                         yy + 5 * np.sin(xx * 0.06 + t), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    elif filter_name == "KALEIDOSCOPE":
        bh, bw = roi.shape[:2]
        xx, yy = np.meshgrid(np.arange(bw, dtype=np.float32), np.arange(bh, dtype=np.float32))
        dx, dy = xx - bw / 2, yy - bh / 2
        angle = np.mod(np.arctan2(dy, dx), np.pi / 3)
        angle = np.minimum(angle, np.pi / 3 - angle)
        radius = np.hypot(dx, dy)
        return cv2.remap(roi, (bw / 2 + radius * np.cos(angle)).astype(np.float32),
                         (bh / 2 + radius * np.sin(angle)).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif filter_name == "DIMENSION":
        bh, bw = roi.shape[:2]
        xx, yy = np.meshgrid(np.arange(bw, dtype=np.float32), np.arange(bh, dtype=np.float32))
        t = time.monotonic() * 2
        radius = np.hypot((xx - bw / 2) / max(bw, 1), (yy - bh / 2) / max(bh, 1))
        rings = ((np.sin(radius * 45 - t * 3) + 1) * 110).astype(np.uint8)
        color = cv2.applyColorMap(rings, cv2.COLORMAP_TWILIGHT)
        return cv2.addWeighted(roi, 0.35, color, 0.65, 0)
    elif filter_name == "GALAXY" and mask_person is not None and frame_galaxy is not None:
        bh, bw = roi.shape[:2]
        roi_mask = mask_person[y:y+bh, x:x+bw]
        roi_galaxy = frame_galaxy[y:y+bh, x:x+bw]
        
        bg_condition = (roi_mask == 0)
        filtered = roi.copy()
        filtered[bg_condition] = roi_galaxy[bg_condition]
        return filtered
        
    return roi

def render_portal(img, points, filter_name, mask_person, frame_galaxy):
    if len(points) < 3:
        return
    poly = cv2.convexHull(np.asarray(points, dtype=np.int32)).reshape(-1, 2)
    if len(poly) < 3 or cv2.contourArea(poly) <= 10:
        return
    height, width = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(poly)
    right, bottom = min(width, x + bw), min(height, y + bh)
    x, y = max(0, x), max(0, y)
    bw, bh = right - x, bottom - y
    if bw <= 0 or bh <= 0:
        return
    roi = img[y:y + bh, x:x + bw].copy()
    filtered = apply_filter(roi, filter_name, x, y, mask_person, frame_galaxy)
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [poly - [x, y]], 255)
    img[y:y + bh, x:x + bw] = np.where(mask[:, :, None] != 0, filtered, roi)
    cv2.polylines(img, [poly], True, (255, 255, 255), 2)
    for i in range(len(poly)):
        pt1, pt2 = poly[i], poly[(i + 1) % len(poly)]
        for _ in range(5):
            alpha = np.random.random()
            px = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-15, 15)
            py = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-15, 15)
            cv2.circle(img, (px, py), np.random.randint(1, 4), (0, 255, 255), -1)

def hand_is_fist(hand, width, height):
    wrist = np.array([hand[0].x * width, hand[0].y * height])
    def distance(index):
        return np.linalg.norm(np.array([hand[index].x * width, hand[index].y * height]) - wrist)
    return sum(distance(tip) < distance(joint) * 1.25
               for tip, joint in zip((8, 12, 16, 20), (5, 9, 13, 17))) >= 3

try:
  while True:
    if not window_is_open():
        break
    success, img = cap.read()
    if not success:
        print('Tidak dapat membaca frame dari kamera.')
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    frame_galaxy = cv2.resize(galaxy_bg, (w, h)) if h > galaxy_bg.shape[0] or w > galaxy_bg.shape[1] else galaxy_bg[:h, :w]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
    last_timestamp_ms = timestamp_ms
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    filter_name = filters[current_filter]
    
    mask_person = None
    if segmenter is not None and "GALAXY" in [filters[(current_filter + i) % len(filters)] for i in range(3)]:
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask_person = seg_result.category_mask.numpy_view()
            if mask_person.shape != (h, w):
                mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

    change_filter = False

    if results.hand_landmarks:
        if len(results.hand_landmarks) >= 2:
            idx0 = results.hand_landmarks[0][8]
            idx1 = results.hand_landmarks[1][8]
            pt0 = (int(idx0.x * w), int(idx0.y * h))
            pt1 = (int(idx1.x * w), int(idx1.y * h))
            hand = results.hand_landmarks[0]
            palm = math.hypot((hand[5].x - hand[17].x) * w,
                              (hand[5].y - hand[17].y) * h)
            if palm > 0 and math.hypot(pt0[0]-pt1[0], pt0[1]-pt1[1]) < palm * 0.4:
                change_filter = True
                
        for hand_lms in results.hand_landmarks:
            thumb = hand_lms[4]
            tx, ty = int(thumb.x * w), int(thumb.y * h)
                
            pinky = hand_lms[20]
            px, py = int(pinky.x * w), int(pinky.y * h)
            palm = math.hypot((hand_lms[5].x - hand_lms[17].x) * w,
                              (hand_lms[5].y - hand_lms[17].y) * h)
            if palm > 0 and math.hypot(tx - px, ty - py) < palm * 0.4:
                change_filter = True
                
        fists = len(results.hand_landmarks) == 2 and all(
            hand_is_fist(hand, w, h) for hand in results.hand_landmarks)
        if fists:
            change_filter = False
        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(filters)
                gesture_triggered = True
        else:
            gesture_triggered = False
        filter_name = filters[current_filter]

        hands = []
        for hand_lms in results.hand_landmarks:
            tips = [(int(hand_lms[i].x * w), int(hand_lms[i].y * h))
                    for i in (4, 8, 12, 16, 20)]
            hands.append(tips)
            for point in tips if multi_finger_mode or mesh_mode else tips[:2]:
                cv2.circle(img, point, 6, (255, 255, 0), cv2.FILLED)

        if fists and not fists_triggered:
            mesh_mode = not mesh_mode
        fists_triggered = fists

        if mesh_mode and len(hands) == 2:
            first, second = sorted(hands, key=lambda tips: sum(p[0] for p in tips))
            render_portal(img, first[:3] + list(reversed(second[:3])),
                          filter_name, mask_person, frame_galaxy)
            next_filter = filters[(current_filter + 1) % len(filters)]
            render_portal(img, first[2:] + list(reversed(second[2:])),
                          next_filter, mask_person, frame_galaxy)
        elif len(hands) == 2:
            points = (hands[0] + hands[1]) if multi_finger_mode else (hands[0][:2] + hands[1][:2])
            render_portal(img, points, filter_name, mask_person, frame_galaxy)
        elif len(hands) == 1:
            render_portal(img, hands[0], filter_name, mask_person, frame_galaxy)
    else:
        gesture_triggered = False
        fists_triggered = False

    mode = 'Dua bidang' if mesh_mode else ('Semua jari' if multi_finger_mode else 'Jempol & telunjuk')

    if not window_is_open():
        break
    if preview_size is None:
        preview_size = fit_size(w, h, *MAX_PREVIEW_SIZE)
    preview = fixed_preview(img, *preview_size)
    draw_hud(preview, filters[current_filter], mode)
    cv2.imshow(window_name, preview)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        multi_finger_mode = not multi_finger_mode
    elif key == ord('c'):
        mesh_mode = not mesh_mode
    elif key == ord('n'):
        current_filter = (current_filter + 1) % len(filters)
    elif key == ord('p'):
        current_filter = (current_filter - 1) % len(filters)
    elif key == ord('s'):
        filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
        draw_hud(img, filters[current_filter], mode)
        print(f"Foto tersimpan: {filename}" if cv2.imwrite(filename, img) else "Gagal menyimpan foto.")
    if not window_is_open():
        break

finally:
    cap.release()
    landmarker.close()
    if segmenter is not None:
        segmenter.close()
    cv2.destroyAllWindows()
