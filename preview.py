import cv2

def fit_size(width, height, max_width, max_height):
    scale = min(max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def fixed_preview(frame, width, height):
    h, w = frame.shape[:2]
    size = fit_size(w, h, width, height)
    interpolation = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
    view = cv2.resize(frame, size, interpolation=interpolation)
    dx, dy = width - size[0], height - size[1]
    if dx or dy:
        view = cv2.copyMakeBorder(view, dy // 2, dy - dy // 2, dx // 2, dx - dx // 2,
                                  cv2.BORDER_CONSTANT)
    return view
