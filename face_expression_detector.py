"""
Face Expression Detector  —  with Anime/Cartoon Sticker Pack
=============================================================
Uses: OpenCV · MediaPipe (Tasks API ≥0.10) · NumPy · math
      collections.deque + Counter · Pillow · urllib

Sticker pack: OpenMoji (open-source, CC BY-SA 4.0)
Model:        MediaPipe FaceLandmarker (.task file)

── Setup ──────────────────────────────────────────────────
pip install opencv-python mediapipe numpy Pillow

Run:
    python3 face_expression_detector.py

Controls:
    Q  — quit
    M  — toggle face mesh
    S  — toggle stats panel
    P  — toggle sticker panel sidebar
    +/- — resize main sticker
"""

import cv2
import numpy as np
import math
import os
import sys
import urllib.request
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode, FaceLandmarksConnections,
)

# ══════════════════════════════════════════════════════════
#  PATHS & AUTO-DOWNLOAD
# ══════════════════════════════════════════════════════════

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "face_landmarker.task")
STICKER_DIR  = os.path.join(SCRIPT_DIR, "stickers")
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
OPENMOJI_BASE = (
    "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618"
)

# ── Expression → primary sticker + panel variants ────────
STICKER_CONFIG = {
    "happy":     {
        "color": (0, 210, 100),
        "label": "Happy!",
        "main":  "1F604",   # 😄
        "panel": ["1F602", "1F60D", "1F609", "1F60A"],  # 😂😍😉😊
    },
    "sad":       {
        "color": (80, 80, 220),
        "label": "Sad...",
        "main":  "1F622",   # 😢
        "panel": ["1F62D", "1F614", "1F641", "1F97A"],  # 😭😔🙁🥺
    },
    "surprised": {
        "color": (0, 190, 255),
        "label": "Surprised!",
        "main":  "1F632",   # 😲
        "panel": ["1F92F", "1F631", "1F62E", "1F9D0"],  # 🤯😱😮🧐
    },
    "angry":     {
        "color": (0, 0, 230),
        "label": "Angry!",
        "main":  "1F621",   # 😡
        "panel": ["1F92C", "1F620", "1F610", "1F611"],  # 🤬😠😐😑
    },
    "disgusted": {
        "color": (30, 150, 30),
        "label": "Disgusted!",
        "main":  "1F922",   # 🤢
        "panel": ["1F92E", "1F915", "1F616", "1F44E"],  # 🤮🤕😖👎
    },
    "fearful":   {
        "color": (180, 0, 200),
        "label": "Fearful!",
        "main":  "1F628",   # 😨
        "panel": ["1F630", "1F975", "1F976", "1F613"],  # 😰🥵🥶😓
    },
    "neutral":   {
        "color": (150, 150, 150),
        "label": "Neutral",
        "main":  "1F610",   # 😐
        "panel": ["1F60F", "1F614", "1F644", "1F60C"],  # 😏😔🙄😌
    },
}

ALL_CODES = set()
for cfg in STICKER_CONFIG.values():
    ALL_CODES.add(cfg["main"])
    ALL_CODES.update(cfg["panel"])


def _download(url, path, label=""):
    try:
        urllib.request.urlretrieve(url, path)
        return os.path.getsize(path) > 500
    except Exception as e:
        print(f"  warn: could not download {label}: {e}")
        return False


def ensure_assets():
    """Download model + stickers if missing."""
    # Model
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        print("Downloading MediaPipe model (~5 MB)…")
        ok = _download(MODEL_URL, MODEL_PATH, "model")
        if not ok:
            sys.exit(
                "ERROR: Could not download model.\n"
                f"Run manually:\n  curl -L -o face_landmarker.task '{MODEL_URL}'"
            )
        print("  model OK")

    # Stickers
    os.makedirs(STICKER_DIR, exist_ok=True)
    missing = [c for c in ALL_CODES
               if not os.path.exists(os.path.join(STICKER_DIR, f"{c}.png"))
               or os.path.getsize(os.path.join(STICKER_DIR, f"{c}.png")) < 500]
    if missing:
        print(f"Downloading {len(missing)} sticker(s)…")
        for code in missing:
            url  = f"{OPENMOJI_BASE}/{code}.png"
            path = os.path.join(STICKER_DIR, f"{code}.png")
            ok   = _download(url, path, code)
            print(f"  {'OK' if ok else 'FAIL'} {code}")
    print("Assets ready.\n")


# ══════════════════════════════════════════════════════════
#  STICKER CACHE
# ══════════════════════════════════════════════════════════

_sticker_cache: dict = {}

def get_sticker(code: str, size: int) -> Image.Image | None:
    key = (code, size)
    if key not in _sticker_cache:
        path = os.path.join(STICKER_DIR, f"{code}.png")
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("RGBA")
            img = img.resize((size, size), Image.LANCZOS)
            _sticker_cache[key] = img
        except Exception:
            return None
    return _sticker_cache[key]


def paste_sticker(frame_bgr: np.ndarray, sticker: Image.Image, x: int, y: int):
    """Alpha-composite a PIL RGBA sticker onto an OpenCV BGR frame."""
    fh, fw = frame_bgr.shape[:2]
    sw, sh = sticker.size

    # Clamp
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + sw), min(fh, y + sh)
    if x2 <= x1 or y2 <= y1:
        return

    crop = sticker.crop((x1 - x, y1 - y, x2 - x, y2 - y))

    # Convert frame ROI to PIL RGBA, composite, write back
    roi      = frame_bgr[y1:y2, x1:x2]
    roi_pil  = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert("RGBA")
    roi_pil.paste(crop, (0, 0), crop)
    frame_bgr[y1:y2, x1:x2] = cv2.cvtColor(
        np.array(roi_pil.convert("RGB")), cv2.COLOR_RGB2BGR
    )


# ══════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ══════════════════════════════════════════════════════════

LEFT_EYE       = [362, 385, 387, 263, 373, 380]
RIGHT_EYE      = [33,  160, 158, 133, 153, 144]
LEFT_BROW      = [336, 296, 334, 293, 300]
RIGHT_BROW     = [107,  66, 105,  63,  70]
LEFT_EYE_TOP   = [386, 374]
RIGHT_EYE_TOP  = [159, 145]
MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOT = 61, 291, 13, 14
FACE_LEFT,  FACE_RIGHT                        = 234, 454


# ══════════════════════════════════════════════════════════
#  GEOMETRY FEATURES
# ══════════════════════════════════════════════════════════

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def xy(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)

def ear(lm, idx, w, h):
    p = [xy(lm, i, w, h) for i in idx]
    return (dist(p[1],p[5]) + dist(p[2],p[4])) / (2*dist(p[0],p[3]) + 1e-6)

def mar(lm, w, h):
    return dist(xy(lm,MOUTH_TOP,w,h), xy(lm,MOUTH_BOT,w,h)) / \
           (dist(xy(lm,MOUTH_LEFT,w,h), xy(lm,MOUTH_RIGHT,w,h)) + 1e-6)

def mouth_width(lm, w, h):
    return dist(xy(lm,MOUTH_LEFT,w,h), xy(lm,MOUTH_RIGHT,w,h)) / \
           (dist(xy(lm,FACE_LEFT,w,h), xy(lm,FACE_RIGHT,w,h)) + 1e-6)

def brow_raise(lm, w, h):
    fh = abs(lm[152].y - lm[10].y) + 1e-6
    lg = np.mean([lm[i].y for i in LEFT_EYE_TOP])  - np.mean([lm[i].y for i in LEFT_BROW])
    rg = np.mean([lm[i].y for i in RIGHT_EYE_TOP]) - np.mean([lm[i].y for i in RIGHT_BROW])
    return ((lg + rg) / 2) / fh

def lip_angle(lm, w, h):
    l, r = xy(lm,MOUTH_LEFT,w,h), xy(lm,MOUTH_RIGHT,w,h)
    return math.degrees(math.atan2(-(r[1]-l[1]), r[0]-l[0]+1e-6))

def classify(lm, w, h):
    e   = (ear(lm, LEFT_EYE, w, h) + ear(lm, RIGHT_EYE, w, h)) / 2
    m   = mar(lm, w, h)
    mw  = mouth_width(lm, w, h)
    br  = brow_raise(lm, w, h)
    ang = lip_angle(lm, w, h)

    if m > 0.35 and br > 0.085 and e > 0.27:  return "surprised"
    if m > 0.25 and br > 0.080 and e < 0.30:  return "fearful"
    if br < 0.055 and m < 0.15 and e < 0.28:  return "angry"
    if br < 0.060 and m > 0.10 and ang < -3:  return "disgusted"
    if mw > 0.44  and ang > 3  and e > 0.24:  return "happy"
    if ang < -4   and br < 0.075:             return "sad"
    return "neutral"


# ══════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════

PANEL_W = 130   # sidebar width for sticker panel

def draw_mesh(frame, lm, w, h):
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        s, e_ = conn.start, conn.end
        if s < len(lm) and e_ < len(lm):
            p1 = (int(lm[s].x*w), int(lm[s].y*h))
            p2 = (int(lm[e_].x*w), int(lm[e_].y*h))
            cv2.line(frame, p1, p2, (55, 55, 55), 1, cv2.LINE_AA)
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        s, e_ = conn.start, conn.end
        if s < len(lm) and e_ < len(lm):
            p1 = (int(lm[s].x*w), int(lm[s].y*h))
            p2 = (int(lm[e_].x*w), int(lm[e_].y*h))
            cv2.line(frame, p1, p2, (0, 200, 100), 1, cv2.LINE_AA)

def draw_label(frame, expression, conf):
    cfg   = STICKER_CONFIG.get(expression, STICKER_CONFIG["neutral"])
    text  = f"{cfg['label']}  {conf:.0%}"
    color = cfg["color"]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    cv2.rectangle(frame, (8, 8), (tw+28, th+22), (15, 15, 15), -1)
    cv2.rectangle(frame, (8, 8), (tw+28, th+22), color, 2)
    cv2.putText(frame, text, (18, th+14),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

def draw_stats(frame, e_val, m_val, mw_val, br_val, ang_val, expression, conf):
    fh, fw = frame.shape[:2]
    ph, pw = 158, 255
    x0, y0 = 10, fh - ph - 10
    ov = frame.copy()
    cv2.rectangle(ov, (x0,y0), (x0+pw,y0+ph), (15,15,15), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    color = STICKER_CONFIG.get(expression, STICKER_CONFIG["neutral"])["color"]
    rows = [
        (f"Expression : {expression.upper()}", color),
        (f"Confidence : {conf:.0%}",           color),
        (f"EAR        : {e_val:.3f}",   (200,200,200)),
        (f"MAR        : {m_val:.3f}",   (200,200,200)),
        (f"Mouth W    : {mw_val:.3f}",  (200,200,200)),
        (f"Brow Raise : {br_val:.3f}",  (200,200,200)),
        (f"Lip Angle  : {ang_val:.1f}°",(200,200,200)),
    ]
    for i, (t, c) in enumerate(rows):
        cv2.putText(frame, t, (x0+8, y0+20+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, c, 1, cv2.LINE_AA)

def draw_hint(frame):
    fh, fw = frame.shape[:2]
    for i, h in enumerate(["Q:quit","M:mesh","S:stats","P:panel","+/-:size"]):
        cv2.putText(frame, h, (fw-95, 20+i*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140,140,140), 1, cv2.LINE_AA)

def draw_sticker_panel(frame, expression):
    """
    Right sidebar showing the 4 variant stickers for current expression.
    Semi-transparent dark panel with stickers laid out vertically.
    """
    fh, fw   = frame.shape[:2]
    cfg      = STICKER_CONFIG.get(expression, STICKER_CONFIG["neutral"])
    panel_codes = cfg["panel"]
    color    = cfg["color"]

    pad     = 8
    cell_sz = 90
    panel_h = fh
    x0      = fw - PANEL_W

    # Dark background
    ov = frame.copy()
    cv2.rectangle(ov, (x0, 0), (fw, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

    # Colour accent bar on left edge of panel
    cv2.rectangle(frame, (x0, 0), (x0+3, panel_h), color, -1)

    # Title
    cv2.putText(frame, "STICKERS", (x0+10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # Sticker tiles
    for i, code in enumerate(panel_codes):
        sticker = get_sticker(code, cell_sz)
        if sticker is None:
            continue
        sx = x0 + (PANEL_W - cell_sz) // 2
        sy = 32 + i * (cell_sz + pad)
        if sy + cell_sz > fh:
            break
        paste_sticker(frame, sticker, sx, sy)


# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════

def main():
    ensure_assets()

    BUFFER     = deque(maxlen=15)
    show_mesh  = False
    show_stats = True
    show_panel = True
    sticker_sz = 160    # main sticker size (px)
    expression = "neutral"
    conf       = 0.0

    # ── Build FaceLandmarker ──────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            if show_mesh:
                draw_mesh(frame, lm, w, h)

            # Features
            e_val  = (ear(lm, LEFT_EYE, w, h) + ear(lm, RIGHT_EYE, w, h)) / 2
            m_val  = mar(lm, w, h)
            mw_val = mouth_width(lm, w, h)
            br_val = brow_raise(lm, w, h)
            ang_val= lip_angle(lm, w, h)

            # Smooth
            BUFFER.append(classify(lm, w, h))
            expression, count = Counter(BUFFER).most_common(1)[0]
            conf = count / len(BUFFER)

            if show_stats:
                draw_stats(frame, e_val, m_val, mw_val, br_val, ang_val,
                           expression, conf)

        # ── Sticker panel (sidebar) ───────────────────────
        if show_panel:
            draw_sticker_panel(frame, expression)

        # ── Main sticker (top-right, left of panel) ───────
        cfg     = STICKER_CONFIG.get(expression, STICKER_CONFIG["neutral"])
        sticker = get_sticker(cfg["main"], sticker_sz)
        if sticker:
            panel_offset = PANEL_W if show_panel else 0
            sx = w - sticker_sz - panel_offset - 8
            sy = 8
            paste_sticker(frame, sticker, sx, sy)

        draw_label(frame, expression, conf)
        draw_hint(frame)

        cv2.imshow("Face Expression Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):                         break
        elif key == ord('m'): show_mesh  = not show_mesh
        elif key == ord('s'): show_stats = not show_stats
        elif key == ord('p'): show_panel = not show_panel
        elif key == ord('+') or key == ord('='): sticker_sz = min(240, sticker_sz + 20)
        elif key == ord('-'):                    sticker_sz = max(60,  sticker_sz - 20)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
