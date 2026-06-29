# 😄 Face Expression Detector with Sticker Pack

A fun, real-time face expression detection system that overlays expressive emoji stickers on your face based on detected emotions. Uses **MediaPipe**, **OpenCV**, and the **OpenMoji** sticker pack.

## 🎯 Features

- **Real-time emotion detection**: Happy, Sad, Surprised, Angry, Disgusted, Fearful, Neutral
- **Dynamic emoji stickers**: Main sticker display + sidebar panel with expression variants
- **Facial mesh visualization**: Optional overlay of 468 face landmarks
- **Live statistics panel**: View calculated facial features (EAR, MAR, Brow Raise, Lip Angle, etc.)
- **Interactive controls**: Toggle mesh, stats, sticker panel; resize stickers on-the-fly
- **Auto-download assets**: Model and sticker pack download automatically on first run

## 📋 Requirements

- Python 3.8+
- Webcam

## 📦 Installation

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe numpy Pillow
```

### 2. Download Stickers (Optional - Auto-downloads on first run)

```bash
python3 download_stickers.py
```

This downloads ~40 OpenMoji stickers to the `stickers/` folder. The detector will automatically download missing stickers when you run it.

## 🚀 Quick Start

```bash
python3 face_expression_detector.py
```

The first run will download the MediaPipe face landmarker model (~5 MB). Subsequent runs load it instantly.

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **M** | Toggle face mesh visualization |
| **S** | Toggle statistics panel |
| **P** | Toggle sticker panel sidebar |
| **+** | Increase main sticker size (max 240px) |
| **-** | Decrease main sticker size (min 60px) |

## 🎨 Expression Detection

The detector classifies 7 emotions based on facial features:

| Expression | Icon | Features |
|------------|------|----------|
| **Happy** | 😄 | Wide mouth, raised cheeks, open eyes |
| **Sad** | 😢 | Downturned lips, raised brows |
| **Surprised** | 😲 | High MAR, raised brows, open eyes |
| **Angry** | 😡 | Low brow raise, closed mouth |
| **Disgusted** | 🤢 | Downturned lips, closed eyes |
| **Fearful** | 😨 | High MAR, high brow raise, open eyes |
| **Neutral** | 😐 | Balanced features |

### Classification Algorithm

The detector uses 5 key facial metrics:

- **EAR** (Eye Aspect Ratio): Measures eye openness
- **MAR** (Mouth Aspect Ratio): Measures mouth openness
- **Mouth Width**: Horizontal mouth spread relative to face width
- **Brow Raise**: Vertical distance between brows and eyes
- **Lip Angle**: Orientation of the mouth corners

A decision tree combines these features to classify expressions in real-time.

## 📂 File Structure

```
sticker_fun/
├── face_expression_detector.py    # Main detector script
├── download_stickers.py           # Sticker downloader utility
├── face_landmarker.task           # MediaPipe model (auto-downloaded)
└── stickers/                       # Emoji sticker pack (auto-downloaded)
    ├── 1F604.png  (😄 happy)
    ├── 1F622.png  (😢 sad)
    └── ...
```

## 🖥️ UI Layout

```
┌─────────────────────────────┐
│ Hint Panel (top-right)      │  ← Q:quit M:mesh S:stats P:panel +/-:size
│ Label: "Happy! 95%"         │  ← Expression label + confidence
│                             │
│   Main Video Feed           │  ← Your face + main sticker (top-right)
│   + Sticker (160-240px)     │
│   + Optional Mesh           │  ← 468 facial landmarks (if M:toggled)
│   + Optional Stats          │  ← Feature values (if S:toggled)
│                             │
│                      ┌──────┤  ← STICKER PANEL (if P:toggled)
│                      │ TICS │
│                      │ ────│
│                      │ 😂   │
│                      │ 😍   │
│                      │ 😉   │
│                      │ 😊   │
└──────────────────────┴──────┘
```

## 📊 Statistics Panel

When enabled (press **S**), displays:

- **Expression**: Detected emotion (smoothed over 15 frames)
- **Confidence**: % of recent frames agreeing on the expression
- **EAR**: Eye Aspect Ratio (higher = eyes more open)
- **MAR**: Mouth Aspect Ratio (higher = mouth more open)
- **Mouth W**: Mouth width as % of face width
- **Brow Raise**: Vertical distance between brows & eyes as % of face height
- **Lip Angle**: Angle of mouth corners in degrees

## 🎯 How It Works

1. **Capture Frame**: Read from webcam (640×480)
2. **Detect Landmarks**: MediaPipe FaceLandmarker finds 468 facial points
3. **Extract Features**: Calculate EAR, MAR, mouth width, brow raise, lip angle
4. **Classify**: Decision tree outputs emotion label
5. **Smooth**: Buffer last 15 predictions, pick majority vote
6. **Render**: Draw stickers, stats, mesh (if enabled)
7. **Display**: Show on screen with interactive overlays

## 🔄 Frame Smoothing

Expressions are smoothed using a **15-frame circular buffer** with majority voting. This reduces jitter and provides more stable, confident predictions.

## 📥 Asset Management

### Model Download

- **URL**: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- **Size**: ~5 MB
- **Location**: Saved as `face_landmarker.task` in the script directory

### Sticker Pack

- **Source**: OpenMoji (CC BY-SA 4.0 licensed)
- **URLs**: GitHub raw content CDN (`https://raw.githubusercontent.com/hfg-gmuend/openmoji/...`)
- **Format**: 618×618 PNG with transparency
- **Total**: ~40 stickers (~5 MB)

Auto-downloads happen silently on first run. Use `download_stickers.py` to pre-download or re-download manually.

## 🎨 Customization

### Change Sticker Size Range

Edit in `face_expression_detector.py`:

```python
sticker_sz = 160    # Initial size (px)
# In the key handler:
elif key == ord('+') or key == ord('='):
    sticker_sz = min(240, sticker_sz + 20)   # Max size
elif key == ord('-'):
    sticker_sz = max(60, sticker_sz - 20)    # Min size
```

### Add or Remove Expression Variants

Edit `STICKER_CONFIG`:

```python
STICKER_CONFIG = {
    "happy": {
        "color": (0, 210, 100),
        "label": "Happy!",
        "main":  "1F604",   # Change this code
        "panel": ["1F602", "1F60D", "1F609", "1F60A"],  # or these
    },
    ...
}
```

### Adjust Detection Sensitivity

Modify thresholds in the `classify()` function:

```python
def classify(lm, w, h):
    # Adjust these thresholds to make detection more/less sensitive
    if m > 0.35 and br > 0.085 and e > 0.27:  return "surprised"
    ...
```

## 📝 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | Latest | Video capture & rendering |
| `mediapipe` | ≥0.10 | Face landmark detection |
| `numpy` | Latest | Numerical computations |
| `Pillow` | Latest | Image loading & compositing |

## ⚠️ Troubleshooting

### Webcam not detected
```
ERROR: Cannot open webcam.
```
- Check if another app is using the webcam
- Try restarting the script
- Verify camera permissions on your OS

### Sticker download fails
- Check internet connection
- Stickers will be skipped if download fails; detector continues without them
- Run `download_stickers.py` manually for detailed error messages

### Model download fails
```
ERROR: Could not download model.
Run manually: curl -L -o face_landmarker.task '...'
```
- Download manually using the provided curl command
- Place the `.task` file in the script directory

### Performance issues
- Reduce camera resolution (edit `cap.set()` calls)
- Disable mesh (`M` key)
- Disable stats panel (`S` key)
- Use fewer sticker variants

## 📄 License

- **Code**: Your choice (add LICENSE file)
- **Stickers**: [OpenMoji](https://openmoji.org/) – CC BY-SA 4.0
- **Model**: MediaPipe – Apache 2.0

## 🎓 Credits

- **MediaPipe**: Google's machine learning framework
- **OpenMoji**: Open-source emoji design
- **OpenCV**: Computer vision library

## 🚀 Future Enhancements

- [ ] Multi-face detection & sticker assignment
- [ ] Pose detection (full-body stickers)
- [ ] Hand gesture recognition
- [ ] Video export with stickers
- [ ] Custom sticker pack upload
- [ ] Confidence threshold adjustment
- [ ] Recording feature for clips

## 💬 Questions?

Open an issue or check the inline code comments for detailed explanations.

---

**Enjoy! 🎉**
