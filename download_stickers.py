"""
Sticker Downloader — run this ONCE before face_expression_detector.py
======================================================================
Downloads all required OpenMoji stickers into a 'stickers/' folder
in the same directory as this script.

Run:
    python3 download_stickers.py
"""

import urllib.request
import os
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
STICKER_DIR = os.path.join(SCRIPT_DIR, "stickers")
BASE_URL    = "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618"

# All sticker codes used by the detector
# Format: "CODE" : "description (for your reference)"
STICKERS = {
    # ── Main expression stickers ──────────────────────────
    "1F604": "😄 happy - main",
    "1F622": "😢 sad - main",
    "1F632": "😲 surprised - main",
    "1F621": "😡 angry - main",
    "1F922": "🤢 disgusted - main",
    "1F628": "😨 fearful - main",
    "1F610": "😐 neutral - main",

    # ── Happy variants ────────────────────────────────────
    "1F602": "😂 laugh-cry",
    "1F60D": "😍 heart eyes",
    "1F609": "😉 wink",
    "1F60A": "😊 blush",

    # ── Sad variants ──────────────────────────────────────
    "1F62D": "😭 sobbing",
    "1F614": "😔 pensive",
    "1F641": "🙁 slightly sad",
    "1F97A": "🥺 pleading",

    # ── Surprised variants ────────────────────────────────
    "1F92F": "🤯 mind blown",
    "1F631": "😱 screaming",
    "1F62E": "😮 open mouth",
    "1F9D0": "🧐 monocle",

    # ── Angry variants ────────────────────────────────────
    "1F92C": "🤬 rage symbols",
    "1F620": "😠 angry",
    "1F611": "😑 expressionless",

    # ── Disgusted variants ────────────────────────────────
    "1F92E": "🤮 vomiting",
    "1F915": "🤕 head bandage",
    "1F616": "😖 confounded",
    "1F44E": "👎 thumbs down",

    # ── Fearful variants ──────────────────────────────────
    "1F630": "😰 cold sweat",
    "1F975": "🥵 hot face",
    "1F976": "🥶 cold face",
    "1F613": "😓 downcast sweat",
}


def download_all():
    os.makedirs(STICKER_DIR, exist_ok=True)
    total   = len(STICKERS)
    success = 0
    failed  = []

    print(f"Downloading {total} stickers to: {STICKER_DIR}\n")

    for i, (code, desc) in enumerate(STICKERS.items(), 1):
        out_path = os.path.join(STICKER_DIR, f"{code}.png")

        # Skip if already downloaded and valid
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            print(f"  [{i:02d}/{total}] SKIP  {code}  {desc}")
            success += 1
            continue

        url = f"{BASE_URL}/{code}.png"
        try:
            urllib.request.urlretrieve(url, out_path)
            size = os.path.getsize(out_path)
            if size < 500:
                raise ValueError(f"File too small ({size} bytes) — likely 404")
            print(f"  [{i:02d}/{total}] OK    {code}  {desc}")
            success += 1
        except Exception as e:
            print(f"  [{i:02d}/{total}] FAIL  {code}  {desc}  →  {e}")
            failed.append(code)
            # Remove bad file
            if os.path.exists(out_path):
                os.remove(out_path)

    print(f"\n{'='*50}")
    print(f"Done!  {success}/{total} stickers downloaded.")

    if failed:
        print(f"\nFailed codes ({len(failed)}): {', '.join(failed)}")
        print("These will be skipped in the detector (no sticker shown for them).")
    else:
        print("All stickers ready! Run face_expression_detector.py now.")


if __name__ == "__main__":
    download_all()
