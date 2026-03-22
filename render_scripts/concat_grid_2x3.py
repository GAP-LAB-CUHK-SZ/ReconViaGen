#!/usr/bin/env python3
"""
Compose 6 videos into a 2x3 grid with column labels.

Layout:
  Col:  reconviagen    |  mvtrellis2    |  reconviagen+trellis2
  Row1: *_color.mp4   |  *_color.mp4  |  *_color.mp4
  Row2: *_normal.mp4  |  *_normal.mp4 |  *_normal.mp4
  [    ReconViaGen    |  MV-TRELLIS.2 |  ReconViaGen+MV-TRELLIS.2  ]

Final output: 2048px wide, ~2K resolution.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
RENDERS_DIR = Path("/root/jiahao/code/ReconViaGen/show_case/case/renders")
OUTPUT = RENDERS_DIR / "grid_2x3.mp4"
FONT_PATH = (
    "/root/miniconda3/envs/reconviagen_v05/lib/python3.10"
    "/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
)

# ── Video properties ───────────────────────────────────────────────────────────
W, H = 1024, 1024   # per-cell size
COLS  = 3
FPS   = 30
DURATION = 8        # seconds (240 frames)
LABEL_H  = 80       # height of label bar (at 3×1024 = 3072 wide)
FONT_SIZE = 44

# Column order: [reconviagen, mvtrellis2, reconviagen+trellis2]
LABELS = ["ReconViaGen", "MV-TRELLIS.2", "ReconViaGen+MV-TRELLIS.2"]

videos = [
    RENDERS_DIR / "reconviagen_color.mp4",
    RENDERS_DIR / "mvtrellis2_color.mp4",
    RENDERS_DIR / "reconviagen+trellis2_color.mp4",
    RENDERS_DIR / "reconviagen_normal.mp4",
    RENDERS_DIR / "mvtrellis2_normal.mp4",
    RENDERS_DIR / "reconviagen+trellis2_normal.mp4",
]

# ── Generate label image with Pillow ──────────────────────────────────────────
def make_label_image(path: Path):
    total_w = W * COLS  # 3072
    img = Image.new("RGB", (total_w, LABEL_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    col_centers = [W // 2, W + W // 2, 2 * W + W // 2]
    for label, cx in zip(LABELS, col_centers):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = cx - tw // 2
        y = (LABEL_H - th) // 2
        draw.text((x, y), label, fill=(255, 255, 255), font=font)

    img.save(str(path))
    print(f"Label image saved: {path} ({total_w}x{LABEL_H})")

# ── Build & run ffmpeg command ─────────────────────────────────────────────────
def run_ffmpeg(label_path: Path):
    # Inputs: 0-5 = videos, 6 = label image
    inputs = []
    for v in videos:
        inputs += ["-i", str(v)]
    # Label image looped for DURATION seconds
    inputs += ["-loop", "1", "-t", str(DURATION), "-framerate", str(FPS),
               "-i", str(label_path)]

    # filter_complex
    fc = (
        # Row 1: color videos side by side
        "[0:v][1:v][2:v]hstack=inputs=3[row0];"
        # Row 2: normal videos side by side
        "[3:v][4:v][5:v]hstack=inputs=3[row1];"
        # Label image: ensure same fps and duration
        f"[6:v]fps={FPS}[label];"
        # Stack vertically
        "[row0][row1][label]vstack=inputs=3[stacked];"
        # Scale to 2K width (height auto, must be even)
        "[stacked]scale=2048:-2[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", fc,
        "-map", "[out]",
        "-c:v", "libopenh264",
        "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        str(OUTPUT),
    ]

    print("\nRunning ffmpeg...")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\nDone! Output: {OUTPUT}")
    else:
        print(f"\nffmpeg failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        label_path = Path(f.name)

    try:
        make_label_image(label_path)
        run_ffmpeg(label_path)
    finally:
        label_path.unlink(missing_ok=True)
