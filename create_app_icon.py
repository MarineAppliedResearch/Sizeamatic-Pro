"""App icon (.ico) generator for Sizeamatic Pro.

Converts a single source image (any reasonably square, high-resolution
PNG/JPG — 512x512 or larger recommended, PNG with transparency preferred)
into the multi-resolution `assets/icon.ico` the app and its PyInstaller
build (`sizeamatic.spec`, ROADMAP.md Phase 9) both read. Windows expects
one .ico file containing several baked-in sizes, not several separate
files — this script builds exactly that from one image, so swapping in
real branded art later is a single command instead of manual per-size
exporting.

Usage:
    uv run python create_app_icon.py path/to/source_image.png

Author:
    Isaac Travers

Date:
    2026-08-12
"""

import argparse
import os

from PIL import Image

# The practical set for Windows 10/11: covers taskbar, Explorer list/
# details/large-icon views, and shortcut/thumbnail rendering. See
# AGENTS.md's Packaging section for why these specific sizes.
ICON_SIZES = [16, 32, 48, 256]


def build_icon(source_path, output_path):
    """Resize a source image into a multi-resolution .ico file.

    Args:
        source_path (str): Path to the source image (PNG/JPG, any
            aspect ratio — resized to square, so a square source avoids
            distortion).
        output_path (str): Path to write the resulting .ico file to.

    Returns:
        None
    """
    with Image.open(source_path) as img:
        img = img.convert("RGBA")
        img.save(output_path, format="ICO", sizes=[(s, s) for s in ICON_SIZES])


def main():
    """Parse CLI arguments and build assets/icon.ico from a source image.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_image", help="Path to the source icon artwork (PNG/JPG).")
    parser.add_argument(
        "--output",
        default=os.path.join("assets", "icon.ico"),
        help="Where to write the .ico file (default: assets/icon.ico).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    build_icon(args.source_image, args.output)

    print(f"Wrote {args.output} ({', '.join(f'{s}x{s}' for s in ICON_SIZES)})")


if __name__ == "__main__":
    main()
