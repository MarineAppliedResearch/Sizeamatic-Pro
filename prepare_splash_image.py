"""Splash-image flattening for Sizeamatic Pro's startup splash.

Used two ways (ROADMAP.md Phase 9):

- `main.py`'s own Tkinter startup splash calls `flatten_splash_image`
  in-memory, on every run (source or packaged), while the app builds.
- `sizeamatic.spec` calls the file-writing `flatten_splash` at build
  time to prepare the image PyInstaller's separate *bootloader* splash
  uses (shown even earlier, during a onefile build's self-extraction,
  before Python/Tk even start).

Both need flattening for the same reason: this project's splash art has
partial alpha (a glow effect fading to transparent at the edges), and
naive Tcl/Tk-level image loading does not composite semi-transparent
pixels the way a modern image viewer does — it shows corrupted
magenta/pink artifacts instead of blending into the window's
background. Baking the alpha channel onto a solid background color
first avoids this entirely.

Usage:
    uv run python prepare_splash_image.py assets/splash-pro.png assets/_splash_flattened.png

Author:
    Isaac Travers

Date:
    2026-08-12
"""

import argparse

from PIL import Image, ImageDraw, ImageFont


def flatten_splash_image(source_path, background=(0, 0, 0)):
    """Flatten a splash image's alpha channel onto a solid background.

    Args:
        source_path (str): Path to the source splash image (PNG, with
            or without an alpha channel).
        background (tuple[int, int, int]): The RGB color to composite
            transparent/semi-transparent pixels onto. Defaults to pure
            black, matching this project's splash art's own background.

    Returns:
        PIL.Image.Image: The flattened, fully-opaque image, in memory.
    """
    with Image.open(source_path) as img:
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, background)
        flat.paste(img, mask=img.getchannel("A"))
        return flat


def add_version_text(img, version_text, color=(255, 255, 255)):
    """Draw a version string into a splash image's bottom-right corner.

    Args:
        img (PIL.Image.Image): The (already-flattened) splash image to
            draw onto. Not modified in place — a copy is drawn on and
            returned, so the caller's original image stays reusable.
        version_text (str): The text to draw, e.g. "v0.1.0".
        color (tuple[int, int, int]): Text color.

    Returns:
        PIL.Image.Image: A new image with the version text drawn on.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)

    margin = max(12, out.width // 40)
    font_size = max(12, out.height // 22)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), version_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = out.width - text_w - margin - bbox[0]
    y = out.height - text_h - margin - bbox[1]
    draw.text((x, y), version_text, fill=color, font=font)

    return out


def flatten_splash(source_path, output_path, background=(0, 0, 0)):
    """Flatten a splash image's alpha channel and write it to a file.

    Args:
        source_path (str): Path to the source splash image (PNG, with
            or without an alpha channel).
        output_path (str): Path to write the fully-opaque flattened PNG
            to.
        background (tuple[int, int, int]): The RGB color to composite
            transparent/semi-transparent pixels onto. Defaults to pure
            black, matching this project's splash art's own background.

    Returns:
        None
    """
    flatten_splash_image(source_path, background).save(output_path)


def main():
    """Parse CLI arguments and flatten a splash image.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_image", help="Path to the source splash PNG.")
    parser.add_argument("output_image", help="Path to write the flattened PNG to.")
    args = parser.parse_args()

    flatten_splash(args.source_image, args.output_image)
    print(f"Wrote {args.output_image}")


if __name__ == "__main__":
    main()
