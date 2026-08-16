"""Tests for prepare_splash_image.py."""

from PIL import Image

import prepare_splash_image


def test_flatten_splash_removes_transparency_by_compositing_onto_background(tmp_path):
    """A source image with partial transparency should come out fully
    opaque, composited onto the requested background color - this is
    the fix for PyInstaller's Tk-based splash renderer showing corrupted
    magenta/pink instead of blending semi-transparent pixels
    (ROADMAP.md Phase 9)."""

    source_path = str(tmp_path / "source.png")
    # A pixel that's 50% transparent white over an otherwise-transparent
    # image - if this weren't flattened, Tk's splash loader would be the
    # one deciding what shows through, which is exactly the bug being
    # worked around here.
    img = Image.new("RGBA", (4, 4), (255, 255, 255, 128))
    img.save(source_path)

    output_path = str(tmp_path / "flattened.png")
    prepare_splash_image.flatten_splash(source_path, output_path, background=(10, 20, 30))

    with Image.open(output_path) as flat:
        assert flat.mode == "RGB"
        # ~50% white over (10, 20, 30) background.
        assert flat.getpixel((0, 0)) == (133, 138, 143)


def test_flatten_splash_leaves_fully_opaque_pixels_unchanged(tmp_path):
    """A pixel that was already fully opaque shouldn't be altered by
    compositing - only the transparent/semi-transparent pixels should
    actually change."""

    source_path = str(tmp_path / "source.png")
    img = Image.new("RGBA", (2, 2), (200, 50, 75, 255))
    img.save(source_path)

    output_path = str(tmp_path / "flattened.png")
    prepare_splash_image.flatten_splash(source_path, output_path, background=(0, 0, 0))

    with Image.open(output_path) as flat:
        assert flat.getpixel((0, 0)) == (200, 50, 75)


def test_add_version_text_draws_text_without_altering_image_size():
    """Drawing the version string onto a splash image shouldn't change
    its dimensions or mutate the original image object - only a copy
    with the text drawn on should come back."""

    original = Image.new("RGB", (200, 100), (0, 0, 0))
    original_bytes = original.tobytes()

    result = prepare_splash_image.add_version_text(original, "v0.1.0")

    assert result.size == original.size
    # The original passed in should be untouched (drawn on a copy).
    assert original.tobytes() == original_bytes
    # Something in the bottom-right region should no longer be pure
    # black, since that's where the version text gets drawn.
    corner_region = [
        result.getpixel((x, y))
        for x in range(result.width - 40, result.width)
        for y in range(result.height - 20, result.height)
    ]
    assert any(pixel != (0, 0, 0) for pixel in corner_region)
