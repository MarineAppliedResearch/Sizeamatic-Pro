"""Tests for create_app_icon.py."""

from PIL import Image

import create_app_icon


def test_build_icon_writes_a_multi_resolution_ico_with_all_expected_sizes(tmp_path):
    """The .ico file created from a single source image should contain
    every size listed in ICON_SIZES, so Windows always has an
    appropriately-sized frame to pick rather than upscaling a smaller
    one (ROADMAP.md Phase 9)."""

    source_path = str(tmp_path / "source.png")
    Image.new("RGBA", (512, 512), (10, 20, 30, 255)).save(source_path)

    output_path = str(tmp_path / "icon.ico")
    create_app_icon.build_icon(source_path, output_path)

    with Image.open(output_path) as ico:
        sizes = {tuple(size) for size in ico.ico.sizes()}

    assert sizes == {(s, s) for s in create_app_icon.ICON_SIZES}


def test_build_icon_accepts_a_non_square_source_image(tmp_path):
    """A non-square source shouldn't crash the conversion - Pillow fits
    it within each requested size box, preserving aspect ratio (so a
    2:1 source becomes e.g. 256x128 rather than a distorted 256x256),
    rather than raising a square-only validation error."""

    source_path = str(tmp_path / "source.png")
    Image.new("RGBA", (800, 400), (10, 20, 30, 255)).save(source_path)

    output_path = str(tmp_path / "icon.ico")
    create_app_icon.build_icon(source_path, output_path)

    with Image.open(output_path) as ico:
        w, h = ico.size
        assert max(w, h) in create_app_icon.ICON_SIZES
        assert w == 2 * h
