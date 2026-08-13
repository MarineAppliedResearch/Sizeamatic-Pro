"""Tests for build_version_info.py."""

import build_version_info


def test_parse_version_tuple_handles_a_plain_three_part_version():
    """The common case: a plain "X.Y.Z" string parses to the matching
    integer tuple."""
    assert build_version_info.parse_version_tuple("0.1.0") == (0, 1, 0)
    assert build_version_info.parse_version_tuple("2.10.3") == (2, 10, 3)


def test_parse_version_tuple_pads_missing_components_with_zero():
    """A version string with fewer than 3 dot-separated parts shouldn't
    raise - Windows version resources always need exactly 3 (well, 4
    with a build number), so missing ones default to 0."""
    assert build_version_info.parse_version_tuple("1.2") == (1, 2, 0)
    assert build_version_info.parse_version_tuple("5") == (5, 0, 0)


def test_parse_version_tuple_strips_a_trailing_pre_release_suffix():
    """A version like "0.1.0rc1" should parse using each part's leading
    digits only, rather than raising on the non-numeric suffix."""
    assert build_version_info.parse_version_tuple("0.1.0rc1") == (0, 1, 0)


def test_write_version_file_embeds_the_version_string(tmp_path):
    """The generated version-info file should contain the exact version
    string in its FileVersion/ProductVersion fields, and the parsed
    integer tuple in its FixedFileInfo, so PyInstaller's EXE(version=...)
    picks both up correctly."""
    output_path = str(tmp_path / "version_info.txt")
    build_version_info.write_version_file("0.1.0", output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'StringStruct(u"FileVersion", u"0.1.0")' in content
    assert 'StringStruct(u"ProductVersion", u"0.1.0")' in content
    assert "filevers=(0, 1, 0, 0)" in content
    assert "prodvers=(0, 1, 0, 0)" in content
