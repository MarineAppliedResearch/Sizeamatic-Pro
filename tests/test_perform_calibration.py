"""Tests for perform_calibration.py's PerformCalibrationWindow.

Uses a FakeApp (via make_fake_app) rather than the real SizeamaticProApp
for most tests: PerformCalibrationWindow only ever reads/writes a
handful of specific app attributes (see AGENTS.md's Testing section for
why a minimal stand-in is preferred over the full GUI here) - listed in
this module's own docstring.
"""

import os
import queue

import cv2
import numpy as np
import tkinter as tk

import perform_calibration


class _FakeCapture:
    """Minimal stand-in for a `cv2.VideoCapture`, used to test
    `_run_auto_scan`'s frame-sampling/stride/offset logic against known
    canned frames without needing a real decodable video file."""

    def __init__(self, frames_by_index):
        """Store the canned frames this stand-in should "decode".

        Args:
            frames_by_index (dict[int, numpy.ndarray]): Mapping of
                frame index to the BGR frame `.read()` should return
                once positioned there - an index with no entry behaves
                like a failed/out-of-range read.

        Returns:
            None
        """
        self._frames_by_index = frames_by_index
        self._pos = 0

    def set(self, _prop, value):
        """Record the requested frame position.

        Args:
            _prop: The `cv2.CAP_PROP_*` constant, unused - this stand-in
                only ever gets asked to seek by frame position.
            value (int): The frame index to seek to.

        Returns:
            None
        """
        self._pos = int(value)

    def read(self):
        """Return the canned frame at the current position, if any.

        Returns:
            tuple[bool, numpy.ndarray | None]: `(True, frame)` if a
            frame was recorded for the current position, else
            `(False, None)`.
        """
        frame = self._frames_by_index.get(self._pos)
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        """No-op, matching the real `cv2.VideoCapture.release()` signature.

        Returns:
            None
        """


def _make_checkerboard_bgr_frame():
    """Build a real, detectable synthetic checkerboard BGR frame.

    Returns:
        numpy.ndarray: A BGR image containing a
        `perform_calibration.CHECKERBOARD_INNER_CORNERS`-sized
        checkerboard pattern that `cv2.findChessboardCorners` can
        actually detect - not just a placeholder image.
    """
    square_px = 60
    cols, rows = 10, 7
    gray = np.zeros((rows * square_px, cols * square_px), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                gray[row * square_px : (row + 1) * square_px, col * square_px : (col + 1) * square_px] = 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_blank_bgr_frame():
    """Build a plain BGR frame with no detectable calibration board.

    Returns:
        numpy.ndarray: An all-black BGR image.
    """
    return np.zeros((420, 600, 3), dtype=np.uint8)


class _FakeVar:
    """Minimal `.get()`/`.set()` stand-in for a Tk variable (IntVar/
    StringVar), shared by every test in this file that needs one
    without requiring a real Tk root."""

    def __init__(self, value=""):
        """Store the initial value this stand-in should report.

        Args:
            value: The initial value `.get()` should return.

        Returns:
            None
        """
        self._value = value

    def set(self, value):
        """Store a new value.

        Args:
            value: The new value to store.

        Returns:
            None
        """
        self._value = value

    def get(self):
        """Return the stored value.

        Returns:
            The most recently stored value.
        """
        return self._value


def _fake_frame():
    """Build a tiny fake BGR frame for capture tests.

    Returns:
        numpy.ndarray: A 4x4 all-black BGR frame, uint8 - just enough
        for `cv2.imwrite` to accept as a valid image, without needing a
        real decoded video frame.
    """
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_scan_capture_folder_returns_empty_list_before_a_folder_is_chosen(make_fake_app):
    """Scanning before any capture folder has been chosen should read
    back as an empty list, not raise."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)

    assert win._scan_capture_folder() == []


def test_scan_capture_folder_finds_only_matched_left_right_pairs(tmp_path, make_fake_app):
    """Only IDs present on *both* the left and right side should count
    as a pair - an unmatched file (e.g. one side manually deleted
    outside the app) should be silently excluded rather than raising or
    showing up as half a pair."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)

    # A real matched pair.
    (tmp_path / "left_0001.png").write_bytes(b"")
    (tmp_path / "right_0001.png").write_bytes(b"")

    # An unmatched left-only file - should not appear as a pair.
    (tmp_path / "left_0002.png").write_bytes(b"")

    # A file that doesn't match the naming convention at all.
    (tmp_path / "notes.txt").write_bytes(b"")

    assert win._scan_capture_folder() == [(1, "left_0001.png", "right_0001.png")]


def test_next_pair_id_starts_at_one_for_an_empty_folder(tmp_path, make_fake_app):
    """With no existing captures, the next pair ID should be 1."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)

    assert win._next_pair_id() == 1


def test_next_pair_id_continues_after_the_highest_existing_pair(tmp_path, make_fake_app):
    """With pairs 1-3 already captured, the next ID should be 4 - not
    reused, so resuming an earlier session's folder never overwrites
    what's already there."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)

    for pair_id in (1, 2, 3):
        (tmp_path / f"left_{pair_id:04d}.png").write_bytes(b"")
        (tmp_path / f"right_{pair_id:04d}.png").write_bytes(b"")

    assert win._next_pair_id() == 4


def test_load_metadata_returns_empty_dict_before_a_folder_is_chosen(make_fake_app):
    """Loading metadata before any capture folder has been chosen should
    read back as an empty dict, not raise."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)

    assert win._load_metadata() == {}


def test_load_metadata_returns_empty_dict_for_a_corrupt_file(tmp_path, make_fake_app):
    """A metadata file that isn't valid JSON should just read back as
    empty - a missing/corrupt metadata file only means "no jump-to-frame
    info available yet", not something to surface as an error."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    (tmp_path / "capture_metadata.json").write_text("not valid json {{{", encoding="utf-8")

    assert win._load_metadata() == {}


def test_save_and_load_metadata_round_trips(tmp_path, make_fake_app):
    """Saving metadata and loading it back should return exactly what
    was saved."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)

    metadata = {"1": {"left_frame_index": 10, "right_frame_index": 13}}
    win._save_metadata(metadata)

    assert win._load_metadata() == metadata
    assert os.path.isfile(tmp_path / "capture_metadata.json")


def test_on_capture_frame_pair_requires_both_videos_loaded(monkeypatch, tmp_path, make_fake_app):
    """Capturing without both videos loaded should show an error and
    write nothing, rather than crashing on a missing capture."""

    app = make_fake_app(capL=None, capR=None)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_capture_frame_pair()

    assert len(errors) == 1
    assert list(tmp_path.iterdir()) == []


def test_on_capture_frame_pair_requires_a_capture_folder(monkeypatch, make_fake_app):
    """Capturing without a capture folder chosen yet should show an
    error rather than crashing trying to write into `None`."""

    app = make_fake_app(
        capL=object(),
        capR=object(),
        left_frame_index=None,
        right_frame_index=None,
    )
    win = perform_calibration.PerformCalibrationWindow(app)

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_capture_frame_pair()

    assert len(errors) == 1


def test_on_capture_frame_pair_reports_a_decode_failure(monkeypatch, tmp_path, make_fake_app):
    """If reading the current frame fails on either side, this should
    report an error and not write a half-saved pair."""

    app = make_fake_app(
        capL=object(),
        capR=object(),
        left_frame_index=_FakeVar(0),
        right_frame_index=_FakeVar(0),
        _read_frame_at=lambda cap, index: None,
    )
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_capture_frame_pair()

    assert len(errors) == 1
    assert list(tmp_path.iterdir()) == []


def test_on_capture_frame_pair_saves_raw_frames_as_a_numbered_pair(tmp_path, make_fake_app):
    """A successful capture should write left_0001.png/right_0001.png
    (reading via app._read_frame_at, never app.current_frameL/
    current_frameR - see the module docstring for why), record both
    frame indices in capture_metadata.json, and update the status line
    with the new pair count."""

    read_calls = []

    def _fake_read_frame_at(cap, index):
        """Record the call and return a fake frame, standing in for
        `main.py`'s real `_read_frame_at`.

        Args:
            cap: The fake capture object passed through.
            index (int): The requested frame index.

        Returns:
            numpy.ndarray: A tiny fake BGR frame.
        """
        read_calls.append((cap, index))
        return _fake_frame()

    left_cap = object()
    right_cap = object()
    app = make_fake_app(
        capL=left_cap,
        capR=right_cap,
        left_frame_index=_FakeVar(10),
        right_frame_index=_FakeVar(13),
        _read_frame_at=_fake_read_frame_at,
    )
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win.status_var = _FakeVar("")

    win.on_capture_frame_pair()

    assert os.path.isfile(tmp_path / "left_0001.png")
    assert os.path.isfile(tmp_path / "right_0001.png")
    assert read_calls == [(left_cap, 10), (right_cap, 13)]
    assert "Captured pair 0001" in win.status_var.get()
    assert "1 total" in win.status_var.get()
    assert win._load_metadata() == {"1": {"left_frame_index": 10, "right_frame_index": 13}}


def test_ensure_window_reuses_the_existing_window_instead_of_rebuilding(
    hidden_tk_root, make_fake_app
):
    """Calling ensure_window twice should not build a second Toplevel -
    the second call should just raise the existing one, matching
    CalibrationSummaryWindow/MeasurementWindow's own ensure_window
    pattern."""

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)

    win.ensure_window()
    first_window = win.win

    win.ensure_window()

    assert win.win is first_window

    win._on_close()


def test_choosing_a_capture_folder_updates_the_display_and_pairs_list(
    hidden_tk_root, monkeypatch, tmp_path, make_fake_app
):
    """Choosing a folder that already has captured pairs in it (e.g.
    resuming an earlier session) should show that folder's path and
    populate the pairs list immediately, not just on the next capture."""

    (tmp_path / "left_0001.png").write_bytes(b"")
    (tmp_path / "right_0001.png").write_bytes(b"")

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.ensure_window()

    monkeypatch.setattr(
        "perform_calibration.filedialog.askdirectory", lambda **_kwargs: str(tmp_path)
    )

    win.on_choose_capture_folder()

    assert win.capture_folder == str(tmp_path)
    assert win.folder_var.get() == str(tmp_path)
    assert win.pairs_listbox.size() == 1

    win._on_close()


def test_on_pair_double_clicked_jumps_the_video_to_the_saved_frame_indices(
    hidden_tk_root, tmp_path, make_fake_app
):
    """Double-clicking a captured pair should move both panes' frame
    index and slider to exactly the position that pair was captured
    from, then re-render - mirroring main.py's own
    _suppress_slider_callbacks jump pattern used elsewhere."""

    (tmp_path / "left_0007.png").write_bytes(b"")
    (tmp_path / "right_0007.png").write_bytes(b"")

    render_calls = []
    label_calls = []

    app = make_fake_app(
        root=hidden_tk_root,
        left_frame_index=_FakeVar(0),
        right_frame_index=_FakeVar(0),
        left_slider=_FakeVar(0),
        right_slider=_FakeVar(0),
        _suppress_slider_callbacks=False,
        _render_current_frames=lambda: render_calls.append(True),
        _update_frame_labels=lambda: label_calls.append(True),
    )
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win._save_metadata({"7": {"left_frame_index": 40, "right_frame_index": 43}})
    win.ensure_window()

    win.pairs_listbox.selection_set(0)
    win.on_pair_double_clicked()

    assert app.left_frame_index.get() == 40
    assert app.right_frame_index.get() == 43
    assert app.left_slider.get() == 40
    assert app.right_slider.get() == 43
    assert app._suppress_slider_callbacks is False
    assert render_calls == [True]
    assert label_calls == [True]

    win._on_close()


def test_on_pair_double_clicked_reports_an_error_for_a_pair_with_no_metadata(
    hidden_tk_root, monkeypatch, tmp_path, make_fake_app
):
    """A pair with no matching metadata entry (e.g. an image dropped
    into the folder outside this window) should report a clear error
    rather than crashing with a KeyError."""

    (tmp_path / "left_0001.png").write_bytes(b"")
    (tmp_path / "right_0001.png").write_bytes(b"")

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win.ensure_window()

    win.pairs_listbox.selection_set(0)

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_pair_double_clicked()

    assert len(errors) == 1

    win._on_close()


def test_on_pair_double_clicked_does_nothing_without_a_selection(hidden_tk_root, make_fake_app):
    """Double-clicking with nothing selected in the list should just be
    a no-op, not raise."""

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.ensure_window()

    win.on_pair_double_clicked()

    win._on_close()


def test_on_delete_selected_pair_removes_files_and_metadata(hidden_tk_root, monkeypatch, tmp_path, make_fake_app):
    """Deleting a selected pair (after confirming) should remove both
    its image files and its metadata entry, and refresh the list."""

    (tmp_path / "left_0001.png").write_bytes(b"")
    (tmp_path / "right_0001.png").write_bytes(b"")

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win._save_metadata({"1": {"left_frame_index": 5, "right_frame_index": 5}})
    win.ensure_window()

    win.pairs_listbox.selection_set(0)
    monkeypatch.setattr("perform_calibration.messagebox.askyesno", lambda title, msg: True)

    win.on_delete_selected_pair()

    assert not os.path.isfile(tmp_path / "left_0001.png")
    assert not os.path.isfile(tmp_path / "right_0001.png")
    assert win._load_metadata() == {}
    assert win.pairs_listbox.size() == 0
    assert "Deleted pair 0001" in win.status_var.get()

    win._on_close()


def test_on_delete_selected_pair_keeps_files_if_not_confirmed(hidden_tk_root, monkeypatch, tmp_path, make_fake_app):
    """Answering "no" to the delete confirmation should leave the files
    untouched."""

    (tmp_path / "left_0001.png").write_bytes(b"")
    (tmp_path / "right_0001.png").write_bytes(b"")

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win.ensure_window()

    win.pairs_listbox.selection_set(0)
    monkeypatch.setattr("perform_calibration.messagebox.askyesno", lambda title, msg: False)

    win.on_delete_selected_pair()

    assert os.path.isfile(tmp_path / "left_0001.png")
    assert os.path.isfile(tmp_path / "right_0001.png")

    win._on_close()


def test_on_delete_selected_pair_requires_a_selection(hidden_tk_root, monkeypatch, make_fake_app):
    """Clicking delete with nothing selected should show a clear error
    rather than silently doing nothing."""

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.ensure_window()

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_delete_selected_pair()

    assert len(errors) == 1

    win._on_close()


def test_on_space_key_captures_when_no_text_entry_is_focused(hidden_tk_root, make_fake_app):
    """Pressing Space while some non-text widget (or nothing) has focus
    should trigger a capture.

    Stubs `app.root.focus_get` directly rather than relying on real Tk
    focus transfer, which isn't reliable against the shared *withdrawn*
    `hidden_tk_root` used across the suite (see conftest.py) - this
    tests `_on_space_key`'s own isinstance guard, not Tk's focus
    machinery itself.
    """

    captures = []
    app = make_fake_app(root=hidden_tk_root)
    app.root.focus_get = lambda: None
    win = perform_calibration.PerformCalibrationWindow(app)
    win.on_capture_frame_pair = lambda: captures.append(True)

    win._on_space_key()

    assert captures == [True]


def test_on_space_key_does_nothing_while_a_text_entry_is_focused(hidden_tk_root, make_fake_app):
    """Pressing Space while a text entry has keyboard focus should not
    trigger a capture - Space should still just type a literal space in
    the entry (e.g. one of the main window's real-time-sync boxes)."""

    captures = []
    entry = tk.Entry(hidden_tk_root)

    app = make_fake_app(root=hidden_tk_root)
    app.root.focus_get = lambda: entry
    win = perform_calibration.PerformCalibrationWindow(app)
    win.on_capture_frame_pair = lambda: captures.append(True)

    win._on_space_key()

    assert captures == []

    entry.destroy()


def test_frame_has_checkerboard_detects_a_real_checkerboard():
    """A real synthetic checkerboard image should be detected."""

    assert perform_calibration.frame_has_checkerboard(
        cv2.cvtColor(_make_checkerboard_bgr_frame(), cv2.COLOR_BGR2GRAY)
    )


def test_frame_has_checkerboard_does_not_detect_a_blank_frame():
    """A plain blank frame should not be detected as a checkerboard."""

    assert not perform_calibration.frame_has_checkerboard(
        cv2.cvtColor(_make_blank_bgr_frame(), cv2.COLOR_BGR2GRAY)
    )


def test_frame_has_charuco_board_detects_a_real_charuco_board():
    """A real generated ChArUco board image (matching this module's own
    board settings) should be detected."""

    detector = perform_calibration.build_charuco_detector()
    dictionary = cv2.aruco.getPredefinedDictionary(perform_calibration.CHARUCO_DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (perform_calibration.CHARUCO_SQUARES_X, perform_calibration.CHARUCO_SQUARES_Y),
        perform_calibration.CHARUCO_SQUARE_SIZE_MM,
        perform_calibration.CHARUCO_MARKER_SIZE_MM,
        dictionary,
    )
    board_image = board.generateImage((1100, 800), marginSize=20)

    assert perform_calibration.frame_has_charuco_board(board_image, detector)


def test_frame_has_calibration_board_accepts_either_board_type():
    """frame_has_calibration_board should recognize a checkerboard even
    though it only tries ChArUco detection second."""

    detector = perform_calibration.build_charuco_detector()

    assert perform_calibration.frame_has_calibration_board(_make_checkerboard_bgr_frame(), detector)
    assert not perform_calibration.frame_has_calibration_board(_make_blank_bgr_frame(), detector)


def test_run_auto_scan_saves_only_sampled_frames_with_a_detected_board(monkeypatch, tmp_path, make_fake_app):
    """The scan worker should sample at AUTO_SCAN_FRAME_STRIDE, save a
    pair only where both sides have a detectable board, record the
    right frame indices in metadata, and post a final "done" message
    with the total found count."""

    checkerboard = _make_checkerboard_bgr_frame()
    blank = _make_blank_bgr_frame()
    stride = perform_calibration.AUTO_SCAN_FRAME_STRIDE

    # Boards present at sampled indices 0 and 2*stride; blank at 1*stride
    # and 3*stride - only the first two should end up saved.
    frames_by_index = {
        0: checkerboard,
        stride: blank,
        2 * stride: checkerboard,
        3 * stride: blank,
    }

    captures = iter(
        [
            _FakeCapture(dict(frames_by_index)),
            _FakeCapture(dict(frames_by_index)),
        ]
    )
    monkeypatch.setattr(perform_calibration.cv2, "VideoCapture", lambda _path: next(captures))

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win._scan_queue = queue.Queue()
    win._scan_stop_requested = False

    win._run_auto_scan("left.mp4", "right.mp4", 0, 3 * stride)

    pairs = win._scan_capture_folder()
    assert len(pairs) == 2
    assert win._load_metadata() == {
        "1": {"left_frame_index": 0, "right_frame_index": 0},
        "2": {"left_frame_index": 2 * stride, "right_frame_index": 2 * stride},
    }

    messages = []
    while True:
        try:
            messages.append(win._scan_queue.get_nowait())
        except queue.Empty:
            break

    assert messages[-1] == ("done", 2)


def test_run_auto_scan_stops_early_when_requested(monkeypatch, tmp_path, make_fake_app):
    """Setting _scan_stop_requested should stop the scan before it
    finishes sampling the whole range."""

    checkerboard = _make_checkerboard_bgr_frame()
    stride = perform_calibration.AUTO_SCAN_FRAME_STRIDE
    frames_by_index = {i * stride: checkerboard for i in range(10)}

    captures = iter(
        [
            _FakeCapture(dict(frames_by_index)),
            _FakeCapture(dict(frames_by_index)),
        ]
    )
    monkeypatch.setattr(perform_calibration.cv2, "VideoCapture", lambda _path: next(captures))

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.capture_folder = str(tmp_path)
    win._scan_queue = queue.Queue()
    win._scan_stop_requested = False

    # Stop after the very first sampled frame is processed - simulated by
    # flipping the flag inside a wrapped detection call.
    original_detect = perform_calibration.frame_has_calibration_board
    call_count = []

    def _detect_then_stop(*args, **kwargs):
        """Flip the stop flag after the first detection call, so the
        scan loop exits on its very next iteration.

        Returns:
            bool: Whatever the real detector returns.
        """
        call_count.append(True)
        if len(call_count) >= 2:
            win._scan_stop_requested = True
        return original_detect(*args, **kwargs)

    monkeypatch.setattr(perform_calibration, "frame_has_calibration_board", _detect_then_stop)

    win._run_auto_scan("left.mp4", "right.mp4", 0, 9 * stride)

    # Only the first couple of sampled frames should have been processed
    # before the stop flag took effect - nowhere near all 10.
    assert len(win._scan_capture_folder()) < 10


def test_on_auto_scan_requires_both_videos_loaded(monkeypatch, make_fake_app):
    """Starting a scan without both videos loaded should show an error
    rather than starting a thread with nothing to scan."""

    app = make_fake_app(left_video_path=None, right_video_path=None)
    win = perform_calibration.PerformCalibrationWindow(app)

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_auto_scan_for_candidates()

    assert len(errors) == 1
    assert win._scan_thread is None


def test_on_auto_scan_requires_a_capture_folder(monkeypatch, make_fake_app):
    """Starting a scan without a capture folder chosen should show an
    error rather than starting a thread with nowhere to save to."""

    app = make_fake_app(left_video_path="left.mp4", right_video_path="right.mp4")
    win = perform_calibration.PerformCalibrationWindow(app)

    errors = []
    monkeypatch.setattr(
        "perform_calibration.messagebox.showerror", lambda title, msg: errors.append(msg)
    )

    win.on_auto_scan_for_candidates()

    assert len(errors) == 1
    assert win._scan_thread is None


def test_on_auto_scan_stops_an_already_running_scan_instead_of_starting_another(make_fake_app):
    """Clicking the button again while a scan is already running should
    request it stop, not start a second scan on top of it."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win._scan_thread = object()  # Any non-None sentinel - "a scan is running".
    win.status_var = _FakeVar("")

    win.on_auto_scan_for_candidates()

    assert win._scan_stop_requested is True


def test_on_stop_scan_sets_the_stop_flag(make_fake_app):
    """on_stop_scan should set the flag _run_auto_scan checks between samples."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.status_var = _FakeVar("")

    win.on_stop_scan()

    assert win._scan_stop_requested is True


def test_poll_scan_queue_does_nothing_once_the_window_is_closed(make_fake_app):
    """_poll_scan_queue should refuse to touch anything once self.win is
    None, rather than erroring trying to call .after on a destroyed
    window."""

    app = make_fake_app()
    win = perform_calibration.PerformCalibrationWindow(app)
    win.win = None
    win._scan_queue = queue.Queue()
    win._scan_queue.put(("done", 3))

    # Should simply return without raising.
    win._poll_scan_queue()


def test_poll_scan_queue_resets_state_and_relabels_button_when_done(hidden_tk_root, make_fake_app):
    """Once the "done" message is drained, the scan state should reset
    and the button should go back to its idle label - not stay stuck
    reading "Stop Scan" forever."""

    app = make_fake_app(root=hidden_tk_root)
    win = perform_calibration.PerformCalibrationWindow(app)
    win.ensure_window()

    win._scan_thread = object()
    win._scan_queue = queue.Queue()
    win._scan_queue.put(("progress", 15, 45, 1))
    win._scan_queue.put(("done", 1))

    win._poll_scan_queue()

    assert win._scan_thread is None
    assert win._scan_queue is None
    assert win.auto_scan_button.cget("text") == "Auto-Scan for Candidates"
    assert "Scan finished" in win.status_var.get()

    win._on_close()
