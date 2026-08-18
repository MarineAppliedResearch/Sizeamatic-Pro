"""V1 "Getting Started" tutorial content - the operational workflow.

Covers exactly the checklist finalized during ROADMAP.md Phase 15's
Step 0 session: loading video/calibration, viewing/syncing, placing and
understanding measurements, recording/managing them, and project save/
load. Deliberately excludes Perform Calibration's *creation* flow,
Generate Calibration Target, the anaglyph 3D preview, and playback
speed controls - all confirmed out of scope for v1.

This is content only - no engine logic lives here (see
`tutorial_engine.py`) and no Qt/widget code lives here (see
`tutorial_window.py`). Kept as its own module, separate from a future
`tutorial_content_calculations.py` (the calculations-tutorial track
Phase 15 is explicitly built to extend into later), so both content
sets plug into the same `tutorial_engine.Tutorial` unmodified.

**Scientific-accuracy sign-off:** the `point_quality_metrics` and
`total_row_and_chain_sigma` steps below explain `ReprojRMS(px)`,
`RayResidual(mm)`, and the `sigma`/`sigma_jac` columns - this text was
reviewed and approved by the project owner during this phase's manual
proof-test pass (2026-08-17), per Step 0's decision that this exact
content needed sign-off before the phase could close.

Author:
    Isaac Travers

Created:
    2026-08-17
"""

from tutorial_engine import TutorialStep

STEPS = [
    # ---- Loading video and calibration ----
    TutorialStep(
        step_id="load_left_video",
        title="Load the Left Video",
        description="Click File → Load Left Video… and choose the left camera's footage.",
        details=(
            "The left video is treated as the reference (master) timeline - its frame "
            "number, elapsed time, and any real-world time sync you set later all "
            "describe the left video specifically, with the right video following it."
        ),
        target=("main", "menu", "File"),
        completion_action="load_left_video",
    ),
    TutorialStep(
        step_id="load_right_video",
        title="Load the Right Video",
        description="Click File → Load Right Video… and choose the matching right camera's footage.",
        details=(
            "Left and right must be the same stereo capture, just from each camera's own "
            "view - Sizeamatic Pro doesn't check this for you, so double-check you've "
            "picked the correct pair before moving on."
        ),
        target=("main", "menu", "File"),
        completion_action="load_right_video",
    ),
    # ---- Real-time sync (done immediately once a video's loaded, before
    # calibration/rectification - it's purely a left-video-frame-to-real-
    # time mapping and doesn't depend on either) ----
    TutorialStep(
        step_id="enter_time_year",
        title="Enter the Year",
        description="Read the 4-digit year from this tutorial video's on-screen clock and type it into the YYYY box.",
        details="The six real-time boxes are plain text fields, filled in one at a time - typing into this one auto-advances to the next once it's full.",
        target=("main", "widget", "real_time_year_edit"),
        completion_action="enter_time_year",
    ),
    TutorialStep(
        step_id="enter_time_month",
        title="Enter the Month",
        description="Read the 2-digit month from this tutorial video's on-screen clock and type it into the MM box.",
        details="Same pattern as the year box - once 2 digits are entered, focus moves to the next box automatically.",
        target=("main", "widget", "real_time_month_edit"),
        completion_action="enter_time_month",
    ),
    TutorialStep(
        step_id="enter_time_day",
        title="Enter the Day",
        description="Read the 2-digit day from this tutorial video's on-screen clock and type it into the DD box.",
        details="Same pattern as the year box - once 2 digits are entered, focus moves to the next box automatically.",
        target=("main", "widget", "real_time_day_edit"),
        completion_action="enter_time_day",
    ),
    TutorialStep(
        step_id="enter_time_hour",
        title="Enter the Hour",
        description="Read the 2-digit hour from this tutorial video's on-screen clock and type it into the HH box.",
        details="24-hour time, matching whatever the video's on-screen clock shows.",
        target=("main", "widget", "real_time_hour_edit"),
        completion_action="enter_time_hour",
    ),
    TutorialStep(
        step_id="enter_time_minute",
        title="Enter the Minute",
        description="Read the 2-digit minute from this tutorial video's on-screen clock and type it into the MM box.",
        details="Same pattern as the other boxes - 2 digits, then focus auto-advances.",
        target=("main", "widget", "real_time_minute_edit"),
        completion_action="enter_time_minute",
    ),
    TutorialStep(
        step_id="enter_time_second",
        title="Enter the Second",
        description="Read the 2-digit second from this tutorial video's on-screen clock and type it into the SS box.",
        details="The last of the six boxes - once it's filled, the next step covers actually anchoring the time you entered.",
        target=("main", "widget", "real_time_second_edit"),
        completion_action="enter_time_second",
    ),
    TutorialStep(
        step_id="real_time_sync",
        title="Set the Real-Time Sync Anchor",
        description="Click Set Time Sync to anchor the date/time you just entered to the current frame.",
        details=(
            "This anchors the video's frame numbers to a real-world date and time - once "
            "set, the Frame/Video Time/Actual Time readout below the video panes keeps "
            "calculating forward and backward from that anchor as you scrub, and every "
            "recorded measurement also carries that calculated real-world time. If the "
            "window is narrow enough that the toolbar hides this button behind a \">>\" "
            "arrow, click that arrow first to reveal it."
        ),
        target=("main", "widget", "btn_set_time_sync"),
        completion_action="set_real_time_sync",
    ),
    TutorialStep(
        step_id="load_calibration",
        title="Load the Calibration",
        description="Click Calibration → Load Calibration… and choose any file inside the calibration folder.",
        details=(
            "A calibration folder holds four files: calibration_intrinsics.npz, "
            "calibration_extrinsics.npz, calibration_rectification.npz, and "
            "calibration_maps.npz. Picking any one of them loads the whole folder - "
            "Sizeamatic Pro also checks the calibration's resolution matches your "
            "loaded videos, and will tell you if it doesn't."
        ),
        target=("main", "menu", "Calibration"),
        completion_action="load_calibration",
    ),
    # ---- Viewing and syncing ----
    TutorialStep(
        step_id="toggle_rectified_view",
        title="Switch to Rectified View",
        description="Open the View menu and check Show Rectified.",
        details=(
            "Rectified view warps each frame so matching points in the left and right "
            "images line up on the same horizontal row - this is what the underlying "
            "stereo math assumes, so real measurements are only accurate in rectified "
            "view, not raw view."
        ),
        target=("main", "menu", "View"),
        completion_action="toggle_rectified",
    ),
    TutorialStep(
        step_id="rectified_indicator",
        title="The RECTIFIED / NOT RECTIFIED Indicator",
        description="Look at the toolbar indicator - it reads RECTIFIED in green, or NOT RECTIFIED in red.",
        details=(
            "This is a safety check: since measurements are only "
            "meaningful in rectified view, this indicator (and the color of clicked "
            "points/lines on screen) makes it obvious at a glance which mode you're in, "
            "so a measurement never gets mistaken for a real one by accident."
        ),
        target=("main", "widget", "rectified_indicator"),
        completion_action=None,
    ),
    TutorialStep(
        step_id="lock_and_resync",
        title="Lock and Resync the Two Timelines",
        description="Compare the \"Frame:\" counter burned into each video, then set the Offset box until they match.",
        details=(
            "This tutorial video's left and right cameras are deliberately out of "
            "sync, like two real cameras that didn't start recording at the exact "
            "same instant - scrub either timeline and compare the burned-in Frame "
            "counter in each pane. \"Lock L and R\" (already checked) keeps both "
            "timelines moving together once aligned, preserving whatever offset is "
            "currently set - the Offset box is that exact offset (right frame index "
            "minus left). Set it once, and playback/scrubbing keeps them aligned "
            "from then on."
        ),
        target=("main", "widget", "offset_spin"),
        completion_action="set_resync_offset",
    ),
    TutorialStep(
        step_id="pan_and_zoom",
        title="Pan and Zoom a Video Pane",
        description="Scroll the mouse wheel over a video pane to zoom, and middle-click-drag to pan.",
        details=(
            "Each pane keeps its own independent zoom/pan. Right-click-drag on empty "
            "space also pans (it only starts the point-refine gesture if you start the "
            "drag on top of an existing point). View → Reset Pan/Zoom puts a pane back "
            "to its default fit-to-window view if you get lost."
        ),
        target=("main", "widget", "pane_left"),
        completion_action="pan_or_zoom",
    ),
    # ---- Placing and understanding measurements ----
    TutorialStep(
        step_id="clear_points",
        title="Clear Points",
        description="Click Clear Points to remove every point currently placed in both panes.",
        details="Useful before starting a fresh measurement, or if a chain of points gets tangled and it's easier to start over than fix each one.",
        target=("main", "widget", "btn_clear_points"),
        completion_action="clear_points",
    ),
    TutorialStep(
        step_id="place_left_point_1",
        title="Place a Point in the Left Video",
        description="Left-click a point in the LEFT video pane.",
        details="Click directly on the exact feature you want to measure from - there's nothing to fix afterward, so place it carefully the first time.",
        target=("main", "widget", "pane_left"),
        completion_action="place_left_point_1",
    ),
    TutorialStep(
        step_id="place_right_point_1",
        title="Place the Matching Point in the Right Video",
        description="Left-click the SAME real-world feature in the RIGHT video pane.",
        details=(
            "This is what pairs the two clicks into one measurable 3D point - click as "
            "precisely as you can on the exact same feature you clicked in the left pane. "
            "You can still drag a placed point afterward to nudge it, or right-click-drag "
            "it to have Sizeamatic Pro search nearby for a better match automatically."
        ),
        target=("main", "widget", "pane_right"),
        completion_action="place_right_point_1",
    ),
    TutorialStep(
        step_id="point_quality_metrics",
        title="Disp, dY, ReprojRMS, and RayResidual",
        description="Look at the Measurement window's results table for your placed point's quality metrics.",
        details=(
            "Disp (px) is the horizontal pixel difference between your left and right "
            "clicks - this is what drives the computed depth. dY (px) is the vertical "
            "difference between them - in a well-rectified pair this should be close to "
            "zero, so a large dY usually means a mis-click rather than a calibration "
            "problem. ReprojRMS (px) and RayResidual (mm) are two different consistency "
            "checks on the same point, not two views of the same number: ReprojRMS "
            "measures how far your clicked pixels are from where the computed 3D point "
            "would land if projected back into the images. RayResidual measures how far "
            "apart the two cameras' actual sightlines to your clicks pass each other in "
            "real space, in millimeters - closer to zero means your left/right clicks "
            "agree with each other better."
        ),
        target=("measurement_window", "widget", "results_table"),
        completion_action=None,
    ),
    TutorialStep(
        step_id="place_left_point_2",
        title="Add a Second Point (Left)",
        description="Left-click a second point in the LEFT pane to start a Segment.",
        details=(
            "You're not limited to one segment - keep placing points (up to 20 per "
            "pane) to build a connected chain, each consecutive pair becoming its own "
            "Segment row, summed into one Total row."
        ),
        target=("main", "widget", "pane_left"),
        completion_action="place_left_point_2",
    ),
    TutorialStep(
        step_id="place_right_point_2",
        title="Add the Matching Second Point (Right)",
        description="Left-click the SAME feature in the RIGHT pane to complete the Segment.",
        details="Same idea as the first point pair - clicking the same real-world feature in both panes is what makes this Segment's length accurate.",
        target=("main", "widget", "pane_right"),
        completion_action="place_right_point_2",
    ),
    TutorialStep(
        step_id="total_row_and_chain_sigma",
        title="The Total Row and Chain Sigma",
        description="With two or more points placed, look at the Total row at the bottom of the results table.",
        details=(
            "The Total row sums every connected segment's length into one number. Its "
            "uncertainty (sigma) isn't just added up directly - it's combined as the "
            "square root of the sum of each segment's own squared uncertainty "
            "(quadrature sum), the standard way to combine independent measurement "
            "errors into one total."
        ),
        target=("measurement_window", "widget", "results_table"),
        completion_action=None,
    ),
    # ---- Recording and managing measurements ----
    TutorialStep(
        step_id="record_measurement",
        title="Record the Measurement",
        description="Click Record in the Measurement window to add the current measurement to the Log.",
        details=(
            "Nothing is recorded automatically while you're still adjusting points - "
            "Record is an explicit action, so the Log only fills with measurements you "
            "actually meant to keep. Every row from one Record click shares a "
            "Measurement ID, so a whole bad chain can be found together later."
        ),
        target=("measurement_window", "widget", "record_button"),
        completion_action="record_measurement",
    ),
    TutorialStep(
        step_id="edit_log",
        title="Fixing a Bad Recorded Row",
        description="Try it now: click into the Log, change one of the recorded values, or delete a whole row.",
        details="There's no separate undo/delete button by design - editing the Log's text is itself the fix/delete mechanism, so you have full control over exactly what stays in it.",
        target=("measurement_window", "widget", "log_text"),
        completion_action=None,
    ),
    # ---- Saving and reopening projects ----
    TutorialStep(
        step_id="save_project",
        title="Save Project",
        description="Click File → Save Project… and give it a memorable name (e.g. \"tutorial_test\") so you recognize the file later.",
        details="A saved project remembers your video/calibration paths, sync offset, rectified-view state, and the Measurement window's Log - reopening it puts you right back where you left off.",
        target=("main", "menu", "File"),
        completion_action="save_project",
    ),
    TutorialStep(
        step_id="open_project",
        title="Open Project",
        description="Click File → Open Project… and choose the project file you just saved.",
        details="Opening a project also jumps both timelines back to whichever frame your last Recorded measurement was on, and re-places those exact points, so it's visibly back on screen, not just a number in the Log.",
        target=("main", "menu", "File"),
        completion_action="open_project",
    ),
    TutorialStep(
        step_id="recent_projects",
        title="Recent Projects",
        description="Open File → Recent Projects to reopen one of your last few projects without a file dialog.",
        details="This list quietly drops any project that's since been moved, renamed, or deleted, so it never shows an entry that would just error if clicked.",
        target=("main", "menu", "File"),
        completion_action="open_recent_project",
    ),
    TutorialStep(
        step_id="copy_to_clipboard",
        title="Copy Measurements to the Clipboard",
        description="Select the text in the Copy box and copy it - it pastes directly into a spreadsheet as one tab-separated block.",
        details="The Copy box always mirrors the current measurement's results table exactly, header row included, so a paste into Excel/Sheets/a text file lines up as real columns without any reformatting.",
        target=("measurement_window", "widget", "copy_text"),
        completion_action=None,
    ),
]
"""The ordered v1 "Getting Started" step list, handed to
`tutorial_engine.Tutorial(STEPS)` by `tutorial_window.TutorialController`.
See this module's docstring for scope and the scientific-accuracy
sign-off note on the two metrics-explanation steps."""
