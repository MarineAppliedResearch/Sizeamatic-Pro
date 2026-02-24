# Sizeamatic Pro

Sizeamatic Pro is a Windows desktop tool for measuring real world distances from stereo video. Load your synchronized left and right recordings, scrub to any moment, click the same feature in both views, and Sizeamatic Pro reports depth and size in millimeters.

Built to run fully offline with your data stored locally.

---

## What you can do with it

- **Measure range to a feature**
  - Click a point in the left view, then the matching point in the right view.
  - Sizeamatic Pro computes distance from the camera in millimeters.

- **Measure the size of an object**
  - Place two matching points per view to define a line segment.
  - Sizeamatic Pro computes the 3D length and shows endpoint depths.

- **Inspect in raw or rectified view**
  - Switch between raw video and rectified video.
  - Rectified view aligns the stereo pair so matching features lie on the same horizontal line, making accurate point matching easier.

- **Navigate precisely**
  - Play and pause
  - Step frame by frame
  - Fast forward and rewind
  - Scrub using the timeline sliders
  - Lock the two videos together for synchronized navigation, or scrub them independently when needed

---

## What you need

- A Windows PC
- Two synchronized stereo recordings:
  - `left.mp4`
  - `right.mp4`
- A completed stereo calibration set exported from OpenCV, provided as NPZ files in one folder:
  - `calibration_intrinsics.npz`
  - `calibration_extrinsics.npz`
  - `calibration_rectification.npz`
  - `calibration_maps.npz`

If your calibration was performed with a known checkerboard square size and the calibration translation is in millimeters, Sizeamatic Pro reports measurements in millimeters.

---

## Typical workflow

1. Open Sizeamatic Pro
2. Load the left video and right video
3. Load the calibration folder
4. Enable **Show Rectified** for easier matching
5. Scrub to the frame you want
6. Click matching points to measure depth, or add a second point pair to measure length

---

## Measurement notes

- Measurements are computed in the rectified left camera coordinate frame.
- **Depth (Z)** is the distance forward along the left camera optical axis.
- Results are only valid when the selected points are true matches between left and right views.
- If the calibration resolution does not match the video resolution, Sizeamatic Pro will warn you and disable rectified measurement until corrected.

---

## Privacy and offline operation

Sizeamatic Pro does not upload or transmit video or calibration data. All files remain on your local machine.

---

## Support and feedback

If you encounter a calibration mismatch, decode issue, or measurement instability, capture:
- the left and right video file properties
- the calibration folder contents
- a screenshot of the Sizeamatic Pro window showing the issue

and include them in your support request.