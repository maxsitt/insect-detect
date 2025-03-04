"""Show OAK camera livestream.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- show downscaled LQ frames + FPS in a new window (e.g. via X11 forwarding)
- optional arguments:
  '-fov' default:  stretch frames to square for visualization ('-fov stretch')
                   -> FOV is preserved, only aspect ratio of LQ frames is changed (adds distortion)
         optional: crop frames to square for visualization ('-fov crop')
                   -> FOV is reduced due to cropping of LQ frames (no distortion)
  '-af'  set auto focus range in cm (min - max distance to camera)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-mf'  set manual focus position in cm (distance to camera)
         -> e.g. '-mf 14' to set manual focus position to 14 cm
  '-big' show a bigger preview window with 640x640 px size (default: 320x320 px)

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import time

import cv2
import depthai as dai

from utils.oak import convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-fov", "--field_of_view", type=str, choices=["stretch", "crop"], default="stretch",
    help=("Stretch frames to square and preserve FOV ('stretch') or "
          "crop frames to square and reduce FOV ('crop') (default: 'stretch')."))
group.add_argument("-af", "--auto_focus_range", type=int, nargs=2, metavar=("CM_MIN", "CM_MAX"),
    help="Set auto focus range in cm (min - max distance to camera).")
group.add_argument("-mf", "--manual_focus", type=int, metavar="CM",
    help="Set manual focus position in cm (distance to camera).")
parser.add_argument("-big", "--big_preview", action="store_true",
    help="Show a bigger preview window with 640x640 px size (default: 320x320 px).")
args = parser.parse_args()

# Set camera frame rate
FPS = 20  # default: 20 FPS

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node and define output
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
if not args.big_preview:
    cam_rgb.setPreviewSize(320, 320)  # downscale frames -> LQ frames
else:
    cam_rgb.setPreviewSize(640, 640)
if args.field_of_view == "stretch":
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
img_width, img_height = cam_rgb.getPreviewSize()

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
cam_rgb.preview.link(xout_rgb.input)

if args.auto_focus_range:
    # Convert cm to lens position values and set auto focus range
    lens_pos_min = convert_cm_lens_position(args.auto_focus_range[1])
    lens_pos_max = convert_cm_lens_position(args.auto_focus_range[0])
    cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
elif args.manual_focus:
    # Convert cm to lens position value and set manual focus position
    lens_pos = convert_cm_lens_position(args.manual_focus)
    cam_rgb.initialControl.setManualFocus(lens_pos)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queue to get the LQ frames
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)

    # Set start time of recording and create counter to measure FPS
    start_time = time.monotonic()
    counter = 0

    while True:
        if q_frame.has():
            # Get LQ frame and show in new window together with FPS
            frame_lq = q_frame.get().getCvFrame()

            counter += 1
            fps = round(counter / (time.monotonic() - start_time), 2)

            cv2.putText(frame_lq, f"FPS: {fps}", (4, img_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("cam_preview", frame_lq)

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
