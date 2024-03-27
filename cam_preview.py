#!/usr/bin/env python3

"""Show OAK camera livestream.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- show downscaled LQ frames + fps in a new window (e.g. via X11 forwarding)
- optional arguments:
  '-af'  set auto focus range in cm (min distance, max distance)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-big' show a bigger preview window with 640x640 px size (default: 320x320 px)
         -> decreases frame rate to ~3 fps (default: ~11 fps)

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import time

import cv2
import depthai as dai

from utils.oak_cam import set_focus_range

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min distance, max distance).", metavar=("CM_MIN", "CM_MAX"))
parser.add_argument("-big", "--big_preview", action="store_true",
    help="Show a bigger preview window with 640x640 px size (default: 320x320 px).")
args = parser.parse_args()

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node and define output
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
if not args.big_preview:
    cam_rgb.setPreviewSize(320, 320)  # downscale frames -> LQ frames
else:
    cam_rgb.setPreviewSize(640, 640)
cam_rgb.setPreviewKeepAspectRatio(False)  # stretch frames (16:9) to square (1:1)
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
cam_rgb.preview.link(xout_rgb.input)

if args.af_range:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queue to get the frames from the output defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)

    if args.af_range:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

        # Set auto focus range to specified cm values
        af_ctrl = set_focus_range(args.af_range[0], args.af_range[1])
        q_ctrl.send(af_ctrl)

    # Set start time of recording and create counter to measure fps
    start_time = time.monotonic()
    counter = 0

    while True:
        # Get LQ frames and show in new window together with fps
        if q_frame.has():
            frame_lq = q_frame.get().getCvFrame()

            counter += 1
            fps = round(counter / (time.monotonic() - start_time), 2)

            cv2.putText(frame_lq, f"fps: {fps}", (4, frame_lq.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("cam_preview", frame_lq)

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
