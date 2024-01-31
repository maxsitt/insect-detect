#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- show downscaled LQ frames + fps in a new window (e.g. via X11 forwarding)
- optional argument:
  "-big" show a bigger (640x640 px) preview window (default: 320x320 px)
         -> decreases frame rate to ~3 fps (default: ~11 fps)

based on open source scripts available at https://github.com/luxonis
'''

import argparse
import time

import cv2
import depthai as dai

# Define optional argument
parser = argparse.ArgumentParser()
parser.add_argument("-big", "--big_preview", action="store_true",
    help="show a bigger (640x640 px) preview window; default = 320x320 px")
args = parser.parse_args()

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node and define output
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
if not args.big_preview:
    cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames
else:
    cam_rgb.setPreviewSize(640, 640)
cam_rgb.setPreviewKeepAspectRatio(False) # stretch frames (16:9) to square (1:1)
cam_rgb.setInterleaved(False) # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25) # frames per second available for auto focus/exposure

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
cam_rgb.preview.link(xout_rgb.input)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queue to get the frames from the output defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)

    # Create start_time and counter variable to measure fps
    start_time = time.monotonic()
    counter = 0

    # Get LQ frames and show in new window together with fps
    while True:
        if q_frame.has():
            frame = q_frame.get().getCvFrame()

            counter += 1
            fps = round(counter / (time.monotonic() - start_time), 2)

            cv2.putText(frame, f"fps: {fps}", (4, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("cam_preview", frame)

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
