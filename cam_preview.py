#!/usr/bin/env python3

'''
Author:       Maximilian Sittinger (https://github.com/maxsitt)
Website:      https://maxsitt.github.io/insect-detect-docs/
License:      GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- show a preview of full FOV 4K frames downscaled to LQ frames (e.g. 416x416)

compiled with open source scripts available at https://github.com/luxonis
'''

import cv2
import depthai as dai

# Create depthai pipeline
pipeline = dai.Pipeline()

# Define camera source and output
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setPreviewSize(416, 416) # downscaled LQ frames
cam_rgb.setInterleaved(False)
cam_rgb.setPreviewKeepAspectRatio(False) # squash full FOV frames to square
cam_rgb.setFps(20) # frames per second available for focus/exposure

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
cam_rgb.preview.link(xout_rgb.input)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create output queue to get the frames from the output defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)

    # Get LQ preview frames and show in window (e.g. via X11 forwarding)
    while True:
        frame = q_frame.get().getCvFrame()
        cv2.imshow("cam_preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break
