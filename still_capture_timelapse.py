#!/usr/bin/env python3

'''
Author:       Maximilian Sittinger (https://github.com/maxsitt)
Website:      https://maxsitt.github.io/insect-detect-docs/
License:      GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save still images in highest possible resolution to .jpg at specified time interval

includes segments from open source scripts available at https://github.com/luxonis
'''

from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

# Create depthai pipeline
pipeline = dai.Pipeline()

# Define camera source
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) # OAK-1 (IMX378)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP) # OAK-1 Lite (IMX214)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_5312X6000) # OAK-1 MAX (LCM48)
cam_rgb.setNumFramesPool(2,2,2,2,2)
cam_rgb.setFps(10) # frames per second available for focus/exposure

# Define MJPEG encoder
still_enc = pipeline.create(dai.node.VideoEncoder)
still_enc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
still_enc.setNumFramesPool(1)

# Define script node
script = pipeline.create(dai.node.Script)

# Set script that will be run on-device (Luxonis OAK)
script.setScript('''
import time

ctrl = CameraControl()
ctrl.setCaptureStill(True)

while True:
    node.io["capture_still"].send(ctrl)
    time.sleep(3) # capture still image every 3 seconds
''')

# Send capture command to camera and still image to the MJPEG encoder
script.outputs["capture_still"].link(cam_rgb.inputControl)
cam_rgb.still.link(still_enc.input)

xout_still = pipeline.create(dai.node.XLinkOut)
xout_still.setStreamName("still")
still_enc.bitstream.link(xout_still.input)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create output queue to get the encoded still images
    q_still = device.getOutputQueue(name="still", maxSize=1, blocking=False)

    rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
    save_path = f"./insect-detect/still/{rec_start[:8]}/{rec_start}"
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)

    while True:
        enc_still = q_still.get()
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
        with open(f"{save_path}/{timestamp}_still.jpg", "wb") as still_jpg:
            still_jpg.write(bytearray(enc_still.getData()))
