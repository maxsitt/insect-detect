#!/usr/bin/env python3

'''
Author:       Maximilian Sittinger (https://github.com/maxsitt)
Website:      https://maxsitt.github.io/insect-detect-docs/
License:      GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save still images in highest possible resolution to .jpg at specified time interval
- optional argument:
  "-min [min]" (default = 2) set recording time in minutes
               (e.g. "-min 5" for 5 min recording time)

includes segments from open source scripts available at https://github.com/luxonis
'''

import argparse
import time
from datetime import datetime
from pathlib import Path

import depthai as dai

# Set capture frequency in seconds (save still to .jpg every XX seconds)
CAPTURE_FREQ = 2

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 720),
                    default=2, help="set record time in minutes")
args = parser.parse_args()

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) # OAK-1 (IMX378)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP) # OAK-1 Lite (IMX214)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_5312X6000) # OAK-1 MAX (LCM48)
cam_rgb.setNumFramesPool(2,2,2,2,2)
cam_rgb.setFps(10) # frames per second available for focus/exposure

# Define camera control input
xin_ctrl = pipeline.create(dai.node.XLinkIn)
xin_ctrl.setStreamName("control")
xin_ctrl.out.link(cam_rgb.inputControl)

# Create and configure video encoder node and define input + output
still_enc = pipeline.create(dai.node.VideoEncoder)
still_enc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
still_enc.setNumFramesPool(1)
cam_rgb.still.link(still_enc.input)

xout_still = pipeline.create(dai.node.XLinkOut)
xout_still.setStreamName("still")
still_enc.bitstream.link(xout_still.input)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create input queue to send the still capture command to the OAK device
    q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False)

    # Create output queue to get the encoded still images
    q_still = device.getOutputQueue(name="still", maxSize=1, blocking=False)

    # Set recording start time and create folder to save the still images
    rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
    save_path = f"./insect-detect/stills/{rec_start[:8]}/{rec_start}"
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()

    # Get recording time in min from optional argument (default: 2)
    rec_time = args.min_rec_time * 60
    print(f"Recording time: {args.min_rec_time} min")

    # Record until recording time is finished
    while time.monotonic() < start_time + rec_time:

        # Send still capture command to OAK device
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        q_ctrl.send(ctrl)

        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")

        # Get encoded still frames and save to .jpg at specified time interval
        enc_still = q_still.get()
        with open(f"{save_path}/{timestamp}_still.jpg", "wb") as still_jpg:
            still_jpg.write(enc_still.getData())

        time.sleep(CAPTURE_FREQ)

# Print number and path of saved still frames to console
frames_still = len(list(Path(f"{save_path}").glob("*.jpg")))
print(f"Saved {frames_still} still frames to {save_path}.")
