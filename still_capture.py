#!/usr/bin/env python3

"""Save encoded still frames from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save encoded still frames in highest possible resolution (default: 4032x3040 px)
  to .jpg at specified capture frequency (default: ~every second)
  -> stop recording early if free disk space drops below threshold
- optional arguments:
  '-min' set recording time in minutes (default: 2 [min])
         -> e.g. '-min 5' for 5 min recording time
  '-af'  set auto focus range in cm (min distance, max distance)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-zip' store all captured data in an uncompressed .zip file for each day
         and delete original directory
         -> increases file transfer speed from microSD to computer
            but also on-device processing time and power consumption

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import depthai as dai
import psutil

from utils.general import zip_data
from utils.oak_cam import set_focus_range

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="Set recording time in minutes (default: 2 [min]).", metavar="1-720")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min distance, max distance).", metavar=("CM_MIN", "CM_MAX"))
parser.add_argument("-zip", "--zip_data", action="store_true",
    help="Store data in an uncompressed .zip file for each day and delete original directory.")
args = parser.parse_args()

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set capture frequency (default: ~every second)
# -> wait for specified amount of seconds between saving still frames
# 'CAPTURE_FREQ = 1' saves ~54 still frames per minute to .jpg (12 MP)
CAPTURE_FREQ = 1

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Create directory per day and recording interval to save still frames
rec_start = datetime.now()
rec_start_format = rec_start.strftime("%Y-%m-%d_%H-%M-%S")
save_path = Path(f"insect-detect/stills/{rec_start.date()}/{rec_start_format}")
save_path.mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)      # OAK-1 (IMX378)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)     # OAK-1 Lite (IMX214)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_5312X6000) # OAK-1 MAX (IMX582)
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setNumFramesPool(2,2,2,2,2)  # decrease frame pool size to avoid memory issues
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure

# Create and configure video encoder node and define input + output
still_enc = pipeline.create(dai.node.VideoEncoder)
still_enc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
still_enc.setNumFramesPool(1)
cam_rgb.still.link(still_enc.input)

xout_still = pipeline.create(dai.node.XLinkOut)
xout_still.setStreamName("still")
still_enc.bitstream.link(xout_still.input)

# Create script node
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

# Set script that will be run on OAK device to send capture still command
script.setScript('''
ctrl = CameraControl()
ctrl.setCaptureStill(True)
while True:
    node.io["capture_still"].send(ctrl)
''')

# Define script node output to send capture still command to color camera node
script.outputs["capture_still"].link(cam_rgb.inputControl)

if args.af_range:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    logging.info("Recording time: %s min\n", int(REC_TIME / 60))

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Create output queue to get the encoded still frames from the output defined above
    q_still = device.getOutputQueue(name="still", maxSize=1, blocking=False)

    if args.af_range:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

        # Set auto focus range to specified cm values
        af_ctrl = set_focus_range(args.af_range[0], args.af_range[1])
        q_ctrl.send(af_ctrl)

    # Set start time of recording
    start_time = time.monotonic()

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        # Get encoded still frames and save to .jpg
        if q_still.has():
            frame_still = q_still.get().getData()
            timestamp_still = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            with open(save_path / f"{timestamp_still}.jpg", "wb") as still_jpg:
                still_jpg.write(frame_still)

        # Wait for specified amount of seconds (default: 1)
        time.sleep(CAPTURE_FREQ)

# Print number and directory of saved still frames
num_frames_still = len(list(save_path.glob("*.jpg")))
logging.info("Saved %s still frames to %s\n", num_frames_still, save_path)

if args.zip_data:
    # Store frames in uncompressed .zip file and delete original folder
    zip_data(save_path)
    logging.info("Stored all captured images in %s.zip\n", save_path.parent)
