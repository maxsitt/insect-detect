#!/usr/bin/env python3

"""Save encoded still frames from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save encoded still frames in highest possible resolution (default: 4032x3040 px)
  to .jpg at the specified capture frequency (default: 1 s)
  -> stop recording early if free disk space drops below threshold
- optional arguments:
  '-min' set recording time in minutes (default: 2 [min])
         -> e.g. '-min 5' for 5 min recording time
  '-af'  set auto focus range in cm (min - max distance to camera)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-mf'  set manual focus position in cm (distance to camera)
         -> e.g. '-mf 14' to set manual focus position to 14 cm

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import depthai as dai
import psutil

from utils.oak_cam import convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="Set recording time in minutes (default: 2 [min]).", metavar="1-720")
group.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min - max distance to camera).", metavar=("CM_MIN", "CM_MAX"))
group.add_argument("-mf", "--manual_focus", type=int,
    help="Set manual focus position in cm (distance to camera).", metavar="CM")
args = parser.parse_args()

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set capture frequency (default: 1 second)
# -> wait for specified amount of seconds between saving still frames
# 'CAPTURE_FREQ = 1' saves ~58 still frames per minute to .jpg (12 MP)
CAPTURE_FREQ = 1

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Create directory per day (date) and recording interval (date_time) to save still frames
rec_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = Path.home() / "insect-detect" / "stills" / rec_start_str[:10] / rec_start_str
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

if args.af_range:
    # Convert cm to lens position values and set auto focus range
    lens_pos_min, lens_pos_max = convert_cm_lens_position((args.af_range[1], args.af_range[0]))
    cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)

if args.manual_focus:
    # Convert cm to lens position value and set manual focus position
    lens_pos = convert_cm_lens_position(args.manual_focus)
    cam_rgb.initialControl.setManualFocus(lens_pos)

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

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    logging.info("Recording time: %s min\n", int(REC_TIME / 60))

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Create output queue to get the encoded still frames from the output defined above
    q_still = device.getOutputQueue(name="still", maxSize=1, blocking=False)

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
