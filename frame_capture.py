#!/usr/bin/env python3

"""Save frames from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save HQ frames (default: 1920x1080 px) to .jpg at the
  specified capture frequency (default: 1 s)
  -> stop recording early if free disk space drops below threshold
- optional arguments:
  '-min' set recording time in minutes (default: 2 [min])
         -> e.g. '-min 5' for 5 min recording time
  '-4k'  set camera resolution to 4K (3840x2160 px) (default: 1080p)
  '-lq'  additionally save downscaled LQ frames (e.g. 320x320 px)
  '-af'  set auto focus range in cm (min - max distance to camera)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import psutil

from utils.oak_cam import convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="Set recording time in minutes (default: 2 [min]).", metavar="1-720")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="Set camera resolution to 4K (3840x2160 px) (default: 1080p).")
parser.add_argument("-lq", "--save_lq_frames", action="store_true",
    help="Additionally save downscaled LQ frames (320x320 px).")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min - max distance to camera).", metavar=("CM_MIN", "CM_MAX"))
args = parser.parse_args()

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set capture frequency (default: 1 second)
# -> wait for specified amount of seconds between saving HQ frames
# 'CAPTURE_FREQ = 0.8' (0.2 for 4K) saves ~58 frames per minute to .jpg
CAPTURE_FREQ = 0.8 if not args.four_k_resolution else 0.2

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Create directory per day (date) and recording interval (date_time) to save HQ frames (+ LQ frames)
rec_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = Path.home() / "insect-detect" / "frames" / rec_start_str[:10] / rec_start_str
save_path.mkdir(parents=True, exist_ok=True)
if args.save_lq_frames:
    (save_path / "LQ_frames").mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node and define output(s)
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2)  # downscale 4K to 1080p resolution -> HQ frames
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure
if args.save_lq_frames:
    cam_rgb.setPreviewSize(320, 320)  # downscale frames -> LQ frames
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch frames (16:9) to square (1:1)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
cam_rgb.video.link(xout_rgb.input)  # HQ frames

if args.save_lq_frames:
    xout_lq = pipeline.create(dai.node.XLinkOut)
    xout_lq.setStreamName("frame_lq")
    cam_rgb.preview.link(xout_lq.input)  # LQ frames

if args.af_range:
    # Convert cm to lens position values and set auto focus range
    lens_pos_min, lens_pos_max = convert_cm_lens_position((args.af_range[1], args.af_range[0]))
    cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    logging.info("Recording time: %s min\n", int(REC_TIME / 60))

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Create output queue(s) to get the frames from the output(s) defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    if args.save_lq_frames:
        q_frame_lq = device.getOutputQueue(name="frame_lq", maxSize=4, blocking=False)

    # Set start time of recording
    start_time = time.monotonic()

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        # Get HQ (+ LQ) frames and save to .jpg
        if q_frame.has():
            frame_hq = q_frame.get().getCvFrame()
            timestamp_frame = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            path_hq = f"{save_path}/{timestamp_frame}.jpg"
            cv2.imwrite(path_hq, frame_hq)

        if args.save_lq_frames:
            if q_frame_lq.has():
                frame_lq = q_frame_lq.get().getCvFrame()
                path_lq = f"{save_path}/LQ_frames/{timestamp_frame}_LQ.jpg"
                cv2.imwrite(path_lq, frame_lq)

        # Wait for specified amount of seconds (default: 0.8 for 1080p; 0.2 for 4K)
        time.sleep(CAPTURE_FREQ)

# Print number and directory of saved frames
num_frames_hq = len(list(save_path.glob("*.jpg")))
if not args.save_lq_frames:
    logging.info("Saved %s HQ frames to %s\n", num_frames_hq, save_path)
else:
    num_frames_lq = len(list((save_path / "LQ_frames").glob("*.jpg")))
    logging.info("Saved %s HQ and %s LQ frames to %s\n", num_frames_hq, num_frames_lq, save_path)
