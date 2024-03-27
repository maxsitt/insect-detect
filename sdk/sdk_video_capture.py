#!/usr/bin/env python3

"""Save video from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save encoded HQ frames (1080p or 4K resolution) with H.265 (HEVC) compression to .mp4 video file
- optional arguments:
  '-min' set recording time in minutes (default: 2 [min])
         -> e.g. '-min 5' for 5 min recording time
  '-4k'  record video in 4K resolution (3840x2160 px) (default: 1080p)
  '-fps' set camera frame rate (default: 25 fps)
         -> e.g. '-fps 20' for 20 fps (less fps = smaller video file size)

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import psutil
from depthai_sdk import OakCamera, RecordType

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 61), default=2,
    help="Set recording time in minutes (default: 2 [min]).", metavar="1-60")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="Set camera resolution to 4K (3840x2160 px) (default: 1080p).")
parser.add_argument("-fps", "--frames_per_second", type=int, choices=range(1, 31), default=25,
    help="Set camera frame rate (default: 25 fps).", metavar="1-30")
args = parser.parse_args()

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Set video resolution
RES = "1080p" if not args.four_k_resolution else "4K"

# Set frame rate (default: 25 fps)
FPS = args.frames_per_second

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Create directory to save video
rec_start = datetime.now()
rec_start_format = rec_start.strftime("%Y-%m-%d_%H-%M-%S")
save_path = Path(f"insect-detect/sdk/videos/{rec_start.date()}/{rec_start_format}")
save_path.mkdir(parents=True, exist_ok=True)

with OakCamera(usb_speed="usb2") as oak:
#with OakCamera(usb_speed="usb2", rotation=180) as oak:  # rotate image 180°
    cam_rgb = oak.camera("RGB", resolution=RES, fps=FPS, encode="H265")

    oak.record(cam_rgb.out.encoded, save_path, RecordType.VIDEO)

    logging.info("\nRecording time: %s min\n", int(REC_TIME / 60))

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Set start time of recording
    start_time = time.monotonic()

    oak.start(blocking=False)

    while oak.running():
        # Record until recording time is finished
        # Stop recording early if free disk space drops below threshold
        if time.monotonic() - start_time > REC_TIME or disk_free < MIN_DISKSPACE:
            break

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        oak.poll()

# Print duration, resolution, fps and directory of saved video + free disk space
logging.info("\nSaved %s min %s video with %s fps to %s", int(REC_TIME / 60), RES, FPS, save_path)
logging.info("\nFree disk space left: %s MB", disk_free)
