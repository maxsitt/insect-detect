#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save encoded HQ frames (1080p or 4K resolution) with H.265 (HEVC) compression to .mp4 video file
- optional arguments:
  "-minutes [min]" (default = 2) set recording time in minutes
                   -> e.g. "-minutes 5" for 5 min recording time
  "-4k" (default = 1080p) record video in 4K resolution (3840x2160 px)
  "-fps [fps]" (default = 25) set frame rate (frames per second) for video capture
               -> e.g. "-fps 20" to decrease video file size

based on open source scripts available at https://github.com/luxonis/depthai/tree/main/depthai_sdk
'''

import argparse
import time
from datetime import datetime
from pathlib import Path

import psutil
from depthai_sdk import OakCamera, RecordType

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-minutes", "--min_rec_time", type=int, choices=range(1, 61), default=2,
    help="set record time in minutes")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="record video in 4K resolution (3840x2160 px); default = 1080p")
parser.add_argument("-fps", "--frames_per_second", type=int, choices=range(1, 31), default=25,
    help="set frame rate (frames per second) for video capture")
args = parser.parse_args()

# Get recording time in min from optional argument (default: 2)
rec_time = args.min_rec_time * 60
print(f"\nRecording time: {args.min_rec_time} min\n")

# Get frame rate (frames per second) from optional argument (default: 25)
FPS = args.frames_per_second

# Set video resolution
if args.four_k_resolution:
    RES = "4K"
else:
    RES = "1080p"

# Create folder to save the videos
rec_start = datetime.now().strftime("%Y%m%d_%H-%M-%S")
save_path = f"insect-detect/sdk/videos/{rec_start}"
Path(f"{save_path}").mkdir(parents=True, exist_ok=True)

# Get free disk space (MB)
disk_free = round(psutil.disk_usage("/").free / 1048576)

# Create start_time variable to set recording time
start_time = time.monotonic()

with OakCamera(usb_speed="usb2") as oak:
#with OakCamera(usb_speed="usb2", rotation=180) as oak:
    cam_rgb = oak.camera("RGB", resolution=RES, fps=FPS, encode="H265")

    oak.record(cam_rgb.out.encoded, save_path, RecordType.VIDEO)

    oak.start(blocking=False)

    while oak.running():
        if time.monotonic() - start_time > rec_time or disk_free < 200:
            break

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        oak.poll()

# Print duration, resolution, fps and path of saved video + free disk space to console
print(f"\nSaved {args.min_rec_time} min {RES} video with {FPS} fps to {save_path}.")
print(f"\nFree disk space left: {disk_free} MB")
