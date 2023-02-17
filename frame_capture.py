#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save HQ frames (e.g. 3840x2160) to .jpg at specified time interval
- optional arguments:
  "-min [min]" (default = 2) set recording time in minutes
               (e.g. "-min 5" for 5 min recording time)
  "-lq" additionally save downscaled full FOV LQ frames (e.g. 320x320)

includes segments from open source scripts available at https://github.com/luxonis
'''

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

# Set capture frequency in seconds
# 'CAPTURE_FREQ = 0.2' saves ~60 frames per minute to .jpg (RPi Zero 2)
CAPTURE_FREQ = 0.2

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721),
                    default=2, help="set record time in minutes")
parser.add_argument("-lq", "--save_lq_frames", action="store_true",
    help="additionally save downscaled full FOV LQ frames (e.g. 320x320)")
args = parser.parse_args()

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure camera node and define outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setVideoSize(3840, 2160) # HQ frames, aspect ratio 16:9 (4K)
cam_rgb.setFps(40) # frames per second available for focus/exposure
if args.save_lq_frames:
    cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames
    cam_rgb.setInterleaved(False)
    cam_rgb.setPreviewKeepAspectRatio(False) # squash full FOV frames to square

xout_hq = pipeline.create(dai.node.XLinkOut)
xout_hq.setStreamName("hq_frame")
cam_rgb.video.link(xout_hq.input)

if args.save_lq_frames:
    xout_lq = pipeline.create(dai.node.XLinkOut)
    xout_lq.setStreamName("lq_frame")
    cam_rgb.preview.link(xout_lq.input)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create output queues to get the frames from the outputs defined above
    q_hq_frame = device.getOutputQueue(name="hq_frame", maxSize=4, blocking=False)
    if args.save_lq_frames:
        q_lq_frame = device.getOutputQueue(name="lq_frame", maxSize=4, blocking=False)

    # Create folders to save the frames
    rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
    save_path = f"./insect-detect/frames/{rec_start[:8]}/{rec_start}"
    Path(f"{save_path}/HQ_frames").mkdir(parents=True, exist_ok=True)
    if args.save_lq_frames:
        Path(f"{save_path}/LQ_frames").mkdir(parents=True, exist_ok=True)

    # Set recording start time
    start_time = time.monotonic()

    # Get recording time in min from optional argument (default: 2)
    rec_time = args.min_rec_time * 60
    print(f"Recording time: {args.min_rec_time} min")

    # Record until recording time is finished
    while time.monotonic() < start_time + rec_time:

        # Get HQ (+ LQ) frames and save to .jpg at specified time interval
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
        hq_path = f"{save_path}/HQ_frames/{timestamp}_HQ.jpg"
        hq_frame = q_hq_frame.get().getCvFrame()
        cv2.imwrite(hq_path, hq_frame)

        if args.save_lq_frames:
            lq_path = f"{save_path}/LQ_frames/{timestamp}_LQ.jpg"
            lq_frame = q_lq_frame.get().getCvFrame()
            cv2.imwrite(lq_path, lq_frame)

        time.sleep(CAPTURE_FREQ)

# Print number and path of saved frames to console
frames_hq = len(list(Path(f"{save_path}/HQ_frames").glob("*.jpg")))
if args.save_lq_frames:
    frames_lq = len(list(Path(f"{save_path}/LQ_frames").glob("*.jpg")))
    print(f"Saved {frames_hq} HQ and {frames_lq} LQ frames to {save_path}.")
else:
    print(f"Saved {frames_hq} HQ frames to {save_path}.")
