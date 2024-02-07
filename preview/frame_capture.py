#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save HQ frames (e.g. 1920x1080 or 3840x2160 px) to .jpg at specified time interval
- optional arguments:
  "-min [min]" (default = 2) set recording time in minutes
               -> e.g. "-min 5" for 5 min recording time
  "-4k" (default = 1080p) save HQ frames in 4K resolution (3840x2160 px)
  "-lq" additionally save downscaled LQ frames (e.g. 320x320 px)

based on open source scripts available at https://github.com/luxonis
'''

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="set record time in minutes")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="save HQ frames in 4K resolution; default = 1080p")
parser.add_argument("-lq", "--save_lq_frames", action="store_true",
    help="additionally save downscaled LQ frames")
args = parser.parse_args()

# Set capture frequency in seconds
# 'CAPTURE_FREQ = 0.8' (0.2 for 4K) saves ~58 frames per minute to .jpg (RPi Zero 2)
CAPTURE_FREQ = 0.8
if args.four_k_resolution:
    CAPTURE_FREQ = 0.2

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure camera node and define output(s)
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2) # downscale 4K to 1080p HQ frames (1920x1080 px)
cam_rgb.setFps(25) # frames per second available for focus/exposure
if args.save_lq_frames:
    cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames
    cam_rgb.setPreviewKeepAspectRatio(False) # "squeeze" frames (16:9) to square (1:1)
    cam_rgb.setInterleaved(False) # planar layout

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
cam_rgb.video.link(xout_rgb.input) # HQ frame

if args.save_lq_frames:
    xout_lq = pipeline.create(dai.node.XLinkOut)
    xout_lq.setStreamName("frame_lq")
    cam_rgb.preview.link(xout_lq.input) # LQ frame

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queue(s) to get the frames from the output(s) defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    if args.save_lq_frames:
        q_frame_lq = device.getOutputQueue(name="frame_lq", maxSize=4, blocking=False)

    # Create folders to save the frames
    rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
    save_path = f"insect-detect/frames/{rec_start[:8]}/{rec_start}"
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)
    if args.save_lq_frames:
        Path(f"{save_path}/LQ_frames").mkdir(parents=True, exist_ok=True)

    # Create start_time variable to set recording time
    start_time = time.monotonic()

    # Get recording time in min from optional argument (default: 2)
    rec_time = args.min_rec_time * 60
    print(f"Recording time: {args.min_rec_time} min")

    # Record until recording time is finished
    while time.monotonic() < start_time + rec_time:

        # Get HQ (+ LQ) frames and save to .jpg at specified time interval
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
        hq_path = f"{save_path}/{timestamp}.jpg"
        hq_frame = q_frame.get().getCvFrame()
        cv2.imwrite(hq_path, hq_frame)

        if args.save_lq_frames:
            lq_path = f"{save_path}/LQ_frames/{timestamp}_LQ.jpg"
            lq_frame = q_frame_lq.get().getCvFrame()
            cv2.imwrite(lq_path, lq_frame)

        time.sleep(CAPTURE_FREQ)

# Print number and path of saved frames to console
frames_hq = len(list(Path(f"{save_path}").glob("*.jpg")))
if args.save_lq_frames:
    frames_lq = len(list(Path(f"{save_path}/LQ_frames").glob("*.jpg")))
    print(f"Saved {frames_hq} HQ and {frames_lq} LQ frames to {save_path}.")
else:
    print(f"Saved {frames_hq} HQ frames to {save_path}.")
