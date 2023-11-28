#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save encoded HQ frames (1080p or 4K resolution) with H.265 (HEVC) compression to .mp4 video file
- optional arguments:
  "-min [min]" (default = 2) set recording time in minutes
               -> e.g. "-min 5" for 5 min recording time
  "-4k" (default = 1080p) record video in 4K resolution (3840x2160 px)
  "-fps [fps]" (default = 25) set frame rate (frames per second) for video capture
               -> e.g. "-fps 20" to decrease video file size

based on open source scripts available at https://github.com/luxonis
'''

import argparse
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import depthai as dai
import psutil

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 61), default=2,
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
rec_start = datetime.now().strftime("%Y%m%d")
save_path = f"insect-detect/videos/{rec_start}"
Path(f"{save_path}").mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2) # downscale 4K to 1080p HQ frames (1920x1080 px)
cam_rgb.setFps(FPS) # frames per second available for focus/exposure

# Create and configure video encoder node and define input + output
video_enc = pipeline.create(dai.node.VideoEncoder)
video_enc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(video_enc.input)

xout_vid = pipeline.create(dai.node.XLinkOut)
xout_vid.setStreamName("video")
video_enc.bitstream.link(xout_vid.input)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queue to get the encoded frames from the output defined above
    q_video = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    # Create .mp4 container with H.265 (HEVC) compression
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    with av.open(f"{save_path}/{timestamp}_{FPS}fps_{RES}_video.mp4", "w") as container:
        stream = container.add_stream("hevc", rate=FPS)
        stream.time_base = Fraction(1, 1000 * 1000)
        if args.four_k_resolution:
            stream.width = 3840
            stream.height = 2160
        else:
            stream.width = 1920
            stream.height = 1080

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Create start_time variable to set recording time
    start_time = time.monotonic()

    # Record until recording time is finished or free disk space drops below threshold
    while time.monotonic() < start_time + rec_time and disk_free > 200:

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        # Get encoded video frames and save to packet
        enc_video = q_video.get().getData()
        packet = av.Packet(enc_video)
        packet.dts = int((time.monotonic() - start_time) * 1000 * 1000)
        packet.pts = int((time.monotonic() - start_time) * 1000 * 1000)

        # Mux packet into the .mp4 container
        container.mux_one(packet)

# Print duration, resolution, fps and path of saved video + free disk space to console
print(f"\nSaved {args.min_rec_time} min {RES} video with {FPS} fps to {save_path}.")
print(f"\nFree disk space left: {disk_free} MB")
