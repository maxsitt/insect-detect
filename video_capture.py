#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save encoded HQ frames (1080p or 4K resolution) with HEVC/H.265 compression to .mp4 video file
- optional arguments:
  "-min [min]" (default = 2) set recording time in minutes
               (e.g. "-min 5" for 5 min recording time)
  "-fps [fps]" (default = 30) set frame rate (frames per second) for video capture
  "-4k" (default = 1080p) record video in 4K resolution (3840x2160 px)

includes segments from open source scripts available at https://github.com/luxonis
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
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 61),
                    default=2, help="set record time in minutes")
parser.add_argument("-fps", "--frames_per_second", type=int, choices=range(1, 31),
                    default=30, help="set frame rate (frames per second) for video capture")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
                    help="record video in 4K resolution (3840x2160 px); default = 1080p")
args = parser.parse_args()

# Get frame rate (frames per second) from optional argument (default: 30)
FPS = args.frames_per_second

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setVideoSize(1920, 1080)
if args.four_k_resolution:
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setVideoSize(3840, 2160)
cam_rgb.setFps(FPS) # frames per second available for focus/exposure

# Create and configure video encoder node and define input + output
video_enc = pipeline.create(dai.node.VideoEncoder)
video_enc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(video_enc.input)

xout_vid = pipeline.create(dai.node.XLinkOut)
xout_vid.setStreamName("video")
video_enc.bitstream.link(xout_vid.input)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create output queue to get the encoded frames from the output defined above
    q_video = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    # Create folder to save the videos
    rec_start = datetime.now().strftime("%Y%m%d")
    save_path = f"./insect-detect/videos/{rec_start}"
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)

    # Create .mp4 container with HEVC/H.265 compression
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    RES = "1080p"
    if args.four_k_resolution:
        RES = "4K"
    with av.open(f"{save_path}/{timestamp}_{FPS}fps_{RES}_video.mp4", "wb") as container:
        stream = container.add_stream("hevc", rate=FPS)
        stream.time_base = Fraction(1, 1000 * 1000)
        stream.width = 1920
        stream.height = 1080
        if args.four_k_resolution:
            stream.width = 3840
            stream.height = 2160

    # Set recording start time
    start_time = time.monotonic()

    # Get recording time in min from optional argument (default: 2)
    rec_time = args.min_rec_time * 60
    print(f"Recording time: {args.min_rec_time} min\n")

    # Get free disk space in MB
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Record until recording time is finished or free disk space drops below threshold
    while time.monotonic() < start_time + rec_time and disk_free > 200:

        # Update free disk space
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        # Get encoded video frames and save to packet
        enc_video = q_video.get().getData()
        packet = av.Packet(enc_video)
        packet.dts = int((time.monotonic() - start_time) * 1000 * 1000)
        packet.pts = int((time.monotonic() - start_time) * 1000 * 1000)

        # Mux packet into the .mp4 container
        container.mux_one(packet)

# Print duration, fps and path of saved video + free disk space to console
if not args.four_k_resolution:
    print(f"\nSaved {args.min_rec_time} min 1080p video with {args.frames_per_second} fps to {save_path}.")
if args.four_k_resolution:
    print(f"\nSaved {args.min_rec_time} min 4K video with {args.frames_per_second} fps to {save_path}.")
print(f"Free disk space left: {disk_free} MB")
