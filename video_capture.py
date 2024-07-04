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
  '-af'  set auto focus range in cm (min distance, max distance)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import logging
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import depthai as dai
import psutil

from utils.oak_cam import set_focus_range

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 61), default=2,
    help="Set recording time in minutes (default: 2 [min]).", metavar="1-60")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="Set camera resolution to 4K (3840x2160 px) (default: 1080p).")
parser.add_argument("-fps", "--frames_per_second", type=int, choices=range(1, 31), default=25,
    help="Set camera frame rate (default: 25 fps).", metavar="1-30")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min distance, max distance).", metavar=("CM_MIN", "CM_MAX"))
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

# Create directory per day to save video
rec_start = datetime.now()
save_path = Path(f"insect-detect/videos/{rec_start.date()}")
save_path.mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2)  # downscale 4K to 1080p resolution -> HQ frames
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure

# Create and configure video encoder node and define input + output
video_enc = pipeline.create(dai.node.VideoEncoder)
video_enc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(video_enc.input)

xout_vid = pipeline.create(dai.node.XLinkOut)
xout_vid.setStreamName("video")
video_enc.bitstream.link(xout_vid.input)  # encoded HQ frames

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

    # Create output queue to get the encoded frames from the output defined above
    q_video = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    if args.af_range:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

        # Set auto focus range to specified cm values
        af_ctrl = set_focus_range(args.af_range[0], args.af_range[1])
        q_ctrl.send(af_ctrl)

    # Create .mp4 container with H.265 (HEVC) compression
    timestamp_video = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    with av.open(f"{save_path}/{timestamp_video}_{FPS}fps_{RES}_video.mp4", "w") as container:
        stream = container.add_stream("hevc", rate=FPS, options={"x265-params": "log_level=none"})
        stream.width, stream.height = cam_rgb.getVideoSize()
        stream.time_base = Fraction(1, 1000 * 1000)

    # Set start time of recording
    start_time = time.monotonic()

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        # Get encoded video frames and save to packet
        if q_video.has():
            enc_video = q_video.get().getData()
            packet = av.Packet(enc_video)
            packet_timestamp = int((time.monotonic() - start_time) * 1000 * 1000)
            packet.dts = packet_timestamp
            packet.pts = packet_timestamp

            # Mux packet into .mp4 container
            container.mux_one(packet)

# Print duration, resolution, fps and directory of saved video + free disk space
logging.info("Saved %s min %s video with %s fps to %s\n", int(REC_TIME / 60), RES, FPS, save_path)
logging.info("Free disk space left: %s MB\n", disk_free)
