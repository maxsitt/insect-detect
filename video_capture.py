#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save encoded HQ frames (1080p or 4K resolution) with H.265 (HEVC) compression to .mp4 video file
- optional arguments:
  "-min" set recording time in minutes (default: 2 min)
         -> e.g. "-min 5" for 5 min recording time
  "-4k"  record video in 4K resolution (3840x2160 px) (default: 1080p)
  "-fps" set frame rate (frames per second) for video capture (default: 25 fps)
         -> e.g. "-fps 20" for 20 fps (less fps = smaller video file size)
  "-af"  set auto focus range in cm (min distance, max distance)
         -> e.g. "-af 14 20" to restrict auto focus range to 14-20 cm

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
    help="set record time in minutes (default: 2 min)")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="record video in 4K resolution (default: 1080p)")
parser.add_argument("-fps", "--frames_per_second", type=int, choices=range(1, 31), default=25,
    help="set frame rate (frames per second) for video capture (default: 25 fps)")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="set auto focus range in cm (min distance, max distance)", metavar=("cm_min", "cm_max"))
args = parser.parse_args()

# Create folders for each day to save videos
rec_start = datetime.now().strftime("%Y%m%d")
save_path = Path(f"insect-detect/videos/{rec_start}")
save_path.mkdir(parents=True, exist_ok=True)

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Set video resolution
RES = "1080p" if not args.four_k_resolution else "4K"

# Set frame rate (default: 25 fps)
FPS = args.frames_per_second

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


def set_focus_range():
    """Convert closest cm values to lens position values and set auto focus range."""
    cm_lenspos_dict = {
        6: 250,
        8: 220,
        10: 190,
        12: 170,
        14: 160,
        16: 150,
        20: 140,
        25: 135,
        30: 130,
        40: 125,
        60: 120
    }

    closest_cm_min = min(cm_lenspos_dict.keys(), key=lambda k: abs(k - args.af_range[0]))
    closest_cm_max = min(cm_lenspos_dict.keys(), key=lambda k: abs(k - args.af_range[1]))
    lenspos_min = cm_lenspos_dict[closest_cm_max]
    lenspos_max = cm_lenspos_dict[closest_cm_min]

    af_ctrl = dai.CameraControl().setAutoFocusLensRange(lenspos_min, lenspos_max)
    q_ctrl.send(af_ctrl)


# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Print recording time to console (default: 2 minutes)
    print(f"\nRecording time: {int(REC_TIME / 60)} min\n")

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Create output queue to get the encoded frames from the output defined above
    q_video = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    if args.af_range:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

        # Set auto focus range to specified cm values
        set_focus_range()

    # Create .mp4 container with H.265 (HEVC) compression
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    with av.open(f"{save_path}/{timestamp}_{FPS}fps_{RES}_video.mp4", "w") as container:
        stream = container.add_stream("hevc", rate=FPS)
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

# Print duration, resolution, fps and path of saved video + free disk space to console
print(f"\nSaved {int(REC_TIME / 60)} min {RES} video with {FPS} fps to {save_path}.")
print(f"\nFree disk space left: {disk_free} MB")
