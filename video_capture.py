"""Save video from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save encoded HQ frames (1080p or 4K resolution) with H.265 (HEVC) compression to .mp4 video file
- optional arguments:
  '-rec' set recording time in minutes (default: 2)
         -> e.g. '-rec 5' for 5 min recording time
  '-res' set camera resolution for HQ frames
         default:  4K resolution    -> 3840x2160 px, cropped from 12MP  ('-res 4k')
         optional: 1080p resolution -> 1920x1080 px, downscaled from 4K ('-res 1080p')
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
from fractions import Fraction
from pathlib import Path

import av
import depthai as dai
import psutil

from utils.oak import convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-rec", "--recording_time", type=int, default=2, metavar="MINUTES",
    help="Set recording time in minutes (default: 2).")
parser.add_argument("-res", "--resolution", type=str, choices=["4k", "1080p"], default="4k",
    help="Set camera resolution (default: 4k).")
group.add_argument("-af", "--auto_focus_range", type=int, nargs=2, metavar=("CM_MIN", "CM_MAX"),
    help="Set auto focus range in cm (min - max distance to camera).")
group.add_argument("-mf", "--manual_focus", type=int, metavar="CM",
    help="Set manual focus position in cm (distance to camera).")
args = parser.parse_args()

# Set camera frame rate
FPS = 20  # default: 20 FPS (maximum: 30 FPS)

# Set minimum free disk space threshold required to start and continue a recording
MIN_DISKSPACE = 1000  # default: 1000 MB

# Set time interval at which free disk space is checked during recording
FREE_SPACE_INT = 30  # default: 30 seconds

# Set recording time
REC_TIME = args.recording_time * 60  # default: 2 minutes

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Get free disk space (MB)
disk_free = round(psutil.disk_usage("/").free / 1048576)

# Create directory per day (date) to save video
timestamp = datetime.now()
timestamp_dir = timestamp.strftime("%Y-%m-%d")
save_path = Path.home() / "insect-detect" / "videos" / timestamp_dir
save_path.mkdir(parents=True, exist_ok=True)
timestamp_video = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
filename_video = f"{timestamp_video}_{FPS}fps_{args.resolution}_video"

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if args.resolution == "1080p":
    cam_rgb.setIspScale(1, 2)  # downscale 4K to 1080p resolution -> HQ frames (16:9)
cam_rgb.setInterleaved(False)  # planar layout

if args.auto_focus_range:
    # Convert cm to lens position values and set auto focus range
    lens_pos_min = convert_cm_lens_position(args.auto_focus_range[1])
    lens_pos_max = convert_cm_lens_position(args.auto_focus_range[0])
    cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
elif args.manual_focus:
    # Convert cm to lens position value and set manual focus position
    lens_pos = convert_cm_lens_position(args.manual_focus)
    cam_rgb.initialControl.setManualFocus(lens_pos)

# Configure ISP settings (default: 1, range: 0-4)
# -> setting Sharpness and LumaDenoise to 0 can reduce artifacts in some cases
cam_rgb.initialControl.setSharpness(1)
cam_rgb.initialControl.setLumaDenoise(1)
cam_rgb.initialControl.setChromaDenoise(1)

# Create and configure video encoder node and define input + output
encoder = pipeline.create(dai.node.VideoEncoder)
encoder.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(encoder.input)  # HQ frames as encoder input

xout_vid = pipeline.create(dai.node.XLinkOut)
xout_vid.setStreamName("video")
encoder.bitstream.link(xout_vid.input)  # HQ frames (H.265-encoded bitstream)

# Connect to OAK device and start pipeline in USB2 mode
with (dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device,
      av.open(f"{save_path}/{filename_video}.mp4", "w") as container):

    # Create output queue to get the encoded HQ frames
    q_video = device.getOutputQueue(name="video", maxSize=30, blocking=True)

    # Add stream with H.265 (HEVC) compression to .mp4 container
    stream = container.add_stream("hevc", rate=FPS)
    stream.width, stream.height = cam_rgb.getVideoSize()
    stream.time_base = Fraction(1, 1000 * 1000)
    stream.options = {"x265-params": "log_level=none"}  # disable x265 log output

    # Print info on start of recording
    logging.info("Rec time: %s min", int(REC_TIME / 60))

    # Create variables for start of recording and check events
    start_time = time.monotonic()
    last_disk_check = start_time

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Get current time
        current_time = time.monotonic()

        if q_video.has():
            # Get H.265-encoded HQ frames and save to packet
            frame_video = q_video.get().getData()
            packet = av.Packet(frame_video)
            packet_timestamp = int((time.monotonic() - start_time) * 1000 * 1000)
            packet.pts = packet.dts = packet_timestamp

            # Mux packet into .mp4 container
            container.mux_one(packet)

            # Update free disk space (MB) at specified interval
            if current_time - last_disk_check >= FREE_SPACE_INT:
                disk_free = round(psutil.disk_usage("/").free / 1048576)
                last_disk_check = current_time

# Print duration, resolution, fps and directory of saved video
logging.info("Saved %s min %s video with %s fps to %s", int(REC_TIME / 60), args.resolution, FPS, save_path)
