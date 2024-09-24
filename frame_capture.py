#!/usr/bin/env python3

"""Save frames from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save HQ frames to .jpg at the specified capture frequency (default: 1 second)
  -> stop recording early if free disk space drops below threshold (default: 100 MB)
- optional arguments:
  '-min' set recording time in minutes (default: 40)
         -> e.g. '-min 5' for 5 min recording time
  '-res' set camera resolution for HQ frames
         -> '-res 1080p' for 1920x1080 px
         -> '-res 4k' for 3840x2160 px (default - cropped from 12MP)
         -> '-res 12mp' for 4032x3040 px
  '-af'  set auto focus range in cm (min - max distance to camera)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-mf'  set manual focus position in cm (distance to camera)
         -> e.g. '-mf 14' to set manual focus position to 14 cm
  '-lq'  additionally save downscaled + stretched LQ frames (default: 320x320 px)
         -> FOV from 4K/1080p resolution, full FOV from 12MP is not preserved

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
group = parser.add_mutually_exclusive_group()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=40,
    help="Set recording time in minutes (default: 40).", metavar="1-720")
parser.add_argument("-res", "--resolution", choices=["1080p", "4k", "12mp"], default="4k", type=str,
    help="Set camera resolution (default: 4k).")
group.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min - max distance to camera).", metavar=("CM_MIN", "CM_MAX"))
group.add_argument("-mf", "--manual_focus", type=int,
    help="Set manual focus position in cm (distance to camera).", metavar="CM")
parser.add_argument("-lq", "--save_lq_frames", action="store_true",
    help="Additionally save downscaled LQ frames.")
args = parser.parse_args()

# Set minimum free disk space required to start and continue a recording (default: 100 MB)
MIN_DISKSPACE = 100

# Set capture frequency (default: 1 second)
# -> wait for specified amount of seconds between saving HQ frames
# 'CAPTURE_FREQ = 0' tries to save HQ frames at the maximum possible frame rate
CAPTURE_FREQ = 1

# Set camera frame rate (default: 25 FPS)
FPS = 25

# Set recording time (default: 40 minutes)
REC_TIME = args.min_rec_time * 60

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Get free disk space (MB)
disk_free = round(psutil.disk_usage("/").free / 1048576)

# Create directory per day (date) and recording interval (date_time) to save HQ frames (+ LQ frames)
rec_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = Path.home() / "insect-detect" / "frames" / rec_start_str[:10] / rec_start_str
save_path.mkdir(parents=True, exist_ok=True)
if args.save_lq_frames:
    (save_path / "LQ_frames").mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
if args.resolution == "12mp":
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)  # 4032x3040 px
else:
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)    # 3840x2160 px
    if args.resolution == "1080p":
        cam_rgb.setIspScale(1, 2)  # downscale 4K to 1080p resolution            # 1920x1080 px
if args.save_lq_frames:
    cam_rgb.setPreviewSize(320, 320)  # downscale frames -> LQ frames
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch frames (16:9) to square (1:1)
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure

if args.save_lq_frames:
    # Create output node for LQ frames
    xout_lq = pipeline.create(dai.node.XLinkOut)
    xout_lq.setStreamName("frame_lq")
    cam_rgb.preview.link(xout_lq.input)

if args.af_range:
    # Convert cm to lens position values and set auto focus range
    lens_pos_min, lens_pos_max = convert_cm_lens_position((args.af_range[1], args.af_range[0]))
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
video_enc = pipeline.create(dai.node.VideoEncoder)
video_enc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
video_enc.setQuality(80)  # JPEG quality (0-100)
cam_rgb.still.link(video_enc.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
video_enc.bitstream.link(xout_rgb.input)  # HQ frames (MJPEG encoded bitstream)

# Create XLinkIn node to send control commands to color camera node
xin_ctrl = pipeline.create(dai.node.XLinkIn)
xin_ctrl.setStreamName("control")
xin_ctrl.out.link(cam_rgb.inputControl)

# Define control command to capture still frame
ctrl_capture = dai.CameraControl().setCaptureStill(True)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queue(s) to get the frames from the output(s) defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    if args.save_lq_frames:
        q_frame_lq = device.getOutputQueue(name="frame_lq", maxSize=4, blocking=False)

    # Create input queue to send control commands to OAK camera
    q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False)

    # Wait for 3 seconds to let camera adjust auto focus and exposure
    time.sleep(3)

    # Set start time of recording
    start_time = time.monotonic()
    logging.info("Recording time: %s min\n", int(REC_TIME / 60))

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Send control command to OAK camera to capture still frame
        q_ctrl.send(ctrl_capture)

        if q_frame.has():
            # Get MJPEG encoded HQ frame and save to .jpg
            timestamp_frame = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            frame_hq = q_frame.get().getData()
            with open(save_path / f"{timestamp_frame}.jpg", "wb") as jpg:
                jpg.write(frame_hq)

            if args.save_lq_frames and q_frame_lq.has():
                # Get LQ frame and save to .jpg
                frame_lq = q_frame_lq.get().getCvFrame()
                cv2.imwrite(f"{save_path}/LQ_frames/{timestamp_frame}_LQ.jpg", frame_lq)

            # Update free disk space (MB)
            disk_free = round(psutil.disk_usage("/").free / 1048576)

            # Wait for specified amount of seconds (default: 1)
            time.sleep(CAPTURE_FREQ)

# Print number and directory of saved frames
num_frames_hq = len(list(save_path.glob("*.jpg")))
if not args.save_lq_frames:
    logging.info("Saved %s HQ frames to %s\n", num_frames_hq, save_path)
else:
    num_frames_lq = len(list((save_path / "LQ_frames").glob("*.jpg")))
    logging.info("Saved %s HQ and %s LQ frames to %s\n", num_frames_hq, num_frames_lq, save_path)
