#!/usr/bin/env python3

"""Save HQ frames from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- save MJPEG-encoded HQ frames to .jpg at the specified time lapse interval (default: 1 s)
  -> stop recording early if free disk space drops below threshold (default: 1000 MB)
- optional arguments:
  '-rec' set recording time in minutes (default: 60)
         -> e.g. '-rec 5' for 5 min recording time
  '-res' set camera resolution for HQ frames
         default:  4K resolution    -> 3840x2160 px, cropped from 12MP  ('-res 4k')
         optional: 1080p resolution -> 1920x1080 px, downscaled from 4K ('-res 1080p')
                   12MP resolution  -> 4032x3040 px, full FOV           ('-res 12mp')
  '-tli' set time lapse interval in seconds at which HQ frame is saved (default: 1)
         -> e.g. '-tli 60' for 1 min time lapse interval
         -> '-tli 0' to save frames at the highest possible frame rate
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import psutil

from utils.general import save_encoded_frame
from utils.oak_cam import convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-rec", "--recording_time", type=int, default=60, metavar="MINUTES",
    help="Set recording time in minutes (default: 60).")
parser.add_argument("-res", "--resolution", type=str, choices=["4k", "1080p", "12mp"], default="4k",
    help="Set camera resolution (default: 4k).")
parser.add_argument("-tli", "--timelapse_interval", type=float, default=1, metavar="SECONDS",
    help=("Set time interval in seconds at which HQ frame is saved (default: 1)."))
group.add_argument("-af", "--auto_focus_range", type=int, nargs=2, metavar=("CM_MIN", "CM_MAX"),
    help="Set auto focus range in cm (min - max distance to camera).")
group.add_argument("-mf", "--manual_focus", type=int, metavar="CM",
    help="Set manual focus position in cm (distance to camera).")
parser.add_argument("-lq", "--save_lq_frames", action="store_true",
    help="Additionally save downscaled LQ frames.")
args = parser.parse_args()

# Set camera frame rate
FPS = 20  # default: 20 FPS

# Set minimum free disk space threshold required to start and continue a recording
MIN_DISKSPACE = 1000  # default: 1000 MB

# Set time interval at which free disk space is checked during recording
FREE_SPACE_INT = 30  # default: 30 seconds

# Set time interval at which HQ frame is saved
TIMELAPSE_INT = args.timelapse_interval  # default: 1 second

# Set recording time
REC_TIME = args.recording_time * 60  # default: 60 minutes

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Get free disk space (MB)
disk_free = round(psutil.disk_usage("/").free / 1048576)

# Create directory per day (date) and recording interval (datetime) to save HQ frames (+ LQ frames)
timestamp_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = Path.home() / "insect-detect" / "frames" / timestamp_dir[:10] / timestamp_dir
save_path.mkdir(parents=True, exist_ok=True)
if args.save_lq_frames:
    (save_path / "LQ_frames").mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure
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

if args.save_lq_frames:
    # Create output node for LQ frames
    xout_lq = pipeline.create(dai.node.XLinkOut)
    xout_lq.setStreamName("frame_lq")
    cam_rgb.preview.link(xout_lq.input)

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
encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
encoder.setQuality(80)  # JPEG quality (0-100)
cam_rgb.still.link(encoder.input)  # still frames as encoder input

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
encoder.bitstream.link(xout_rgb.input)  # HQ frames (MJPEG-encoded bitstream)

# Create XLinkIn node to send control commands to color camera node
xin_ctrl = pipeline.create(dai.node.XLinkIn)
xin_ctrl.setStreamName("control")
xin_ctrl.out.link(cam_rgb.inputControl)

# Define control command to capture still frame
ctrl_capture = dai.CameraControl().setCaptureStill(True)

# Connect to OAK device and start pipeline in USB2 mode
with (dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device,
      ThreadPoolExecutor(max_workers=3) as executor):

    # Create output queue(s) to get the HQ (+ LQ) frames
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    if args.save_lq_frames:
        q_frame_lq = device.getOutputQueue(name="frame_lq", maxSize=4, blocking=False)

    # Create input queue to send control commands to OAK camera
    q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False)

    # Wait for 2 seconds to let camera adjust auto focus and exposure
    time.sleep(2)

    # Print info on start of recording
    logging.info("Rec time: %s min", int(REC_TIME / 60))

    # Create variables for start of recording and capture/check events
    start_time = time.monotonic()
    last_capture = start_time - TIMELAPSE_INT  # capture first frame immediately at start
    last_disk_check = start_time

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Activate trigger to save HQ frame based on current time and specified intervals
        current_time = time.monotonic()
        trigger_timelapse = current_time - last_capture >= TIMELAPSE_INT

        if trigger_timelapse:
            # Send control command to OAK camera to capture still frame
            q_ctrl.send(ctrl_capture)

            if q_frame.has():
                # Get MJPEG-encoded HQ frame
                timestamp = datetime.now()
                timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")
                frame_hq = q_frame.get().getData()

                # Save MJPEG-encoded HQ frame to .jpg file in separate thread
                executor.submit(save_encoded_frame, save_path, timestamp_str, frame_hq)
                last_capture = current_time

                if args.save_lq_frames and q_frame_lq.has():
                    # Get LQ frame and save to .jpg
                    frame_lq = q_frame_lq.get().getCvFrame()
                    cv2.imwrite(f"{save_path}/LQ_frames/{timestamp_str}_LQ.jpg", frame_lq)

                # Update free disk space (MB) at specified interval
                if current_time - last_disk_check >= FREE_SPACE_INT:
                    disk_free = round(psutil.disk_usage("/").free / 1048576)
                    last_disk_check = current_time

        # Sleep for a short duration to avoid busy waiting
        time.sleep(0.02)

# Print number and directory of saved frames
num_frames_hq = len(list(save_path.glob("*.jpg")))
if not args.save_lq_frames:
    logging.info("Saved %s HQ frames to %s", num_frames_hq, save_path)
else:
    num_frames_lq = len(list((save_path / "LQ_frames").glob("*.jpg")))
    logging.info("Saved %s HQ and %s LQ frames to %s", num_frames_hq, num_frames_lq, save_path)
