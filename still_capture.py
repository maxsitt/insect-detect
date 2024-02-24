#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- save encoded still frames in highest possible resolution (default: 4032x3040 px)
  to .jpg at specified capture frequency (default: ~every second)
  -> stop recording early if free disk space drops below threshold
- optional arguments:
  "-min" set recording time in minutes (default: 2 min)
         -> e.g. "-min 5" for 5 min recording time
  "-af"  set auto focus range in cm (min distance, max distance)
         -> e.g. "-af 14 20" to restrict auto focus range to 14-20 cm
  "-zip" store all captured data in an uncompressed .zip
         file for each day and delete original folder
         -> increases file transfer speed from microSD to computer
            but also on-device processing time and power consumption

based on open source scripts available at https://github.com/luxonis
'''

import argparse
import time
from datetime import datetime
from pathlib import Path

import depthai as dai
import psutil

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="set record time in minutes (default: 2 min)")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="set auto focus range in cm (min distance, max distance)", metavar=("cm_min", "cm_max"))
parser.add_argument("-zip", "--save_zip", action="store_true",
    help="store all captured data in an uncompressed .zip \
          file for each day and delete original folder")
args = parser.parse_args()

if args.save_zip:
    import shutil
    from zipfile import ZipFile

# Create folders for each day and recording interval to save still frames
rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
save_path = Path(f"insect-detect/stills/{rec_start[:8]}/{rec_start}")
save_path.mkdir(parents=True, exist_ok=True)

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set capture frequency (default: ~every second)
# -> wait for specified amount of seconds between saving still frames
# 'CAPTURE_FREQ = 1' saves ~54 still frames per minute to .jpg (12 MP)
CAPTURE_FREQ = 1

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)      # OAK-1 (IMX378)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)     # OAK-1 Lite (IMX214)
#cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_5312X6000) # OAK-1 MAX (IMX582)
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setNumFramesPool(2,2,2,2,2)  # decrease frame pool size to avoid memory issues
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure

# Create and configure video encoder node and define input + output
still_enc = pipeline.create(dai.node.VideoEncoder)
still_enc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
still_enc.setNumFramesPool(1)
cam_rgb.still.link(still_enc.input)

xout_still = pipeline.create(dai.node.XLinkOut)
xout_still.setStreamName("still")
still_enc.bitstream.link(xout_still.input)

# Create script node
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

# Set script that will be run on OAK device to send capture still command
script.setScript('''
ctrl = CameraControl()
ctrl.setCaptureStill(True)
while True:
    node.io["capture_still"].send(ctrl)
''')

# Define script node output to send capture still command to color camera node
script.outputs["capture_still"].link(cam_rgb.inputControl)

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


def save_zip():
    """Store all captured data in an uncompressed .zip
    file for each day and delete original folder."""
    with ZipFile(f"{save_path.parent}.zip", "a") as zip_file:
        for file in save_path.rglob("*"):
            zip_file.write(file, file.relative_to(save_path.parent))
    shutil.rmtree(save_path.parent, ignore_errors=True)


# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Print recording time to console (default: 2 minutes)
    print(f"\nRecording time: {int(REC_TIME / 60)} min\n")

    # Get free disk space (MB)
    disk_free = round(psutil.disk_usage("/").free / 1048576)

    # Create output queue to get the encoded still frames from the output defined above
    q_still = device.getOutputQueue(name="still", maxSize=1, blocking=False)

    if args.af_range:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

        # Set auto focus range to specified cm values
        set_focus_range()

    # Set start time of recording
    start_time = time.monotonic()

    # Record until recording time is finished
    # Stop recording early if free disk space drops below threshold
    while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

        # Update free disk space (MB)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

        # Get encoded still frames and save to .jpg
        if q_still.has():
            frame_still = q_still.get().getData()
            timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
            with open(save_path / f"{timestamp}.jpg", "wb") as still_jpg:
                still_jpg.write(frame_still)

        # Wait for specified amount of seconds (default: 1)
        time.sleep(CAPTURE_FREQ)

# Print number and path of saved still frames to console
num_frames_still = len(list(save_path.glob("*.jpg")))
print(f"Saved {num_frames_still} still frames to {save_path}.")

if args.save_zip:
    # Store frames in uncompressed .zip file and delete original folder
    save_zip()
    print(f"\nStored all captured images in {save_path.parent}.zip\n")
