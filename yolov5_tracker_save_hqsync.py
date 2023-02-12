#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- run a custom YOLOv5 object detection model (.blob format) on-device (Luxonis OAK)
- use an object tracker (Intel DL Streamer) to track detected objects and set unique tracking IDs
- synchronize tracker output (+ detections) from inference on full FOV LQ frames (e.g. 416x416)
  with HQ frames (e.g. 3840x2160) on-device using the respective sequence numbers
- save detections (bounding box area) cropped from HQ frames to .jpg
- save metadata from model + tracker output (time, label, confidence, tracking ID,
  relative bbox coordinates, .jpg file path) to "metadata_{timestamp}.csv"
- write info and error (stderr) + traceback messages to log file ("script_log.log")
- duration of each recording interval conditional on PiJuice battery charge level
- shut down without recording if charge level or free disk space is lower than threshold
- write record info (recording ID, start/end time, duration, number of cropped detections,
  number of unique tracking IDs, free disk space and battery charge level) to "record_log.csv"
  and safely shut down RPi after recording interval is finished or if charge level drops
  below threshold or if an error occurs
- optional arguments:
  "-raw" additionally save HQ frames to .jpg (e.g. for training data collection) OR
  "-overlay" additionally save HQ frames with overlay (bbox + info) to .jpg
  "-log" write RPi CPU + OAK VPU temperatures, RPi available memory (MB) +
         CPU utilization (percent) and battery info to "info_log_{timestamp}.csv"

includes segments from open source scripts available at https://github.com/luxonis
'''

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import pandas as pd
import psutil
from gpiozero import CPUTemperature
from pijuice import PiJuice

# Create data folder if not already present
Path("./insect-detect/data").mkdir(parents=True, exist_ok=True)

# Create logger and send info + error messages to log file
logging.basicConfig(filename = "./insect-detect/data/script_log.log",
                    encoding = "utf-8",
                    format = "%(asctime)s - %(levelname)s: %(message)s",
                    level = logging.DEBUG)
logger = logging.getLogger()
sys.stderr.write = logger.error

# Set file paths to the detection model and config JSON
MODEL_PATH = Path("./insect-detect/models/yolov5n_416_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("./insect-detect/models/json/yolov5_416.json")

# Instantiate PiJuice
pijuice = PiJuice(1, 0x14)

# Continue script only if PiJuice battery charge level and free disk space are higher than thresholds
chargelevel_start = pijuice.status.GetChargeLevel().get("data", -1)
disk_free = round(psutil.disk_usage("/").free / 1048576) # free disk space in MB
if chargelevel_start < 10 or disk_free < 200:
    logger.info(f"Shut down without recording | Charge level: {chargelevel_start}%\n")
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
    time.sleep(5) # wait 5 seconds for RPi to shut down

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-raw", "--save_raw_frames", action="store_true",
    help="additionally save full raw HQ frames in separate folder (e.g. for training data)")
group.add_argument("-overlay", "--save_overlay_frames", action="store_true",
    help="additionally save full HQ frames with overlay (bbox + info) in separate folder")
parser.add_argument("-log", "--save_logs", action="store_true",
    help="save RPi CPU + OAK VPU temperatures, RPi available memory (MB) + \
          CPU utilization (percent) and battery info to .csv file")
args = parser.parse_args()

# Extract detection model metadata from config JSON
with CONFIG_PATH.open(encoding="utf-8") as f:
    config = json.load(f)
nn_config = config.get("nn_config", {})
nn_metadata = nn_config.get("NN_specific_metadata", {})
classes = nn_metadata.get("classes", {})
coordinates = nn_metadata.get("coordinates", {})
anchors = nn_metadata.get("anchors", {})
anchor_masks = nn_metadata.get("anchor_masks", {})
iou_threshold = nn_metadata.get("iou_threshold", {})
confidence_threshold = nn_metadata.get("confidence_threshold", {})
nn_mappings = config.get("mappings", {})
labels = nn_mappings.get("labels", {})

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setVideoSize(3840, 2160) # HQ frames for syncing, aspect ratio 16:9 (4K)
cam_rgb.setPreviewSize(416, 416) # downscaled LQ frames for model input
cam_rgb.setInterleaved(False)
cam_rgb.setPreviewKeepAspectRatio(False) # squash full FOV frames to square
cam_rgb.setFps(30) # frames per second available for focus/exposure/model input

# Create detection network node and define input
nn = pipeline.create(dai.node.YoloDetectionNetwork)
cam_rgb.preview.link(nn.input) # downscaled LQ frames as model input
nn.input.setBlocking(False)

# Set detection model specific settings
nn.setBlobPath(MODEL_PATH)
nn.setNumClasses(classes)
nn.setCoordinateSize(coordinates)
nn.setAnchors(anchors)
nn.setAnchorMasks(anchor_masks)
nn.setIouThreshold(iou_threshold)
nn.setConfidenceThreshold(confidence_threshold)
nn.setNumInferenceThreads(2)

# Create and configure object tracker node and define inputs
tracker = pipeline.create(dai.node.ObjectTracker)
tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
nn.passthrough.link(tracker.inputTrackerFrame)
nn.passthrough.link(tracker.inputDetectionFrame)
nn.out.link(tracker.inputDetections)

# Create script node and define inputs (to sync detections with HQ frames)
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)
tracker.out.link(script.inputs["tracker"]) # tracker output + passthrough detections
cam_rgb.video.link(script.inputs["frames"]) # HQ frames
script.inputs["frames"].setBlocking(False)

# Set script that will be run on-device (Luxonis OAK)
script.setScript('''
# Create empty list to save HQ frames + sequence numbers
lst = []

def get_synced_frame(track_seq):
    """Compare tracker with frame sequence number and send frame if equal."""
    global lst
    for i, frame in enumerate(lst):
        if track_seq == frame.getSequenceNum():
            lst = lst[i:]
            break
    return lst[0]

# Sync tracker output with HQ frames
while True:
    lst.append(node.io["frames"].get())
    tracks = node.io["tracker"].tryGet()
    if tracks is not None:
        track_seq = node.io["tracker"].get().getSequenceNum()
        if len(lst) == 0: continue
        node.io["frame_out"].send(get_synced_frame(track_seq))
        node.io["track_out"].send(tracks)
        lst.pop(0) # remove synchronized frame from the list
''')

# Define script node outputs
xout_track = pipeline.create(dai.node.XLinkOut)
xout_track.setStreamName("track")
script.outputs["track_out"].link(xout_track.input) # synced tracker output

xout_frame = pipeline.create(dai.node.XLinkOut)
xout_frame.setStreamName("frame")
script.outputs["frame_out"].link(xout_frame.input) # synced HQ frames

# Create new folders for each day, recording interval and object class
rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
save_path = f"./insect-detect/data/{rec_start[:8]}/{rec_start}"
for text in labels:
    Path(f"{save_path}/cropped/{text}").mkdir(parents=True, exist_ok=True)
    if args.save_overlay_frames:
        Path(f"{save_path}/overlay/{text}").mkdir(parents=True, exist_ok=True)
if args.save_raw_frames:
    Path(f"{save_path}/raw").mkdir(parents=True, exist_ok=True)

# Calculate current recording ID by subtracting number of directories with date-prefix
folders_dates = len([f for f in Path("./insect-detect/data").glob("**/20*") if f.is_dir()])
folders_days = len([f for f in Path("./insect-detect/data").glob("20*") if f.is_dir()])
rec_id = folders_dates - folders_days

def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def store_data(frame, tracklets):
    """Save synced cropped (+ full) frames and tracker output (+ detections) to .jpg and metadata .csv."""
    with open(f"{save_path}/metadata_{rec_start}.csv", "a", encoding="utf-8") as metadata_file:
        metadata = csv.DictWriter(metadata_file, fieldnames=
            ["rec_ID", "timestamp", "label", "confidence", "track_ID",
             "x_min", "y_min", "x_max", "y_max", "file_path"])
        if metadata_file.tell() == 0:
            metadata.writeheader() # write header only once
        for t in tracklets:
            # Do not save (cropped) frames when tracking status == "NEW" or "LOST" or "REMOVED"
            if t.status.name == "TRACKED":
                timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")

                # Save detection cropped from synced HQ frame
                bbox = frame_norm(frame, (t.srcImgDetection.xmin, t.srcImgDetection.ymin,
                                          t.srcImgDetection.xmax, t.srcImgDetection.ymax))
                det_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cropped_path = f"{save_path}/cropped/{labels[t.srcImgDetection.label]}/{timestamp}_{t.id}_cropped.jpg"
                cv2.imwrite(cropped_path, det_crop)

                # Save corresponding metadata to .csv file for each cropped detection
                data = {
                    "rec_ID": rec_id,
                    "timestamp": timestamp,
                    "label": labels[t.srcImgDetection.label],
                    "confidence": round(t.srcImgDetection.confidence, 2),
                    "track_ID": t.id,
                    "x_min": round(t.srcImgDetection.xmin, 4),
                    "y_min": round(t.srcImgDetection.ymin, 4),
                    "x_max": round(t.srcImgDetection.xmax, 4),
                    "y_max": round(t.srcImgDetection.ymax, 4),
                    "file_path": cropped_path
                }
                metadata.writerow(data)
                metadata_file.flush() # write data immediately to .csv to avoid potential data loss

                # Save full HQ frame with overlay (bounding box, label, confidence, tracking ID) drawn on frame
                # text position, font size and thickness optimized for 3840x2160 HQ frame size
                if args.save_overlay_frames:
                    overlay_frame = frame.copy()
                    cv2.putText(overlay_frame, labels[t.srcImgDetection.label], (bbox[0], bbox[3] + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
                    cv2.putText(overlay_frame, f"{round(t.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                    cv2.putText(overlay_frame, f"ID:{t.id}", (bbox[0], bbox[3] + 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
                    cv2.rectangle(overlay_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                    overlay_path = f"{save_path}/overlay/{labels[t.srcImgDetection.label]}/{timestamp}_{t.id}_overlay.jpg"
                    cv2.imwrite(overlay_path, overlay_frame)

        # Save full raw HQ frame (e.g. for training data collection)
        if args.save_raw_frames:
            # save only once in case of multiple detections
            for i, t in enumerate(tracklets):
                if i == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                    raw_path = f"{save_path}/raw/{timestamp}_raw.jpg"
                    cv2.imwrite(raw_path, frame)

def record_log():
    """Write information about each recording interval to .csv file."""
    try:
        df_meta = pd.read_csv(f"{save_path}/metadata_{rec_start}.csv", encoding="utf-8")
        unique_ids = df_meta.track_ID.nunique()
    except pd.errors.EmptyDataError:
        unique_ids = 0
    with open("./insect-detect/data/record_log.csv", "a", encoding="utf-8") as log_rec_file:
        log_rec = csv.DictWriter(log_rec_file, fieldnames=
            ["rec_ID", "record_start_date", "record_start_time", "record_end_time", "record_time_min",
             "num_crops", "num_IDs", "disk_free_gb", "chargelevel_start", "chargelevel_end"])
        if log_rec_file.tell() == 0:
            log_rec.writeheader()
        logs_rec = {
            "rec_ID": rec_id,
            "record_start_date": rec_start[:8],
            "record_start_time": rec_start[9:],
            "record_end_time": datetime.now().strftime("%H-%M"),
            "record_time_min": round((time.monotonic() - start_time) / 60, 2),
            "num_crops": len(list(Path(f"{save_path}/cropped").glob("**/*.jpg"))),
            "num_IDs": unique_ids,
            "disk_free_gb": round(psutil.disk_usage("/").free / 1073741824, 1),
            "chargelevel_start": chargelevel_start,
            "chargelevel_end": chargelevel
        }
        log_rec.writerow(logs_rec)

def save_logs():
    """
    Write recording ID, time, RPi CPU + OAK VPU temp, RPi available memory (MB) +
    CPU utilization (percent) and PiJuice battery info + temp to .csv file.
    """
    with open(f"./insect-detect/data/{rec_start[:8]}/info_log_{rec_start[:8]}.csv",
              "a", encoding="utf-8") as log_info_file:
        log_info = csv.DictWriter(log_info_file, fieldnames=
            ["rec_ID", "timestamp", "temp_pi", "temp_oak", "pi_mem_available", "pi_cpu_used",
             "power_input", "charge_status", "charge_level", "temp_batt", "voltage_batt_mV",
             "current_batt_mA", "current_gpio_mA"])
        if log_info_file.tell() == 0:
            log_info.writeheader()
        logs_info = {
            "rec_ID": rec_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H-%M-%S"),
            "temp_pi": round(CPUTemperature().temperature),
            "temp_oak": round(device.getChipTemperature().average),
            "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
            "pi_cpu_used": psutil.cpu_percent(interval=None),
            "power_input": pijuice.status.GetStatus().get("data", {}).get("powerInput", 0),
            "charge_status": pijuice.status.GetStatus().get("data", {}).get("battery", 0),
            "charge_level": chargelevel,
            "temp_batt": pijuice.status.GetBatteryTemperature().get("data", 0),
            "voltage_batt_mV": pijuice.status.GetBatteryVoltage().get("data", 0),
            "current_batt_mA": pijuice.status.GetBatteryCurrent().get("data", 0),
            "current_gpio_mA": pijuice.status.GetIoCurrent().get("data", 0)
        }
        log_info.writerow(logs_info)
        log_info_file.flush()

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create output queues to get the frames and detections from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    # Set battery charge level and recording start time
    chargelevel = chargelevel_start
    start_time = time.monotonic()

    # Define recording time conditional on PiJuice battery charge level
    if chargelevel >= 70:
        rec_time = 60 * 40
    elif 50 <= chargelevel < 70:
        rec_time = 60 * 30
    elif 30 <= chargelevel < 50:
        rec_time = 60 * 20
    elif 15 <= chargelevel < 30:
        rec_time = 60 * 10
    else:
        rec_time = 60 * 5
    logger.info(f"Rec ID: {rec_id} | Rec time: {int(rec_time / 60)} min | Charge level: {chargelevel}%")

    try:
        # Record until recording time is finished or chargelevel drops below threshold
        while time.monotonic() < start_time + rec_time and chargelevel >= 10:

            # Update PiJuice battery charge level
            chargelevel = pijuice.status.GetChargeLevel().get("data", -1)

            # Get synced HQ frames and tracker output (detections + tracking IDs)
            frame_synced = q_frame.get().getCvFrame()
            track_synced = q_track.get()
            if track_synced is not None:
                tracklets_data = track_synced.tracklets
                if frame_synced is not None:
                    # save cropped detections every second (slower if saving additional HQ frames)
                    store_data(frame_synced, tracklets_data)
                    time.sleep(1)

            # Write RPi CPU + OAK VPU temp, RPi info and battery info + temp to .csv log file
            if args.save_logs:
                save_logs()

        # Write record logs to .csv and shutdown Raspberry Pi after recording time is finished
        record_log()
        logger.info(f"Recording {rec_id} finished | Charge level: {chargelevel}%\n")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

    # Write record logs to .csv, log error traceback and shutdown Raspberry Pi if an error occurs
    except Exception:
        logger.error(traceback.format_exc())
        logger.error(f"Error during recording {rec_id} | Charge level: {chargelevel}%\n")
        record_log()
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
