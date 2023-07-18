#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- write info and error (stderr) messages (+ traceback) to log file ("script_log.log")
- shut down without recording if free disk space (MB) is lower than specified threshold
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled LQ frames (e.g. 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs (on-device)
- synchronize tracker output (+ passthrough detections) from inference on LQ frames
  with HQ frames (e.g. 1920x1080 px) on-device using the respective sequence numbers
- create new folders for each day, recording interval and object class
- save detections (bounding box area) cropped from HQ frames to .jpg (1080p: ~12.5 fps)
- save corresponding metadata from model + tracker output (time, label, confidence,
  tracking ID, relative bbox coordinates, .jpg file path) to "metadata_{timestamp}.csv"
- write record info (recording ID, start/end time, duration, number of cropped detections,
  number of unique tracking IDs, free disk space) to "record_log.csv"
  and safely shut down RPi after recording interval is finished or if an error occurs
- optional arguments:
  "-min [min]" (default = 2) set recording time in minutes
               -> e.g. "-min 5" for 5 min recording time
  "-4k" (default = 1080p) crop detections from (+ save HQ frames in) 4K resolution
        -> will slow down pipeline speed to ~3.4 fps (1080p: ~12.5 fps)
  "-square" save cropped detections with aspect ratio 1:1
            -> increase bbox size on both sides of the minimum dimension
  "-raw" additionally save HQ frames to .jpg (e.g. for training data collection)
         -> will slow down pipeline speed to ~4.5 fps (4K sync: ~1.2 fps)
  "-overlay" additionally save HQ frames with overlay (bbox + info) to .jpg
             -> will slow down pipeline speed to ~4.5 fps (4K sync: ~1.2 fps)
  "-log" write RPi CPU + OAK chip temperature and RPi available memory (MB) +
         CPU utilization (%) to "info_log_{timestamp}.csv"

based on open source scripts available at https://github.com/luxonis
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

# Create folder to save images + metadata + logs (if not already present)
Path("insect-detect/data").mkdir(parents=True, exist_ok=True)

# Create logger and write info + error messages to log file
logging.basicConfig(filename="insect-detect/data/script_log.log", encoding="utf-8",
                    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger()
sys.stderr.write = logger.error

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="set record time in minutes")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="crop detections from (+ save HQ frames in) 4K resolution; default = 1080p")
parser.add_argument("-square", "--square_bbox_crop", action="store_true",
    help="save cropped detections with aspect ratio 1:1")
parser.add_argument("-raw", "--save_raw_frames", action="store_true",
    help="additionally save full raw HQ frames in separate folder (e.g. for training data)")
parser.add_argument("-overlay", "--save_overlay_frames", action="store_true",
    help="additionally save full HQ frames with overlay (bbox + info) in separate folder")
parser.add_argument("-log", "--save_logs", action="store_true",
    help="save RPi CPU + OAK chip temperature and RPi available memory (MB) + \
          CPU utilization (%) to .csv file")
args = parser.parse_args()

if args.save_logs:
    from apscheduler.schedulers.background import BackgroundScheduler
    from gpiozero import CPUTemperature

# Continue script only if free disk space (MB) is higher than threshold
disk_free = round(psutil.disk_usage("/").free / 1048576)
if disk_free < 200:
    logger.info(f"Shut down without recording | Free disk space left: {disk_free} MB\n")
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
    time.sleep(5) # wait 5 seconds for RPi to shut down

# Set file paths to the detection model and config JSON
MODEL_PATH = Path("insect-detect/models/yolov5n_320_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("insect-detect/models/json/yolov5_v7_320.json")

# Get detection model metadata from config JSON
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
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2) # downscale 4K to 1080p HQ frames (1920x1080 px)
cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames for model input
cam_rgb.setPreviewKeepAspectRatio(False) # "squeeze" frames (16:9) to square (1:1)
cam_rgb.setInterleaved(False) # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25) # frames per second available for focus/exposure/model input

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
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
#tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS) # better for low fps
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
nn.passthrough.link(tracker.inputTrackerFrame)
nn.passthrough.link(tracker.inputDetectionFrame)
nn.out.link(tracker.inputDetections)

# Create script node and define inputs
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)
cam_rgb.video.link(script.inputs["frames"]) # HQ frames
script.inputs["frames"].setBlocking(False)
tracker.out.link(script.inputs["tracker"]) # tracklets + passthrough detections
script.inputs["tracker"].setBlocking(False)

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
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
script.outputs["frame_out"].link(xout_rgb.input) # synced HQ frames

xout_tracker = pipeline.create(dai.node.XLinkOut)
xout_tracker.setStreamName("track")
script.outputs["track_out"].link(xout_tracker.input) # synced tracker output

# Create new folders for each day, recording interval and object class
rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
save_path = f"insect-detect/data/{rec_start[:8]}/{rec_start}"
for text in labels:
    Path(f"{save_path}/cropped/{text}").mkdir(parents=True, exist_ok=True)
if args.save_raw_frames:
    Path(f"{save_path}/raw").mkdir(parents=True, exist_ok=True)
if args.save_overlay_frames:
    Path(f"{save_path}/overlay").mkdir(parents=True, exist_ok=True)

# Calculate current recording ID by subtracting number of directories with date-prefix
folders_dates = len([f for f in Path("insect-detect/data").glob("**/20*") if f.is_dir()])
folders_days = len([f for f in Path("insect-detect/data").glob("20*") if f.is_dir()])
rec_id = folders_dates - folders_days

# Define functions
def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def store_data(frame, tracks):
    """Save cropped detections (+ full HQ frames) to .jpg and tracker output to metadata .csv."""
    with open(f"{save_path}/metadata_{rec_start}.csv", "a", encoding="utf-8") as metadata_file:
        metadata = csv.DictWriter(metadata_file, fieldnames=
            ["rec_ID", "timestamp", "label", "confidence", "track_ID",
             "x_min", "y_min", "x_max", "y_max", "file_path"])
        if metadata_file.tell() == 0:
            metadata.writeheader() # write header only once

        # Save full raw HQ frame (e.g. for training data collection)
        if args.save_raw_frames:
            for track in tracks:
                if track == tracks[-1]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                    raw_path = f"{save_path}/raw/{timestamp}_raw.jpg"
                    cv2.imwrite(raw_path, frame)
                    #cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        for track in tracks:
            # Don't save cropped detections if tracking status == "NEW" or "LOST" or "REMOVED"
            if track.status.name == "TRACKED":

                # Save detections cropped from HQ frame to .jpg
                bbox = frame_norm(frame, (track.srcImgDetection.xmin, track.srcImgDetection.ymin,
                                          track.srcImgDetection.xmax, track.srcImgDetection.ymax))
                if args.square_bbox_crop:
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    bbox_diff = (max(bbox_width, bbox_height) - min(bbox_width, bbox_height)) // 2
                    if bbox_width < bbox_height:
                        if bbox[0] - bbox_diff < 0:
                            det_crop = frame[bbox[1]:bbox[3], 0:bbox[2] + (bbox_diff * 2)]
                        else:
                            det_crop = frame[bbox[1]:bbox[3], bbox[0] - bbox_diff:bbox[2] + bbox_diff]
                    else:
                        if bbox[1] - bbox_diff < 0:
                            det_crop = frame[0:bbox[3] + (bbox_diff * 2), bbox[0]:bbox[2]]
                        else:
                            det_crop = frame[bbox[1] - bbox_diff:bbox[3] + bbox_diff, bbox[0]:bbox[2]]
                else:
                    det_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                label = labels[track.srcImgDetection.label]
                timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                crop_path = f"{save_path}/cropped/{label}/{timestamp}_{track.id}_crop.jpg"
                cv2.imwrite(crop_path, det_crop)

                # Save corresponding metadata to .csv file for each cropped detection
                data = {
                    "rec_ID": rec_id,
                    "timestamp": timestamp,
                    "label": label,
                    "confidence": round(track.srcImgDetection.confidence, 2),
                    "track_ID": track.id,
                    "x_min": round(track.srcImgDetection.xmin, 4),
                    "y_min": round(track.srcImgDetection.ymin, 4),
                    "x_max": round(track.srcImgDetection.xmax, 4),
                    "y_max": round(track.srcImgDetection.ymax, 4),
                    "file_path": crop_path
                }
                metadata.writerow(data)
                metadata_file.flush() # write data immediately to .csv to avoid potential data loss

                # Save full HQ frame with overlay (bounding box, label, confidence, tracking ID) drawn on frame
                if args.save_overlay_frames:
                    # Text position, font size and thickness optimized for 1920x1080 px HQ frame size
                    if not args.four_k_resolution:
                        cv2.putText(frame, labels[track.srcImgDetection.label], (bbox[0], bbox[3] + 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(frame, f"{round(track.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(frame, f"ID:{track.id}", (bbox[0], bbox[3] + 92),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    # Text position, font size and thickness optimized for 3840x2160 px HQ frame size
                    else:
                        cv2.putText(frame, labels[track.srcImgDetection.label], (bbox[0], bbox[3] + 48),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)
                        cv2.putText(frame, f"{round(track.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 98),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
                        cv2.putText(frame, f"ID:{track.id}", (bbox[0], bbox[3] + 164),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                    if track == tracks[-1]:
                        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                        overlay_path = f"{save_path}/overlay/{timestamp}_overlay.jpg"
                        cv2.imwrite(overlay_path, frame)
                        #cv2.imwrite(overlay_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

def record_log():
    """Write information about each recording interval to .csv file."""
    try:
        df_meta = pd.read_csv(f"{save_path}/metadata_{rec_start}.csv", encoding="utf-8")
        unique_ids = df_meta["track_ID"].nunique()
    except pd.errors.EmptyDataError:
        unique_ids = 0
    with open("insect-detect/data/record_log.csv", "a", encoding="utf-8") as log_rec_file:
        log_rec = csv.DictWriter(log_rec_file, fieldnames=
            ["rec_ID", "record_start_date", "record_start_time", "record_end_time",
             "record_time_min", "num_crops", "num_IDs", "disk_free_gb"])
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
            "disk_free_gb": round(psutil.disk_usage("/").free / 1073741824, 1)
        }
        log_rec.writerow(logs_rec)

def save_logs():
    """
    Write recording ID, time, RPi CPU + OAK chip temperature and RPi available memory (MB) +
    CPU utilization (%) to .csv file.
    """
    with open(f"insect-detect/data/{rec_start[:8]}/info_log_{rec_start[:8]}.csv", "a",
              encoding="utf-8") as log_info_file:
        log_info = csv.DictWriter(log_info_file, fieldnames=
            ["rec_ID", "timestamp", "temp_pi", "temp_oak", "pi_mem_available", "pi_cpu_used"])
        if log_info_file.tell() == 0:
            log_info.writeheader()
        try:
            temp_oak = round(device.getChipTemperature().average)
        except RuntimeError:
            temp_oak = "NA"
        logs_info = {
            "rec_ID": rec_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H-%M-%S"),
            "temp_pi": round(CPUTemperature().temperature),
            "temp_oak": temp_oak,
            "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
            "pi_cpu_used": psutil.cpu_percent(interval=None)
        }
        log_info.writerow(logs_info)
        log_info_file.flush()

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Write RPi + OAK info to .csv log file at specified interval
    if args.save_logs:
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        scheduler = BackgroundScheduler()
        scheduler.add_job(save_logs, "interval", seconds=30, id="log")
        scheduler.start()

    # Create output queues to get the frames and tracklets + detections from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    # Create start_time variable to set recording time
    start_time = time.monotonic()

    # Get recording time in min from optional argument (default: 2)
    rec_time = args.min_rec_time * 60

    # Write info on start of recording to log file
    logger.info(f"Rec ID: {rec_id} | Rec time: {args.min_rec_time} min")

    try:
        # Record until recording time is finished
        while time.monotonic() < start_time + rec_time:

            # Get synchronized HQ frames + tracker output (passthrough detections)
            if q_frame.has():
                frame = q_frame.get().getCvFrame()

                if q_track.has():
                    tracks = q_track.get().tracklets

                    # Save cropped detections every second (slower if saving additional HQ frames)
                    store_data(frame, tracks)
                    time.sleep(1)

        # Write info on end of recording to log file and write record logs to .csv
        logger.info(f"Recording {rec_id} finished\n")
        record_log()

        # Shutdown Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

    # Write info on error during recording to log file and write record logs to .csv
    except Exception:
        logger.error(traceback.format_exc())
        logger.error(f"Error during recording {rec_id}\n")
        record_log()

        # Shutdown Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
