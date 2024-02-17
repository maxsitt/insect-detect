#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- write info and error (stderr) messages (+ traceback) to log file ("script_log.log")
- shut down Raspberry Pi without recording if free disk space
  is lower than the specified threshold (default: 100 MB)
- create folders for each day, recording interval and object class to save images + metadata
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on stretched + downscaled LQ frames (default: 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on object motion speed and inference speed of the detection model
- synchronize tracker output (including detections) from inference on LQ frames with
  HQ frames (default: 1920x1080 px) on-device using the respective message timestamps
  -> pipeline speed (= inference speed): ~13.4 fps (1080p sync) or ~3.3 fps (4K sync)
- save detections (bounding box area) cropped from HQ frames to .jpg at the
  specified capture frequency (default: 1 s), optionally together with full frames
- save corresponding metadata from tracker (+ model) output (time, label, confidence,
  tracking ID, relative bbox coordinates, .jpg file path) to "metadata_{timestamp}.csv"
- write info about recording interval (rec ID, start/end time, duration, number of cropped
  detections, unique tracking IDs, free disk space) to "record_log.csv"
- shut down Raspberry Pi after recording interval is finished or if free
  disk space drops below the specified threshold or if an error occurs
- optional arguments:
  "-min"     set recording time in minutes (default: 2 min)
             -> e.g. "-min 5" for 5 min recording time
  "-4k"      crop detections from (+ save HQ frames in) 4K resolution (default: 1080p)
             -> decreases pipeline speed to ~3.3 fps (1080p: ~13.4 fps)
  "-af"      set auto focus range in cm (min distance, max distance)
             -> e.g. "-af 14 20" to restrict auto focus range to 14-20 cm
  "-ae"      use bounding box coordinates from detections to set auto exposure region
             -> can improve image quality of crops and thereby classification accuracy
  "-crop"    default:  save cropped detections with aspect ratio 1:1 ("-crop square") OR
             optional: keep original bbox size with variable aspect ratio ("-crop tight")
             -> "-crop square" increases bbox size on both sides of the minimum dimension,
                               or only on one side if object is localized at frame margin
                -> can increase classification accuracy by avoiding stretching of the
                   cropped insect image during resizing for classification inference
  "-raw"     additionally save HQ frames to .jpg (e.g. for training data collection)
             -> decreases pipeline speed to ~4.7 fps for 1080p sync (4K sync: ~1.2 fps)
  "-overlay" additionally save HQ frames with overlays (bbox + info) to .jpg
             -> decreases pipeline speed to ~4.5 fps for 1080p sync (4K sync: ~1.2 fps)
  "-log"     write RPi CPU + OAK chip temperature and RPi available memory (MB) +
             CPU utilization (%) to "info_log_{timestamp}.csv"
  "-zip"     store all captured data in an uncompressed .zip
             file for each day and delete original folder
             -> increases file transfer speed from microSD to computer
                but also on-device processing time and power consumption

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
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import pandas as pd
import psutil

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--min_rec_time", type=int, choices=range(1, 721), default=2,
    help="set record time in minutes (default: 2 min)")
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="crop detections from (+ save HQ frames in) 4K resolution (default: 1080p)")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="set auto focus range in cm (min distance, max distance)", metavar=("cm_min", "cm_max"))
parser.add_argument("-ae", "--bbox_ae_region", action="store_true",
    help="use bounding box coordinates from detections to set auto exposure region")
parser.add_argument("-crop", "--crop_bbox", choices=["square", "tight"], default="square", type=str,
    help="save cropped detections with aspect ratio 1:1 ('-crop square') or \
          keep original bbox size with variable aspect ratio ('-crop tight')")
parser.add_argument("-raw", "--save_raw_frames", action="store_true",
    help="additionally save full raw HQ frames in separate folder (e.g. for training data)")
parser.add_argument("-overlay", "--save_overlay_frames", action="store_true",
    help="additionally save full HQ frames with overlays (bbox + info) in separate folder")
parser.add_argument("-log", "--save_logs", action="store_true",
    help="write RPi CPU + OAK chip temperature and RPi available memory (MB) + \
          CPU utilization (%) to .csv file")
parser.add_argument("-zip", "--save_zip", action="store_true",
    help="store all captured data in an uncompressed .zip \
          file for each day and delete original folder")
args = parser.parse_args()

if args.save_logs:
    from apscheduler.schedulers.background import BackgroundScheduler
    from gpiozero import CPUTemperature

if args.save_zip:
    import shutil
    from zipfile import ZipFile

# Create folders for each day and recording interval to save images + metadata + logs
rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
save_path = Path(f"insect-detect/data/{rec_start[:8]}/{rec_start}")
save_path.mkdir(parents=True, exist_ok=True)
if args.save_raw_frames:
    (save_path / "raw").mkdir(parents=True, exist_ok=True)
if args.save_overlay_frames:
    (save_path / "overlay").mkdir(parents=True, exist_ok=True)

# Create logger and write info + error messages to log file
logging.basicConfig(filename=save_path.parents[1] / "script_log.log", encoding="utf-8",
                    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger()
sys.stderr.write = logger.error

# Set threshold value required to start and continue a recording
MIN_DISKSPACE = 100  # minimum free disk space (MB) (default: 100 MB)

# Set file paths to the detection model and corresponding config JSON
MODEL_PATH = Path("insect-detect/models/yolov5n_320_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("insect-detect/models/json/yolov5_v7_320.json")

# Set frequency for saving logs to .csv file (default: 30 seconds)
LOG_FREQ = 30

# Set capture frequency (default: 1 second)
# -> wait for specified amount of seconds between saving cropped detections + metadata
# -> frequency decreases if full frames are saved additionally ("-raw" or "-overlay")
CAPTURE_FREQ = 1

# Set recording time (default: 2 minutes)
REC_TIME = args.min_rec_time * 60

# Calculate current recording ID by subtracting number of directories with date-prefix
folders_dates = len([f for f in Path("insect-detect/data").glob("**/20*") if f.is_dir()])
folders_days = len([f for f in Path("insect-detect/data").glob("20*") if f.is_dir()])
rec_id = folders_dates - folders_days

# Shut down Raspberry Pi if free disk space (MB) is lower than threshold
disk_free = round(psutil.disk_usage("/").free / 1048576)
if disk_free < MIN_DISKSPACE:
    logger.info("Shut down without recording | Free disk space left: %s MB\n", disk_free)
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
    time.sleep(5)  # wait 5 seconds for Raspberry Pi to shut down

# Get detection model metadata from config JSON
with CONFIG_PATH.open(encoding="utf-8") as config_json:
    config = json.load(config_json)
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

# Create folders for each object class to save cropped detections
for det_class in labels:
    (save_path / f"crop/{det_class}").mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2)     # downscale 4K to 1080p resolution -> HQ frames
cam_rgb.setPreviewSize(320, 320)  # downscale frames for model input -> LQ frames
cam_rgb.setPreviewKeepAspectRatio(False)  # stretch frames (16:9) to square (1:1) for model input
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure and model input

# Get sensor resolution
SENSOR_RES = cam_rgb.getResolutionSize()

# Create detection network node and define input
nn = pipeline.create(dai.node.YoloDetectionNetwork)
cam_rgb.preview.link(nn.input)  # downscaled LQ frames as model input
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
#tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)  # better for low fps
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
nn.passthrough.link(tracker.inputTrackerFrame)
nn.passthrough.link(tracker.inputDetectionFrame)
nn.out.link(tracker.inputDetections)

# Create and configure sync node and define inputs
sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=200))
cam_rgb.video.link(sync.inputs["frames"])  # HQ frames
tracker.out.link(sync.inputs["tracker"])   # tracker output

# Create message demux node and define input + outputs
demux = pipeline.create(dai.node.MessageDemux)
sync.out.link(demux.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
demux.outputs["frames"].link(xout_rgb.input)  # synced HQ frames

xout_tracker = pipeline.create(dai.node.XLinkOut)
xout_tracker.setStreamName("track")
demux.outputs["tracker"].link(xout_tracker.input)  # synced tracker output

if args.af_range or args.bbox_ae_region:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)


def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


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


def bbox_set_exposure_region(xmin_roi, ymin_roi, xmax_roi, ymax_roi):
    """Use bounding box coordinates from detections to set auto exposure region."""
    xmin_roi = max(0.001, xmin_roi)
    ymin_roi = max(0.001, ymin_roi)
    xmax_roi = min(0.999, xmax_roi)
    ymax_roi = min(0.999, ymax_roi)

    roi_x = int(xmin_roi * SENSOR_RES[0])
    roi_y = int(ymin_roi * SENSOR_RES[1])
    roi_width = int((xmax_roi - xmin_roi) * SENSOR_RES[0])
    roi_height = int((ymax_roi - ymin_roi) * SENSOR_RES[1])

    ae_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_width, roi_height)
    q_ctrl.send(ae_ctrl)


def make_bbox_square(frame, bbox):
    """Increase bbox size on both sides of the minimum dimension,
    or only on one side if localized at frame margin.
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_diff = abs(bbox_width - bbox_height) // 2

    if bbox_width < bbox_height:
        if bbox[0] - bbox_diff < 0:
            bbox[0] = 0
            bbox[2] = bbox[2] + bbox_diff * 2 - bbox[0]
        elif bbox[2] + bbox_diff > frame.shape[1]:
            bbox[0] = bbox[0] - bbox_diff * 2 + frame.shape[1] - bbox[2]
            bbox[2] = frame.shape[1]
        else:
            bbox[0] = bbox[0] - bbox_diff
            bbox[2] = bbox[2] + bbox_diff
    else:
        if bbox[1] - bbox_diff < 0:
            bbox[1] = 0
            bbox[3] = bbox[3] + bbox_diff * 2 - bbox[1]
        elif bbox[3] + bbox_diff > frame.shape[0]:
            bbox[1] = bbox[1] - bbox_diff * 2 + frame.shape[0] - bbox[3]
            bbox[3] = frame.shape[0]
        else:
            bbox[1] = bbox[1] - bbox_diff
            bbox[3] = bbox[3] + bbox_diff
    return bbox


def save_crop_metadata(frame, bbox):
    """Save detections cropped from HQ frame to .jpg and corresponding metadata to .csv."""
    if args.crop_bbox == "square":
        bbox = make_bbox_square(frame, bbox)
    det_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    timestamp_crop = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
    path_crop = f"{save_path}/crop/{label}/{timestamp_crop}_{track_id}_crop.jpg"
    cv2.imwrite(path_crop, det_crop)

    data = {
        "rec_ID": rec_id,
        "timestamp": timestamp_crop,
        "label": label,
        "confidence": det_conf,
        "track_ID": track_id,
        "x_min": round(xmin, 4),
        "y_min": round(ymin, 4),
        "x_max": round(xmax, 4),
        "y_max": round(ymax, 4),
        "file_path": path_crop
    }

    with open(save_path / f"metadata_{rec_start}.csv", "a", encoding="utf-8") as metadata_file:
        metadata = csv.DictWriter(metadata_file, fieldnames=[
            "rec_ID", "timestamp", "label", "confidence", "track_ID",
            "x_min", "y_min", "x_max", "y_max", "file_path"
        ])
        if metadata_file.tell() == 0:
            metadata.writeheader()  # write header only once
        metadata.writerow(data)
        metadata_file.flush()  # write data immediately to .csv to avoid potential data loss


def save_raw_frame(frame):
    """Save full raw HQ frame to .jpg."""
    timestamp_raw = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
    path_raw = f"{save_path}/raw/{timestamp_raw}_raw.jpg"
    cv2.imwrite(path_raw, frame)


def save_overlay_frame(frame, bbox, track):
    """Save full HQ frame with overlays (bounding box, label, confidence, tracking ID) to .jpg."""
    text_pos = (28, 55, 92) if not args.four_k_resolution else (48, 98, 164)
    font_size = (0.9, 0.8, 1.1) if not args.four_k_resolution else (1.7, 1.6, 2)
    thickness = 2 if not args.four_k_resolution else 3

    cv2.putText(frame, label, (bbox[0], bbox[3] + text_pos[0]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size[0], (255, 255, 255), thickness)
    cv2.putText(frame, f"{det_conf}", (bbox[0], bbox[3] + text_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size[1], (255, 255, 255), thickness)
    cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[3] + text_pos[2]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size[2], (255, 255, 255), thickness)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness)

    if track == tracks[-1]:
        timestamp_overlay = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
        path_overlay = f"{save_path}/overlay/{timestamp_overlay}_overlay.jpg"
        cv2.imwrite(path_overlay, frame)


def save_logs():
    """Write recording ID, time, RPi CPU + OAK chip temperature and
    RPi available memory (MB) + CPU utilization (%) to .csv.
    """
    try:
        temp_oak = round(device.getChipTemperature().average)
    except RuntimeError:
        temp_oak = "NA"

    try:
        logs = {
            "rec_ID": rec_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H-%M-%S"),
            "temp_pi": round(CPUTemperature().temperature),
            "temp_oak": temp_oak,
            "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
            "pi_cpu_used": psutil.cpu_percent(interval=None)
        }
    except IndexError:
        logs = {}

    with open(save_path.parent / f"info_log_{rec_start[:8]}.csv", "a", encoding="utf-8") as log_file:
        log_info = csv.DictWriter(log_file, fieldnames=[
            "rec_ID", "timestamp", "temp_pi", "temp_oak", "pi_mem_available", "pi_cpu_used"
        ])
        if log_file.tell() == 0:
            log_info.writeheader()
        log_info.writerow(logs)
        log_file.flush()


def record_log():
    """Write information about each recording interval to .csv file."""
    try:
        df_meta = pd.read_csv(save_path / f"metadata_{rec_start}.csv", encoding="utf-8")
        unique_ids = df_meta["track_ID"].nunique()
    except pd.errors.EmptyDataError:
        unique_ids = 0

    logs_rec = {
        "rec_ID": rec_id,
        "rec_start_date": rec_start[:8],
        "rec_start_time": rec_start[9:],
        "rec_end_time": datetime.now().strftime("%H-%M"),
        "rec_time_min": round((time.monotonic() - start_time) / 60, 2),
        "num_crops": len(list((save_path / "crop").glob("**/*.jpg"))),
        "num_IDs": unique_ids,
        "disk_free_gb": round(psutil.disk_usage("/").free / 1073741824, 1)
    }

    with open(save_path.parents[1] / "record_log.csv", "a", encoding="utf-8") as log_rec_file:
        log_rec = csv.DictWriter(log_rec_file, fieldnames=[
            "rec_ID", "rec_start_date", "rec_start_time", "rec_end_time",
            "rec_time_min", "num_crops", "num_IDs", "disk_free_gb"
        ])
        if log_rec_file.tell() == 0:
            log_rec.writeheader()
        log_rec.writerow(logs_rec)


def save_zip():
    """Store all captured data in an uncompressed .zip
    file for each day and delete original folder."""
    with ZipFile(f"{save_path.parent}.zip", "a") as zip_file:
        for file in save_path.rglob("*"):
            zip_file.write(file, file.relative_to(save_path.parent))

    shutil.rmtree(save_path.parent, ignore_errors=True)


# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    if args.save_logs:
        # Write RPi + OAK info to .csv log file at specified interval
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        scheduler = BackgroundScheduler()
        scheduler.add_job(save_logs, "interval", seconds=LOG_FREQ, id="log")
        scheduler.start()

    # Write info on start of recording to log file
    logger.info("Rec ID: %s | Rec time: %s min", rec_id, int(REC_TIME / 60))

    # Create output queues to get the frames and tracklets (+ detections) from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    if args.af_range or args.bbox_ae_region:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

    if args.af_range:
        # Set auto focus range to specified cm values
        set_focus_range()

    # Set start time of recording
    start_time = time.monotonic()

    try:
        # Record until recording time is finished
        # Stop recording early if free disk space drops below threshold
        while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE:

            # Update free disk space (MB)
            disk_free = round(psutil.disk_usage("/").free / 1048576)

            # Get synchronized HQ frame + tracker output (including passthrough detections)
            if q_frame.has() and q_track.has():
                frame_hq = q_frame.get().getCvFrame()
                tracks = q_track.get().tracklets

                for tracklet in tracks:
                    # Only use tracklets that are currently tracked (not "NEW", "LOST" or "REMOVED")
                    if tracklet.status.name == "TRACKED":
                        # Get bounding box from passthrough detections
                        xmin, ymin = tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin
                        xmax, ymax = tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax
                        bbox_det = frame_norm(frame_hq, (xmin, ymin, xmax, ymax))

                        # Get metadata from tracker output (including passthrough detections)
                        label = labels[tracklet.srcImgDetection.label]
                        det_conf = round(tracklet.srcImgDetection.confidence, 2)
                        track_id = tracklet.id

                        if args.bbox_ae_region and tracklet == tracks[-1]:
                            # Use model bbox from latest tracking ID to set auto exposure region
                            bbox_set_exposure_region(xmin, ymin, xmax, ymax)

                        # Save detections cropped from HQ frame together with metadata
                        save_crop_metadata(frame_hq, bbox_det)

                        if args.save_raw_frames and tracklet == tracks[0]:
                            # Save full raw HQ frame
                            save_raw_frame(frame_hq)

                        if args.save_overlay_frames:
                            # Save full HQ frame with overlays
                            save_overlay_frame(frame_hq, bbox_det, tracklet)

            # Wait for specified amount of seconds (default: 1)
            time.sleep(CAPTURE_FREQ)

        # Write info on end of recording to log file and write record logs to .csv
        logger.info("Recording %s finished\n", rec_id)
        record_log()

        if args.save_zip:
            # Store data in uncompressed .zip file and delete original folder
            save_zip()

        # Shut down Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

    # Write info on error during recording to log file and write record logs to .csv
    except Exception:
        logger.error(traceback.format_exc())
        logger.error("Error during recording %s\n", rec_id)
        record_log()

        if args.save_zip:
            # Store data in uncompressed .zip file and delete original folder
            save_zip()

        # Shut down Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
