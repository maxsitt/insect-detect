#!/usr/bin/env python3

"""Save HQ frame + associated metadata from OAK camera if object is detected and tracked.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- write info and error (+ traceback) messages to log file
- shut down Raspberry Pi without recording if free disk space or battery charge level
  is lower than the specified threshold (default: 1000 MB | 20%)
- duration of each recording interval conditional on current battery charge level
  -> increases efficiency of battery usage and can prevent gaps in recordings
- create a directory for each day and recording interval where images + metadata + logs are stored
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled + stretched/cropped LQ frames (default: 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on object motion speed and inference speed of the detection model
- synchronize tracker output (including model output) from inference on LQ frames with
  MJPEG-encoded HQ frames (default: 3840x2160 px) on-device using the respective timestamps
  -> maximum pipeline speed (including saving HQ frames):
     full FOV (16:9):    ~19 FPS (3840x2160) | ~42 FPS (1920x1080)
     reduced FOV (~1:1): ~29 FPS (2176x2160) | ~42 FPS (1088x1080)
- save MJPEG-encoded HQ frames to .jpg at specified intervals if object is detected
  and tracked (default: 1 s) and independent of detections (default: 600 s)
- save corresponding metadata from tracker and model output (time, label, confidence,
  tracking ID, tracking status, relative bbox coordinates) to metadata .csv file
- write info about recording interval (rec ID, start/end time, duration,
  number of unique tracking IDs, free disk space, battery charge level) to 'record_log.csv' file
- shut down Raspberry Pi after recording interval is finished or if free disk space
  or battery charge level drop below the specified threshold or if an error occurs
- optional arguments:
  '-res'  set camera resolution for HQ frames
          default:  4K resolution    -> 3840x2160 px, cropped from 12MP  ('-res 4k')
          optional: 1080p resolution -> 1920x1080 px, downscaled from 4K ('-res 1080p')
  '-fov'  default:  stretch frames to square for model input ('-fov stretch')
                    -> FOV is preserved, only aspect ratio of LQ frames is changed (adds distortion)
                    -> HQ frame resolution: 3840x2160 px (default) or 1920x1080 px ('-res 1080p')
          optional: crop frames to square for model input ('-fov crop')
                    -> FOV is reduced due to cropping of LQ and HQ frames (no distortion)
                    -> HQ frame resolution: 2176x2160 px (default) or 1088x1080 px ('-res 1080p')
  '-cpi'  set capture interval in seconds at which HQ frame + associated metadata is saved
          if object is currently detected and tracked (default: 1)
          -> e.g. '-cpi 0.1' for 0.1 s interval (~10 FPS) or '-cpi 3' for 3 s interval (~0.33 FPS)
  '-tli'  set time lapse interval in seconds at which HQ frame is saved
          independent of detected and tracked objects (default: 600)
          -> e.g. '-tli 60' for 1 min time lapse interval
          -> can be used to capture images for training data collection or as control mechanism
  '-af'   set auto focus range in cm (min - max distance to camera)
          -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-mf'   set manual focus position in cm (distance to camera)
          -> e.g. '-mf 14' to set manual focus position to 14 cm
  '-ae'   use bounding box coordinates from detections to set auto exposure region
          -> can improve image quality of detections in certain lighting conditions
  '-log'  write RPi CPU + OAK chip temperature, RPi available memory (MB) +
          CPU utilization (%) and battery info to .csv file at specified interval (default: 30 s)
  '-post' set post-processing method(s) for saved HQ frames
          -> e.g. '-post crop delete' to save cropped detections and delete original HQ frames
          -> several methods can be combined ('delete' requires 'crop' or 'overlay')
             'crop':    crop detections from HQ frames and save as individual .jpg images
             'overlay': draw overlays (bbox + info) on HQ frames and save copy as .jpg image
             'delete':  delete original HQ frames after processing
          -> increases on-device processing time and power consumption
  '-crop' default:  save cropped detections with aspect ratio 1:1 ('-crop square')
                    -> increases bbox size on both sides of the minimum dimension,
                       or only on one side if object is localized at frame margin
                    -> can increase classification accuracy by avoiding stretching of the
                       cropped detection during resizing for classification inference
          optional: keep original bbox size with variable aspect ratio ('-crop tight')
                    -> no additional background information is added but cropped detection
                       can be stretched during resizing for classification inference
  '-arx'  archive all captured data + logs and manage disk space
          -> increases file transfer speed (microSD to computer or upload to cloud)
          -> increases on-device processing time and power consumption
  '-ul'   upload archived data to cloud storage provider using Rclone
          -> increases on-device processing time and power consumption

partly based on open source scripts available at https://github.com/luxonis
"""

import argparse
import csv
import json
import logging
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import depthai as dai
import psutil
from apscheduler.schedulers.background import BackgroundScheduler

from utils.general import archive_data, create_signal_handler, save_encoded_frame, upload_data
from utils.log import record_log, save_logs
from utils.oak_cam import convert_bbox_roi, convert_cm_lens_position
from utils.post import process_images
from utils.wittypi import WittyPiStatus

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-res", "--resolution", type=str, choices=["4k", "1080p"], default="4k",
    help="Set camera resolution (default: 4k).")
parser.add_argument("-fov", "--field_of_view", type=str, choices=["stretch", "crop"], default="stretch",
    help=("Stretch frames to square and preserve FOV ('stretch') or "
          "crop frames to square and reduce FOV ('crop') (default: 'stretch')."))
parser.add_argument("-cpi", "--capture_interval", type=float, default=1, metavar="SECONDS",
    help=("Set time interval in seconds at which HQ frame + associated metadata "
          "is saved if object is currently detected and tracked (default: 1)."))
parser.add_argument("-tli", "--timelapse_interval", type=float, default=600, metavar="SECONDS",
    help=("Set time interval in seconds at which HQ frame is saved "
          "independent of detected and tracked objects (default: 600)."))
group.add_argument("-af", "--auto_focus_range", type=int, nargs=2, metavar=("CM_MIN", "CM_MAX"),
    help="Set auto focus range in cm (min - max distance to camera).")
group.add_argument("-mf", "--manual_focus", type=int, metavar="CM",
    help="Set manual focus position in cm (distance to camera).")
parser.add_argument("-ae", "--auto_exposure_region", action="store_true",
    help="Use bounding box coordinates from detections to set auto exposure region.")
parser.add_argument("-log", "--save_logs", action="store_true",
    help=("Write RPi CPU + OAK chip temperature, RPi available memory (MB) + "
          "CPU utilization (%%) and battery info to .csv file."))
parser.add_argument("-post", "--post_processing", type=str, nargs="+", choices=["crop", "overlay", "delete"],
    help="Set post-processing method(s) for saved HQ frames.", metavar="METHOD")
parser.add_argument("-crop", "--crop_method", type=str, choices=["square", "tight"], default="square",
    help=("Save cropped detections with aspect ratio 1:1 ('square') or "
          "keep original bbox size with variable aspect ratio ('tight') (default: 'square')."))
parser.add_argument("-arx", "--archive_data", action="store_true",
    help="Archive all captured data + logs and manage disk space.")
parser.add_argument("-ul", "--upload_data", action="store_true",
    help="Upload archived data to cloud storage provider.")
args = parser.parse_args()

# Set path to directory where all captured data will be stored (images + metadata + logs)
DATA_PATH = Path.home() / "insect-detect" / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Set file paths to the detection model and corresponding config JSON
MODEL_PATH = Path.home() / "insect-detect" / "models" / "yolov5n_320_openvino_2022.1_4shave.blob"
CONFIG_PATH = Path.home() / "insect-detect" / "models" / "json" / "yolov5_v7_320.json"

# Set camera trap ID
CAM_ID = socket.gethostname()  # default: hostname

# Set camera frame rate
FPS = 20  # default: 20 FPS

# Set minimum free disk space threshold required to start and continue a recording
MIN_DISKSPACE = 1000  # default: 1000 MB

# Set low free disk space threshold up to which no original data will be removed if "-arx" is used
LOW_DISKSPACE = 5000  # default: 5000 MB

# Set minimum battery charge level threshold required to start and continue a recording
MIN_CHARGELEVEL = 20  # default: 20%

# Set time interval at which logs are saved to .csv file if "-log" is used
LOG_INT = 30  # default: 30 seconds

# Set time intervals at which free disk space and charge level are checked during recording
FREE_SPACE_INT = 30    # default: 30 seconds
CHARGE_LEVEL_INT = 30  # default: 30 seconds

# Set time interval at which HQ frame + metadata is saved if object is detected and tracked
CAPTURE_INT = args.capture_interval  # default: 1 second

# Set time interval at which HQ frame is saved independent of detected and tracked objects
TIMELAPSE_INT = args.timelapse_interval  # default: 600 seconds (= 10 minutes)

# Set logging level and format, write logs to file
script_name = Path(__file__).stem
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
                    filename=f"{DATA_PATH}/{script_name}_log.log", encoding="utf-8")
logger = logging.getLogger()
logger.info("-------- Logger initialized --------")
logging.getLogger("apscheduler").setLevel(logging.WARNING)  # decrease apscheduler logging level

# Handle SIGTERM signal (e.g. from external shutdown trigger)
external_shutdown = threading.Event()
signal.signal(signal.SIGTERM, create_signal_handler(external_shutdown))

# Initialize Witty Pi 4 L3V7
wittypi = WittyPiStatus()

# Shut down Raspberry Pi if free disk space (MB) or battery charge level is lower than threshold
disk_free = round(psutil.disk_usage("/").free / 1048576)
chargelevel_start = wittypi.estimate_chargelevel()
if disk_free < MIN_DISKSPACE or (chargelevel_start != "USB_C_IN" and chargelevel_start < MIN_CHARGELEVEL):
    logger.info("Shut down without recording | Free disk space: %s MB | Charge level: %s%%",
                disk_free, chargelevel_start)
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

# Set recording time conditional on battery charge level
if chargelevel_start == "USB_C_IN":
    REC_TIME = 60 * 40             # power from USB C:   40 min
elif chargelevel_start >= 70:
    REC_TIME = 60 * 30             # charge level > 70:  30 min
elif 50 <= chargelevel_start < 70:
    REC_TIME = 60 * 20             # charge level 50-70: 20 min
elif 30 <= chargelevel_start < 50:
    REC_TIME = 60 * 10             # charge level 30-50: 10 min
else:
    REC_TIME = 60 * 5              # charge level < 30:   5 min

# Get last recording ID from text file and increment by 1 (create text file for first recording)
rec_id_file = DATA_PATH / "last_rec_id.txt"
rec_id = int(rec_id_file.read_text(encoding="utf-8")) + 1 if rec_id_file.exists() else 1
rec_id_file.write_text(str(rec_id), encoding="utf-8")

# Create directory per day (date) and recording interval (datetime) to save images + metadata + logs
timestamp_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = DATA_PATH / timestamp_dir[:10] / timestamp_dir
save_path.mkdir(parents=True, exist_ok=True)

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

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure and model input
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if args.resolution == "1080p":
    cam_rgb.setIspScale(1, 2)     # downscale 4K to 1080p resolution -> HQ frames (16:9)
cam_rgb.setPreviewSize(320, 320)  # downscale frames for model input -> LQ frames (1:1)
if args.field_of_view == "stretch":
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square for model input
elif args.field_of_view == "crop" and args.resolution == "4k":
    cam_rgb.setVideoSize(2176, 2160)  # crop LQ and HQ frames to square
elif args.field_of_view == "crop" and args.resolution == "1080p":
    cam_rgb.setVideoSize(1088, 1080)  # width must be multiple of 32 for MJPEG encoder
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
SENSOR_RES = cam_rgb.getResolutionSize()

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

# Create and configure video encoder node and define input
encoder = pipeline.create(dai.node.VideoEncoder)
encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
encoder.setQuality(80)  # JPEG quality (0-100)
cam_rgb.video.link(encoder.input)  # HQ frames as encoder input

# Create and configure YOLO detection network node and define input
yolo = pipeline.create(dai.node.YoloDetectionNetwork)
yolo.setBlobPath(MODEL_PATH)
yolo.setNumClasses(classes)
yolo.setCoordinateSize(coordinates)
yolo.setAnchors(anchors)
yolo.setAnchorMasks(anchor_masks)
yolo.setIouThreshold(iou_threshold)
yolo.setConfidenceThreshold(confidence_threshold)
yolo.setNumInferenceThreads(2)
cam_rgb.preview.link(yolo.input)  # downscaled + stretched/cropped LQ frames as model input
yolo.input.setBlocking(False)     # non-blocking input stream

# Create and configure object tracker node and define inputs
tracker = pipeline.create(dai.node.ObjectTracker)
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
yolo.passthrough.link(tracker.inputTrackerFrame)  # passthrough LQ frames as tracker input
yolo.passthrough.link(tracker.inputDetectionFrame)
yolo.out.link(tracker.inputDetections)            # detections from YOLO model as tracker input

# Create and configure sync node and define inputs
sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=100))
encoder.bitstream.link(sync.inputs["frames"])  # HQ frames (MJPEG-encoded bitstream)
tracker.out.link(sync.inputs["tracker"])       # tracker + model output

# Create message demux node and define input + outputs
demux = pipeline.create(dai.node.MessageDemux)
sync.out.link(demux.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
demux.outputs["frames"].link(xout_rgb.input)       # synced HQ frames

xout_tracker = pipeline.create(dai.node.XLinkOut)
xout_tracker.setStreamName("track")
demux.outputs["tracker"].link(xout_tracker.input)  # synced tracker + model output

if args.auto_exposure_region:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

try:
    # Connect to OAK device and start pipeline in USB2 mode
    with (dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device,
          open(save_path / f"{timestamp_dir}_metadata.csv", "a", encoding="utf-8") as metadata_file,
          ThreadPoolExecutor(max_workers=3) as executor):

        # Create output queues to get the synchronized HQ frames and tracker + model output
        q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
        q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

        if args.auto_exposure_region:
            # Create input queue to send control commands to OAK camera
            q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False)

        # Write header to metadata .csv file
        metadata_writer = csv.DictWriter(metadata_file, fieldnames=[
            "cam_ID", "rec_ID", "timestamp", "label", "confidence",
            "track_ID", "track_status", "x_min", "y_min", "x_max", "y_max"
        ])
        metadata_writer.writeheader()

        if args.save_logs:
            # Write RPi + OAK + battery info to .csv file at specified interval
            scheduler = BackgroundScheduler()
            scheduler.add_job(save_logs, "interval", seconds=LOG_INT, id="log",
                              args=[save_path, CAM_ID, rec_id, device, wittypi],
                              next_run_time=datetime.now() + timedelta(seconds=2))
            scheduler.start()

        # Wait for 2 seconds to let camera adjust auto focus and exposure
        time.sleep(2)

        # Write info on start of recording to log file
        logger.info("Cam ID: %s | Rec ID: %s | Rec time: %s min | Charge level: %s%%",
                    CAM_ID, rec_id, int(REC_TIME / 60), chargelevel_start)

        # Create variables for start of recording and capture/check events
        rec_start = datetime.now()
        start_time = time.monotonic()
        last_capture = start_time - TIMELAPSE_INT  # capture first frame immediately at start
        next_capture = start_time + CAPTURE_INT
        last_disk_check = start_time
        last_charge_check = start_time
        chargelevel = chargelevel_start
        chargelevels = []

        try:
            # Record until recording time is finished
            # Stop recording early if free disk space drops below threshold
            # or if charge level dropped below threshold for 5 times
            while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE and len(chargelevels) < 5:

                # Activate trigger to save HQ frame based on current time and specified intervals
                track_active = False
                current_time = time.monotonic()
                trigger_capture = current_time >= next_capture
                trigger_timelapse = current_time - last_capture >= TIMELAPSE_INT

                if q_frame.has() and (trigger_capture or trigger_timelapse):
                    # Get MJPEG-encoded HQ frame (synced with tracker output)
                    timestamp = datetime.now()
                    timestamp_iso = timestamp.isoformat()
                    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")
                    frame_hq = q_frame.get().getData()

                    if q_track.has():
                        # Get tracker output (including passthrough model output)
                        tracklets = q_track.get().tracklets
                        for tracklet in tracklets:
                            # Check if tracklet is active (not "LOST" or "REMOVED")
                            tracklet_status = tracklet.status.name
                            if tracklet_status in {"TRACKED", "NEW"}:
                                track_active = True

                                # Get bounding box from model output
                                bbox = (tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin,
                                        tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax)

                                # Get metadata from tracker + model output and save to .csv file
                                metadata = {
                                    "cam_ID": CAM_ID,
                                    "rec_ID": rec_id,
                                    "timestamp": timestamp_iso,
                                    "label": labels[tracklet.srcImgDetection.label],
                                    "confidence": round(tracklet.srcImgDetection.confidence, 2),
                                    "track_ID": tracklet.id,
                                    "track_status": tracklet_status,
                                    "x_min": round(bbox[0], 4),
                                    "y_min": round(bbox[1], 4),
                                    "x_max": round(bbox[2], 4),
                                    "y_max": round(bbox[3], 4)
                                }
                                metadata_writer.writerow(metadata)

                                if args.auto_exposure_region and tracklet_status == "TRACKED" and tracklet is tracklets[-1]:
                                    # Use model bbox from latest active tracking ID to set auto exposure region
                                    roi_x, roi_y, roi_w, roi_h = convert_bbox_roi(bbox, SENSOR_RES)
                                    q_ctrl.send(dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h))

                    if track_active or trigger_timelapse:
                        # Save MJPEG-encoded HQ frame to .jpg file in separate thread
                        executor.submit(save_encoded_frame, save_path, timestamp_str, frame_hq)
                        last_capture = current_time
                        next_capture = current_time + CAPTURE_INT

                        # Update free disk space (MB) at specified interval
                        if current_time - last_disk_check >= FREE_SPACE_INT:
                            disk_free = round(psutil.disk_usage("/").free / 1048576)
                            last_disk_check = current_time

                # Update charge level at specified interval and add to list if lower than threshold
                if current_time - last_charge_check >= CHARGE_LEVEL_INT:
                    chargelevel = wittypi.estimate_chargelevel()
                    if chargelevel != "USB_C_IN" and chargelevel < MIN_CHARGELEVEL:
                        chargelevels.append(chargelevel)
                    last_charge_check = current_time

                # Sleep for a short duration to avoid busy waiting
                time.sleep(0.02)

            # Write info on end of recording to log file
            logger.info("Recording %s finished | Free disk space: %s MB | Charge level: %s%%",
                        rec_id, disk_free, chargelevel)

        except Exception:
            logger.exception("Error during recording %s | Charge level: %s%%", rec_id, chargelevel)
        finally:
            # Write recording logs to .csv file
            rec_end = datetime.now()
            record_log(save_path, CAM_ID, rec_id, rec_start, rec_end, chargelevel_start, chargelevel)

            if "scheduler" in locals():
                # Shut down scheduler (wait until currently executing jobs are finished)
                scheduler.shutdown()

    if args.post_processing:
        if disk_free > MIN_DISKSPACE and (chargelevel == "USB_C_IN" or chargelevel > MIN_CHARGELEVEL + 10):
            try:
                # Post-process saved HQ frames based on specified methods
                if any(save_path.glob("*.jpg")):
                    processing_methods = set(args.post_processing)
                    required_methods = {"crop", "overlay"}
                    if required_methods.intersection(processing_methods):
                        process_images(save_path, processing_methods, args.crop_method)
                    logger.info("Post-processing of saved HQ frames finished")
            except Exception:
                logger.exception("Error during post-processing of saved HQ frames")
        else:
            logger.info("Shut down without post-processing | Free disk space: %s MB | Charge level: %s%%",
                        disk_free, chargelevel)

    if args.archive_data or args.upload_data:
        if disk_free > MIN_DISKSPACE and (chargelevel == "USB_C_IN" or chargelevel > MIN_CHARGELEVEL + 10):
            try:
                # Archive all captured data + logs and manage disk space
                archive_path = archive_data(DATA_PATH, CAM_ID, LOW_DISKSPACE)
                if args.upload_data:
                    # Upload archived data to cloud storage provider
                    upload_data(DATA_PATH, archive_path)
                logger.info("Archiving/uploading of data finished")
            except Exception:
                logger.exception("Error during archiving/uploading of data")
        else:
            logger.info("Shut down without archiving/uploading | Free disk space: %s MB | Charge level: %s%%",
                        disk_free, chargelevel)

except SystemExit:
    logger.info("Recording %s stopped by external trigger | Charge level: %s%%", rec_id, chargelevel)
except KeyboardInterrupt:
    logger.info("Recording %s stopped by Ctrl+C", rec_id)
except Exception:
    logger.exception("Error during initialization of recording %s | Charge level: %s%%", rec_id, chargelevel)
finally:
    if not external_shutdown.is_set():
        # Shut down Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
