#!/usr/bin/env python3

"""Save cropped detections with associated metadata from detection model and object tracker.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- write info and error (+ traceback) messages to log file
- shut down Raspberry Pi without recording if free disk space or current Witty Pi
  battery charge level are lower than the specified thresholds (default: 100 MB and 20%)
- duration of each recording interval conditional on current Witty Pi battery charge level
  -> increases efficiency of battery usage and can prevent gaps in recordings
- create directory for each day, recording interval and object class to save images + metadata
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled + stretched/cropped LQ frames (default: 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on object motion speed and inference speed of the detection model
- synchronize tracker output (including detections) from inference on LQ frames with
  HQ frames (default: 1920x1080 px) on-device using the respective message timestamps
  -> pipeline speed (= inference speed): ~13.4 fps (1080p sync) or ~3.4 fps (4K sync) for full FOV
                                         ~23 fps (1080x1080) or ~5.8 fps (2160x2160) for reduced FOV
- save detections (bounding box area) cropped from HQ frames to .jpg at the
  specified capture frequency (default: 1 s), optionally together with full frames
- save corresponding metadata from tracker (+ model) output (time, label, confidence,
  tracking ID, relative bbox coordinates, .jpg file path) to .csv
- write info about recording interval (rec ID, start/end time, duration, number of cropped
  detections, unique tracking IDs, free disk space, battery charge level) to 'record_log.csv'
- shut down Raspberry Pi after recording interval is finished or if charge level or
  free disk space drop below the specified thresholds or if an error occurs
- optional arguments:
  '-4k'      crop detections from (+ save HQ frames in) 4K resolution (default: 1080p)
             -> decreases pipeline speed to ~3.4 fps (1080p: ~13.4 fps)
  '-fov'     default:  stretch frames to square for model input ('-fov stretch')
                       -> full FOV is preserved, only aspect ratio is changed (adds distortion)
                       -> HQ frame resolution: 1920x1080 px (default) or 3840x2160 px ('-4k')
             optional: crop frames to square for model input ('-fov crop')
                       -> FOV is reduced due to cropping of left and right side (no distortion)
                       -> HQ frame resolution: 1080x1080 px (default) or 2160x2160 px ('-4k')
                       -> increases pipeline speed to ~23 fps (4K: ~5.8 fps)
  '-af'      set auto focus range in cm (min distance, max distance)
             -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-ae'      use bounding box coordinates from detections to set auto exposure region
             -> can improve image quality of crops and thereby classification accuracy
  '-crop'    default:  save cropped detections with aspect ratio 1:1 ('-crop square') OR
             optional: keep original bbox size with variable aspect ratio ('-crop tight')
             -> '-crop square' increases bbox size on both sides of the minimum dimension,
                               or only on one side if object is localized at frame margin
                -> can increase classification accuracy by avoiding stretching of the
                   cropped insect image during resizing for classification inference
  '-full'    additionally save full HQ frames to .jpg (e.g. for training data collection)
             -> '-full det'  save full frame together with cropped detections
                             -> slightly decreases pipeline speed
             -> '-full freq' save full frame at specified frequency (default: 60 s)
  '-overlay' additionally save full HQ frames with overlays (bbox + info) to .jpg
             -> slightly decreases pipeline speed
  '-log'     write RPi CPU + OAK chip temperature, RPi available memory (MB) +
             CPU utilization (%) and battery info to .csv file at specified frequency
  '-archive' archive all captured data + logs and manage disk space
             -> increases file transfer speed (microSD to computer or upload to cloud)
                but also increases on-device processing time and power consumption

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import json
import logging
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import depthai as dai
import psutil
from apscheduler.schedulers.background import BackgroundScheduler

from utils.general import archive_data, create_signal_handler, frame_norm
from utils.log import record_log, save_logs
from utils.oak_cam import bbox_set_exposure_region, set_focus_range
from utils.save_data import save_crop_metadata, save_full_frame, save_overlay_frame
from utils.wittypi import WittyPiStatus

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-4k", "--four_k_resolution", action="store_true",
    help="Set camera resolution to 4K (3840x2160 px) (default: 1080p).")
parser.add_argument("-fov", "--adjust_fov", choices=["stretch", "crop"], default="stretch", type=str,
    help="Stretch frames to square ('stretch') and preserve full FOV or "
         "crop frames to square ('crop') and reduce FOV.")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min distance, max distance).", metavar=("CM_MIN", "CM_MAX"))
parser.add_argument("-ae", "--bbox_ae_region", action="store_true",
    help="Use bounding box coordinates from detections to set auto exposure region.")
parser.add_argument("-crop", "--crop_bbox", choices=["square", "tight"], default="square", type=str,
    help=("Save cropped detections with aspect ratio 1:1 ('square') or "
          "keep original bbox size with variable aspect ratio ('tight')."))
parser.add_argument("-full", "--save_full_frames", choices=["det", "freq"], default=None, type=str,
    help="Additionally save full HQ frames to .jpg together with cropped detections ('det') "
         "or at specified frequency, independent of detections ('freq').")
parser.add_argument("-overlay", "--save_overlay_frames", action="store_true",
    help="Additionally save full HQ frames with overlays (bbox + info) to .jpg.")
parser.add_argument("-log", "--save_logs", action="store_true",
    help=("Write RPi CPU + OAK chip temperature, RPi available memory (MB) + "
          "CPU utilization (%%) and battery info to .csv file."))
parser.add_argument("-archive", "--archive_data", action="store_true",
    help="Archive all captured data + logs and manage disk space.")
args = parser.parse_args()

# Set path to directory where all captured data will be stored (images + metadata + logs)
DATA_PATH = Path.home() / "insect-detect" / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Set file paths to the detection model and corresponding config JSON
MODEL_PATH = Path.home() / "insect-detect" / "models" / "yolov5n_320_openvino_2022.1_4shave.blob"
CONFIG_PATH = Path.home() / "insect-detect" / "models" / "json" / "yolov5_v7_320.json"

# Set threshold values required to start and continue a recording
MIN_DISKSPACE = 100   # minimum free disk space (MB) (default: 100 MB)
MIN_CHARGELEVEL = 20  # minimum Witty Pi battery charge level (default: 20%)

# Set threshold value up to which no original data will be removed if "-archive" is used
LOW_DISKSPACE = 1000  # low free disk space (MB) (default: 1000 MB)

# Set capture frequency (default: 1 second)
# -> wait for specified amount of seconds between saving cropped detections + metadata
CAPTURE_FREQ = 1

# Set frequency for saving full frames if "-full freq" is used (default: 60 seconds)
FULL_FREQ = 60

# Set frequency for saving logs to .csv file if "-log" is used (default: 30 seconds)
LOG_FREQ = 30

# Set camera trap ID (default: hostname)
CAM_ID = socket.gethostname()

# Set logging level and format, write logs to file
script_name = Path(__file__).stem
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
                    filename=f"{DATA_PATH}/{script_name}_log.log", encoding="utf-8")
logger = logging.getLogger()

# Handle SIGTERM signal (e.g. from external shutdown trigger)
external_shutdown = threading.Event()
signal.signal(signal.SIGTERM, create_signal_handler(external_shutdown))

# Instantiate Witty Pi 4 L3V7
wittypi = WittyPiStatus()

# Shut down Raspberry Pi if battery charge level or free disk space (MB) are lower than thresholds
chargelevel_start = wittypi.estimate_chargelevel()
disk_free = round(psutil.disk_usage("/").free / 1048576)
if (chargelevel_start != "USB_C_IN" and chargelevel_start < MIN_CHARGELEVEL) or disk_free < MIN_DISKSPACE:
    logger.info("Shut down without recording | Charge level: %s%%\n", chargelevel_start)
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

# Set recording time conditional on Witty Pi battery charge level
if chargelevel_start == "USB_C_IN":
    REC_TIME = 60 * 40             # Power from main battery (USB C):     40 min
elif chargelevel_start >= 70:
    REC_TIME = 60 * 30             # Witty Pi battery charge level > 70:  30 min
elif 50 <= chargelevel_start < 70:
    REC_TIME = 60 * 20             # Witty Pi battery charge level 50-70: 20 min
elif 30 <= chargelevel_start < 50:
    REC_TIME = 60 * 10             # Witty Pi battery charge level 30-50: 10 min
else:
    REC_TIME = 60 * 5              # Witty Pi battery charge level < 30:   5 min

# Get last recording ID from text file and increment by 1 (create text file for first recording)
rec_id_file = DATA_PATH / "last_rec_id.txt"
rec_id = int(rec_id_file.read_text(encoding="utf-8")) + 1 if rec_id_file.exists() else 1
rec_id_file.write_text(str(rec_id), encoding="utf-8")

# Create directory per day (date) and recording interval (date_time) to save images + metadata + logs
rec_start = datetime.now()
rec_start_str = rec_start.strftime("%Y-%m-%d_%H-%M-%S")
save_path = DATA_PATH / rec_start_str[:10] / rec_start_str
save_path.mkdir(parents=True, exist_ok=True)
if args.save_full_frames is not None:
    (save_path / "full").mkdir(parents=True, exist_ok=True)
if args.save_overlay_frames:
    (save_path / "overlay").mkdir(parents=True, exist_ok=True)

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
    (save_path / "crop" / f"{det_class}").mkdir(parents=True, exist_ok=True)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
if not args.four_k_resolution:
    cam_rgb.setIspScale(1, 2)     # downscale 4K to 1080p resolution -> HQ frames
cam_rgb.setPreviewSize(320, 320)  # downscale frames for model input -> LQ frames
if args.adjust_fov == "stretch":
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch frames (16:9) to square (1:1) for model input
elif args.adjust_fov == "crop" and not args.four_k_resolution:
    cam_rgb.setVideoSize(1080, 1080)  # crop HQ frames to square
elif args.adjust_fov == "crop" and args.four_k_resolution:
    cam_rgb.setVideoSize(2160, 2160)
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure and model input

# Get sensor resolution
SENSOR_RES = cam_rgb.getResolutionSize()

# Create detection network node and define input
nn = pipeline.create(dai.node.YoloDetectionNetwork)
cam_rgb.preview.link(nn.input)  # downscaled + stretched/cropped LQ frames as model input
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

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    if args.save_logs or (args.save_full_frames == "freq"):
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        scheduler = BackgroundScheduler()
    else:
        scheduler = None

    if args.save_logs:
        # Write RPi + OAK + battery info to .csv file at specified frequency
        scheduler.add_job(save_logs, "interval", seconds=LOG_FREQ, id="log",
                          args=[CAM_ID, rec_id, device, rec_start_str, save_path, wittypi])
        scheduler.start()

    if args.save_full_frames == "freq":
        # Save full HQ frame at specified frequency
        scheduler.add_job(save_full_frame, "interval", seconds=FULL_FREQ, id="full",
                          args=[None, save_path])
        if not scheduler.running:
            scheduler.start()

    # Write info on start of recording to log file
    logger.info("Cam ID: %s | Rec ID: %s | Rec time: %s min | Charge level: %s%%",
                CAM_ID, rec_id, int(REC_TIME / 60), chargelevel_start)

    # Create output queues to get the frames and tracklets (+ detections) from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    if args.af_range or args.bbox_ae_region:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

    if args.af_range:
        # Set auto focus range to specified cm values
        af_ctrl = set_focus_range(args.af_range[0], args.af_range[1])
        q_ctrl.send(af_ctrl)

    # Set start time of recording and create empty lists to save charge level and threads
    start_time = time.monotonic()
    chargelevel_list = []
    threads = []

    try:
        # Record until recording time is finished
        # Stop recording early if free disk space drops below threshold OR
        # if charge level dropped below threshold for 10 times
        while time.monotonic() < start_time + REC_TIME and disk_free > MIN_DISKSPACE and len(chargelevel_list) < 10:

            # Get synchronized HQ frame + tracker output (including passthrough detections)
            if q_frame.has() and q_track.has():
                frame_hq = q_frame.get().getCvFrame()
                tracks = q_track.get().tracklets

                if args.save_full_frames == "freq":
                    # Save full HQ frame at specified frequency
                    scheduler.modify_job("full", args=[frame_hq, save_path])

                if args.save_overlay_frames:
                    # Copy frame for drawing overlays
                    frame_hq_copy = frame_hq.copy()

                for tracklet in tracks:
                    # Only use tracklets that are currently tracked (not "NEW", "LOST" or "REMOVED")
                    if tracklet.status.name == "TRACKED":
                        # Get bounding box from passthrough detections
                        bbox_orig = (tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin,
                                     tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax)
                        bbox_norm = frame_norm(frame_hq, bbox_orig)

                        # Get metadata from tracker output (including passthrough detections)
                        label = labels[tracklet.srcImgDetection.label]
                        det_conf = round(tracklet.srcImgDetection.confidence, 2)
                        track_id = tracklet.id

                        if args.bbox_ae_region and tracklet == tracks[-1]:
                            # Use model bbox from latest tracking ID to set auto exposure region
                            ae_ctrl = bbox_set_exposure_region(bbox_orig, SENSOR_RES)
                            q_ctrl.send(ae_ctrl)

                        # Save detections cropped from HQ frame together with metadata
                        save_crop_metadata(CAM_ID, rec_id, frame_hq, bbox_norm, label, det_conf, track_id,
                                           bbox_orig, rec_start_str, save_path, args.crop_bbox)

                        if args.save_full_frames == "det" and tracklet == tracks[-1]:
                            # Save full HQ frame
                            thread_full = threading.Thread(target=save_full_frame,
                                                           args=(frame_hq, save_path))
                            thread_full.start()
                            threads.append(thread_full)

                        if args.save_overlay_frames:
                            # Save full HQ frame with overlays
                            thread_overlay = threading.Thread(target=save_overlay_frame,
                                                              args=(frame_hq_copy, bbox_norm, label,
                                                                    det_conf, track_id, tracklet, tracks,
                                                                    save_path, args.four_k_resolution))
                            thread_overlay.start()
                            threads.append(thread_overlay)

            # Update free disk space (MB)
            disk_free = round(psutil.disk_usage("/").free / 1048576)

            # Update charge level (add to list if lower than threshold)
            chargelevel = wittypi.estimate_chargelevel()
            if chargelevel != "USB_C_IN" and chargelevel < MIN_CHARGELEVEL:
                chargelevel_list.append(chargelevel)

            # Keep only active threads in list
            threads = [thread for thread in threads if thread.is_alive()]

            # Wait for specified amount of seconds (default: 1)
            time.sleep(CAPTURE_FREQ)

        # Write info on end of recording to log file
        logger.info("Recording %s finished | Charge level: %s%%\n", rec_id, chargelevel)

    except SystemExit:
        # Write info on external shutdown trigger (e.g. button) to log file
        logger.info("Recording %s stopped by external trigger | Charge level: %s%%\n", rec_id, chargelevel)

    except KeyboardInterrupt:
        # Write info on KeyboardInterrupt (Ctrl+C) to log file
        logger.info("Recording %s stopped by Ctrl+C | Charge level: %s%%\n", rec_id, chargelevel)

    except Exception:
        # Write info on error + traceback during recording to log file
        logger.exception("Error during recording %s | Charge level: %s%%", rec_id, chargelevel)

    finally:
        # Shut down scheduler (wait until currently executing jobs are finished)
        if scheduler:
            scheduler.shutdown()

        # Wait for active threads to finish
        for thread in threads:
            thread.join()

        # Write record logs to .csv file
        rec_end = datetime.now()
        record_log(CAM_ID, rec_id, rec_start, rec_start_str, rec_end, save_path,
                   chargelevel_start, chargelevel)

        if args.archive_data:
            # Archive all captured data + logs and manage disk space
            archive_path = archive_data(DATA_PATH, CAM_ID, LOW_DISKSPACE)

        if not external_shutdown.is_set():
            # Shut down Raspberry Pi
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
