"""Save HQ frame from OAK camera + synced metadata from model and tracker if object is detected.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Run this script with the Python interpreter from the virtual environment where you installed the
required packages, e.g. with 'env_insdet/bin/python3 insect-detect/yolo_tracker_save_hqsync.py'.

Modify the "configs/config_custom.yaml" file to change the settings that are used in
this Python script. Refer to the "configs/config_default.yaml" for the default settings.

Optional arguments:
'-config' set path to YAML file that contains all configuration parameters
          -> e.g. '-config configs/config_custom.yaml' to use custom config file

- load YAML file with configuration parameters and JSON file with detection model parameters
- write info, warning and error (+ traceback) messages to log file
- initialize power manager if enabled in config (Witty Pi 4 L3V7 or PiJuice Zero)
- shut down Raspberry Pi without recording if free disk space or
  battery charge level is lower than the configured threshold
- duration of each recording session conditional on current battery charge level (if enabled)
  -> increases efficiency of battery usage and can prevent gaps in recordings
- create a directory for each day and recording session to store images, metadata, logs and configs
- run a custom YOLO object detection model (.blob format) on device (Luxonis OAK)
  -> inference on downscaled + stretched/cropped LQ frames
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on camera fps, inference speed of the detection model and object motion speed
- synchronize tracker output (including model output) from inference on LQ frames
  with MJPEG-encoded HQ frames on device (OAK) using the respective timestamps
  -> maximum pipeline speed (including saving HQ frames):
     full FOV (16:9):    ~19 FPS (3840x2160) | ~42 FPS (1920x1080)
     reduced FOV (~1:1): ~29 FPS (2176x2160) | ~42 FPS (1088x1080)
- save MJPEG-encoded HQ frames to .jpg at configured intervals
  if object is detected (trigger capture) and independent of detections (timelapse capture)
- save corresponding metadata from tracker and model output to metadata .csv file
  (time, label, confidence, tracking ID, tracking status, relative bbox coordinates)
- stop recording session and shut down Raspberry Pi if either:
  - configured recording duration is reached
  - free disk space drops below configured threshold
  - battery charge level drops below configured threshold for three times (if enabled)
  - recording is stopped by external trigger (e.g. button press)
  - error occurs during recording
- write info about recording session to record log .csv file (rec ID, start/end time,
  duration, number of unique tracking IDs, free disk space, battery charge level)
- post-process saved HQ frames based on configured methods (if enabled)
- archive all captured data + logs/configs and manage disk space (if enabled)
- upload archived data to cloud storage provider (if enabled)

partly based on open source scripts available at https://github.com/luxonis
"""

import argparse
import csv
import json
import logging
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

from utils.config import parse_json, parse_yaml
from utils.data import archive_data, save_encoded_frame, upload_data
from utils.log import record_log, save_logs
from utils.oak import convert_bbox_roi, convert_cm_lens_position, create_get_temp_oak
from utils.post import process_images
from utils.power import init_power_manager

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str, default="configs/config_default.yaml",
    help="Set path to YAML file with configuration parameters.")
args = parser.parse_args()

# Set camera trap ID (default: hostname)
CAM_ID = socket.gethostname()

# Set base path (default: "insect-detect" directory)
BASE_PATH = Path.home() / "insect-detect"

# Create directory where all data will be stored (images, metadata, logs, configs)
DATA_PATH = BASE_PATH / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Set logging levels and format, write logs to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
                    filename=f"{DATA_PATH}/yolo_tracker_save_hqsync.log", encoding="utf-8")
logger = logging.getLogger()
logger.info("-------- Logger initialized --------")
logging.getLogger("apscheduler").setLevel(logging.WARNING)  # decrease apscheduler logging level
logging.getLogger("tzlocal").setLevel(logging.ERROR)        # suppress timezone warning

# Parse configuration files
config = parse_yaml(BASE_PATH / args.config)
config_model = parse_json(BASE_PATH / config.detection.model.config)

# Extract some frequently used configuration parameters
PWR_MGMT = config.powermanager.enabled
PWR_MGMT_MODEL = config.powermanager.model if PWR_MGMT else None
CHARGE_MIN = config.powermanager.charge_min if PWR_MGMT else None
CHARGE_CHECK = config.powermanager.charge_check if PWR_MGMT else None
TEMP_OAK_MAX = config.oak.temp_max
TEMP_OAK_CHECK = config.oak.temp_check
DISK_MIN = config.storage.disk_min
DISK_CHECK = config.storage.disk_check
RES_HQ = (config.camera.resolution.width, config.camera.resolution.height)
RES_LQ = (config.detection.resolution.width, config.detection.resolution.height)
CAP_INT_DET = config.recording.capture_interval.detection
CAP_INT_TL = config.recording.capture_interval.timelapse
EXP_REGION = config.detection.exposure_region.enabled

# Initialize power manager (Witty Pi 4 L3V7 or PiJuice Zero - None if disabled)
try:
    get_chargelevel, get_power_info, external_shutdown = init_power_manager(PWR_MGMT_MODEL)
    chargelevel_start = get_chargelevel() if PWR_MGMT else None
except Exception:
    logger.exception("Error during initialization of power manager, disabling power management")
    PWR_MGMT = False
    PWR_MGMT_MODEL, CHARGE_MIN, CHARGE_CHECK, chargelevel_start = None, None, None, None
    get_chargelevel, get_power_info = lambda: None, lambda: {}
    external_shutdown = threading.Event()

# Check free disk space (MB) and battery charge level (%) before starting recording session
disk_free = round(psutil.disk_usage("/").free / 1048576)
if disk_free < DISK_MIN:
    logger.warning("Shut down without recording due to low disk space: %s MB", disk_free)
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
if PWR_MGMT:
    if (chargelevel_start != "USB_C_IN" and chargelevel_start < CHARGE_MIN) or chargelevel_start == "NA":
        logger.warning("Shut down without recording due to low charge level: %s%%", chargelevel_start)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

# Set duration of recording session (*60 to convert from min to s)
if PWR_MGMT:
    if chargelevel_start == "USB_C_IN" or chargelevel_start >= 70:
        REC_TIME = config.recording.duration.battery.high * 60
    elif 50 <= chargelevel_start < 70:
        REC_TIME = config.recording.duration.battery.medium * 60
    elif 30 <= chargelevel_start < 50:
        REC_TIME = config.recording.duration.battery.low * 60
    else:
        REC_TIME = config.recording.duration.battery.minimal * 60
else:
    REC_TIME = config.recording.duration.default * 60

# Get last recording ID from text file and increment by 1 (create text file for first recording)
rec_id_file = DATA_PATH / "last_rec_id.txt"
rec_id = int(rec_id_file.read_text(encoding="utf-8")) + 1 if rec_id_file.exists() else 1
rec_id_file.write_text(str(rec_id), encoding="utf-8")

# Create directory per day (date) and recording session (datetime)
timestamp_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = DATA_PATH / timestamp_dir[:10] / timestamp_dir
save_path.mkdir(parents=True, exist_ok=True)

# Save configurations of the current recording session as JSON files
config_path = save_path / f"{timestamp_dir}_{Path(args.config).stem}.json"
config_model_path = save_path / f"{timestamp_dir}_{Path(config.detection.model.config).stem}.json"
json.dump(config, config_path.open("w", encoding="utf-8"), indent=2)
json.dump(config_model, config_model_path.open("w", encoding="utf-8"), indent=2)

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(config.camera.fps)  # frames per second available for focus/exposure and model input
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
SENSOR_RES = cam_rgb.getResolutionSize()
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

if RES_HQ == (1920, 1080):
    cam_rgb.setIspScale(1, 2)      # use ISP to downscale 4K to 1080p resolution -> HQ frames
elif RES_HQ != (3840, 2160):
    cam_rgb.setVideoSize(*RES_HQ)  # crop to configured HQ resolution -> HQ frames
cam_rgb.setPreviewSize(*RES_LQ)    # downscale frames for model input -> LQ frames
if abs(RES_HQ[0] / RES_HQ[1] - 1) > 0.01:     # check if HQ resolution is not ~1:1 aspect ratio
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square for model input

if config.camera.focus.mode == "range":
    # Set auto focus range using either distance to camera (cm) or lens position (0-255)
    if config.camera.focus.distance.enabled:
        lens_pos_min = convert_cm_lens_position(config.camera.focus.distance.range.max)
        lens_pos_max = convert_cm_lens_position(config.camera.focus.distance.range.min)
        cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
    elif config.camera.focus.lens_position.enabled:
        lens_pos_min = config.camera.focus.lens_position.range.min
        lens_pos_max = config.camera.focus.lens_position.range.max
        cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
elif config.camera.focus.mode == "manual":
    # Set manual focus position using either distance to camera (cm) or lens position (0-255)
    if config.camera.focus.distance.enabled:
        lens_pos = convert_cm_lens_position(config.camera.focus.distance.manual)
        cam_rgb.initialControl.setManualFocus(lens_pos)
    elif config.camera.focus.lens_position.enabled:
        lens_pos = config.camera.focus.lens_position.manual
        cam_rgb.initialControl.setManualFocus(lens_pos)

# Set ISP configuration parameters
cam_rgb.initialControl.setSharpness(config.camera.isp.sharpness)
cam_rgb.initialControl.setLumaDenoise(config.camera.isp.luma_denoise)
cam_rgb.initialControl.setChromaDenoise(config.camera.isp.chroma_denoise)

# Create and configure video encoder node and define input
encoder = pipeline.create(dai.node.VideoEncoder)
encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
encoder.setQuality(config.camera.jpeg_quality)
cam_rgb.video.link(encoder.input)  # HQ frames as encoder input

# Create and configure YOLO detection network node and define input
yolo = pipeline.create(dai.node.YoloDetectionNetwork)
labels = config_model.mappings.labels
yolo.setBlobPath(BASE_PATH / config.detection.model.weights)
yolo.setConfidenceThreshold(config.detection.conf_threshold)
yolo.setIouThreshold(config.detection.iou_threshold)
yolo.setNumClasses(config_model.nn_config.NN_specific_metadata.classes)
yolo.setCoordinateSize(config_model.nn_config.NN_specific_metadata.coordinates)
yolo.setAnchors(config_model.nn_config.NN_specific_metadata.anchors)
yolo.setAnchorMasks(config_model.nn_config.NN_specific_metadata.anchor_masks)
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

if EXP_REGION:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

try:
    # Create metadata .csv file, connect to OAK device and start pipeline in USB2 mode
    metadata_path = save_path / f"{timestamp_dir}_metadata.csv"
    with (open(metadata_path, "a", buffering=1, encoding="utf-8") as metadata_file,
          dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device,
          ThreadPoolExecutor(max_workers=3) as executor):

        # Create output queues to get the synchronized HQ frames and tracker + model output
        q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
        q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

        # Create input queue to send control commands to OAK camera (if exposure_region is enabled)
        q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False) if EXP_REGION else None

        # Write header to metadata .csv file
        metadata_writer = csv.DictWriter(metadata_file, fieldnames=[
            "cam_ID", "rec_ID", "timestamp", "label", "confidence",
            "track_ID", "track_status", "x_min", "y_min", "x_max", "y_max"
        ])
        metadata_writer.writeheader()

        # Create function to get OAK chip temperature
        get_temp_oak = create_get_temp_oak(device)
        temp_oak = get_temp_oak()

        if config.logging.enabled:
            # Write RPi + OAK info to .csv file at configured interval
            scheduler = BackgroundScheduler()
            scheduler.add_job(save_logs, "interval", seconds=config.logging.interval, id="log",
                              args=[save_path, CAM_ID, rec_id, get_temp_oak, get_power_info],
                              next_run_time=datetime.now() + timedelta(seconds=2))
            scheduler.start()

        # Wait for 2 seconds to let camera adjust auto focus and exposure
        time.sleep(2)

        # Write info on start of recording session to log file
        if PWR_MGMT:
            logger.info("Recording %s started | Duration: %s min | Free disk space: %s MB | Charge level: %s%%",
                        rec_id, round(REC_TIME / 60), disk_free, chargelevel_start)
        else:
            logger.info("Recording %s started | Duration: %s min | Free disk space: %s MB",
                        rec_id, round(REC_TIME / 60), disk_free)

        # Create variables for start of recording and capture/check events
        rec_start = datetime.now()
        start_time = time.monotonic()
        last_capture = start_time - CAP_INT_TL  # capture first frame immediately at start
        next_capture = start_time + CAP_INT_DET
        last_temp_check = start_time
        last_disk_check = start_time
        last_charge_check = start_time if PWR_MGMT else None
        chargelevel = chargelevel_start if PWR_MGMT else None
        chargelevels = []

        try:
            # Run recording session until either:
            while (time.monotonic() < start_time + REC_TIME and  # configured recording duration is reached
                   disk_free > DISK_MIN and                      # free disk space drops below threshold
                   temp_oak < TEMP_OAK_MAX and                   # OAK chip temperature exceeds threshold
                   len(chargelevels) < 3):                       # charge level drops below threshold for three times

                # Activate HQ frame capture events based on current time and configured intervals
                track_active = False
                current_time = time.monotonic()
                trigger_capture = current_time >= next_capture
                timelapse_capture = current_time >= last_capture + CAP_INT_TL

                if q_frame.has() and (trigger_capture or timelapse_capture):
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

                                if EXP_REGION and tracklet_status == "TRACKED" and tracklet is tracklets[-1]:
                                    # Use model bbox from latest active tracking ID to set auto exposure region
                                    roi_x, roi_y, roi_w, roi_h = convert_bbox_roi(bbox, SENSOR_RES)
                                    q_ctrl.send(dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h))

                    if track_active or timelapse_capture:
                        # Save MJPEG-encoded HQ frame to .jpg file in separate thread
                        executor.submit(save_encoded_frame, save_path, timestamp_str, frame_hq)
                        last_capture = current_time
                        next_capture = current_time + CAP_INT_DET

                        # Update free disk space (MB) at configured interval
                        if current_time >= last_disk_check + DISK_CHECK:
                            disk_free = round(psutil.disk_usage("/").free / 1048576)
                            last_disk_check = current_time

                # Update OAK chip temperature at configured interval
                if current_time >= last_temp_check + TEMP_OAK_CHECK:
                    temp_oak = get_temp_oak()
                    if temp_oak == "NA":
                        temp_oak = 0  # set to 0 if "NA" is returned
                    last_temp_check = current_time

                # Update charge level at configured interval and add to list if lower than threshold or not readable
                if PWR_MGMT:
                    if current_time >= last_charge_check + CHARGE_CHECK:
                        chargelevel = get_chargelevel()
                        if (chargelevel != "USB_C_IN" and chargelevel < CHARGE_MIN) or chargelevel == "NA":
                            chargelevels.append(chargelevel)
                        last_charge_check = current_time

                # Sleep for a short duration to avoid busy waiting
                time.sleep(0.02)

            # Write info on end of recording to log file
            rec_stop_disk = disk_free < DISK_MIN
            rec_stop_temp_oak = temp_oak >= TEMP_OAK_MAX
            rec_stop_charge = PWR_MGMT and len(chargelevels) >= 3
            if rec_stop_disk:
                logger.warning("Recording %s stopped early due to low disk space: %s MB", rec_id, disk_free)
            elif rec_stop_temp_oak:
                logger.warning("Recording %s stopped early due to high OAK chip temperature: %s °C", rec_id, temp_oak)
            elif rec_stop_charge:
                logger.warning("Recording %s stopped early due to low charge level: %s%%", rec_id, chargelevel)
            elif PWR_MGMT:
                logger.info("Recording %s finished | Duration: %s min | Free disk space: %s MB | Charge level: %s%%",
                            rec_id, round(REC_TIME / 60), disk_free, chargelevel)
            else:
                logger.info("Recording %s finished | Duration: %s min | Free disk space: %s MB",
                            rec_id, round(REC_TIME / 60), disk_free)

        except Exception:
            logger.exception("Error during recording %s", rec_id)
        finally:
            # Write recording logs to .csv file
            rec_end = datetime.now()
            record_log(save_path, CAM_ID, rec_id, rec_start, rec_end, chargelevel_start, chargelevel)

            if "scheduler" in locals():
                # Shut down scheduler (wait until currently executing jobs are finished)
                scheduler.shutdown()

    if config.post_processing.crop.enabled or config.post_processing.overlay.enabled:
        if not rec_stop_disk and not rec_stop_charge:
            power_ok = not PWR_MGMT or (chargelevel == "USB_C_IN" or chargelevel > CHARGE_MIN + 5)
            if power_ok:
                try:
                    # Post-process saved HQ frames based on configured methods
                    if next(save_path.glob("*.jpg"), None):
                        processing_methods = {method for method in ["crop", "overlay", "delete"]
                                              if getattr(config.post_processing, method).enabled}
                        process_images(save_path, processing_methods, config.post_processing.crop.method)
                    logger.info("Post-processing of saved HQ frames finished")
                except Exception:
                    logger.exception("Error during post-processing of saved HQ frames")
            else:
                logger.warning("Skipped post-processing due to low charge level: %s%%", chargelevel)
        elif rec_stop_disk:
            logger.warning("Skipped post-processing as recording was stopped due to low disk space: %s MB", disk_free)
        elif rec_stop_charge:
            logger.warning("Skipped post-processing as recording was stopped due to low charge level: %s%%", chargelevel)

    if config.archive.enabled or config.upload.enabled:
        if not rec_stop_disk and not rec_stop_charge:
            power_ok = not PWR_MGMT or (chargelevel == "USB_C_IN" or chargelevel > CHARGE_MIN + 10)
            if power_ok:
                try:
                    # Archive all captured data + logs/configs and manage disk space
                    archive_path = archive_data(DATA_PATH, CAM_ID, config.archive.disk_low)
                    logger.info("Archiving of data finished")
                    if config.upload.enabled:
                        # Upload archived data to cloud storage provider
                        upload_data(DATA_PATH, archive_path, config.upload.content)
                        logger.info("Uploading of data finished")
                except Exception:
                    logger.exception("Error during archiving/uploading of data")
            else:
                logger.warning("Skipped archiving/uploading due to low charge level: %s%%", chargelevel)
        elif rec_stop_disk:
            logger.warning("Skipped archiving/uploading as recording was stopped due to low disk space: %s MB", disk_free)
        elif rec_stop_charge:
            logger.warning("Skipped archiving/uploading as recording was stopped due to low charge level: %s%%", chargelevel)

except KeyboardInterrupt:
    logger.warning("Recording %s stopped by Ctrl+C", rec_id)
except SystemExit:
    logger.warning("Recording %s stopped by external trigger", rec_id)
except Exception:
    logger.exception("Error during initialization of recording %s", rec_id)
finally:
    if not external_shutdown.is_set():
        # Shut down Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
