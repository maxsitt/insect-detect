"""Save HQ frame from OAK camera + synced metadata from model and tracker if object is detected.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Run this script with the Python interpreter from the virtual environment where you installed the
required packages, e.g. with 'env_insdet/bin/python3 insect-detect/yolo_tracker_save_hqsync.py'.

Modify the 'configs/config_selector.yaml' file to select the active configuration file
that will be used to load all configuration parameters.

Modify the 'configs/config_custom.yaml' file to change the settings that are used in
this Python script. Refer to the 'configs/config_default.yaml' for the default settings.

- write info, warning and error (+ traceback) messages to log file
- load YAML file with configuration parameters and JSON file with detection model parameters
- initialize power manager if enabled in config (Witty Pi 4 L3V7 or PiJuice Zero)
- shut down Raspberry Pi without recording if free disk space or
  battery charge level is lower than the configured threshold
- duration of each recording session conditional on current battery charge level (if enabled)
  -> increases efficiency of battery usage and can prevent gaps in recordings
- create a directory for each day and recording session to store images, metadata and configs
- run a custom YOLO object detection model (.blob format) on device (Luxonis OAK)
  -> inference on downscaled + cropped/stretched LQ frames
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on camera fps, inference speed of the detection model and object motion speed
- synchronize tracker output (including model output) from inference on LQ frames
  with MJPEG-encoded HQ frames on device (OAK) using the respective timestamps
  -> maximum pipeline speed (including saving HQ frames):
     full FOV (16:9):    ~19 FPS (3840x2160) | ~42 FPS (1920x1080)
     reduced FOV (~1:1): ~29 FPS (2176x2160) | ~42 FPS (1088x1080)
- save MJPEG-encoded HQ frames to .jpg at configured intervals
  if object is detected (triggered capture) and independent of detections (timelapse capture)
- save corresponding metadata from tracker and model output to metadata .csv file
  (time, label, confidence, tracking ID, tracking status, relative bbox coordinates)
- stop recording session (and shut down Raspberry Pi if enabled) if either:
  - configured recording duration is reached
  - recording is stopped by external trigger (e.g. button press)
  - free disk space drops below configured threshold
  - OAK chip temperature exceeds configured threshold
  - battery charge level drops below configured threshold for three times (if enabled)
  - error occurs during recording session
- write info about recording session to record log .csv file (rec ID, start/end time,
  duration, number of unique tracking IDs, free disk space, battery charge level)
- post-process saved HQ frames based on configured methods (if enabled)
- archive all captured data + logs/configs and manage disk space (if enabled)
- upload archived data to cloud storage provider (if enabled)

partly based on open source scripts available at https://github.com/luxonis
"""

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
from gpiozero import LED

from utils.config import parse_json, parse_yaml, sanitize_config
from utils.data import archive_data, save_encoded_frame, upload_data
from utils.log import record_log, save_logs
from utils.oak import convert_bbox_roi, create_get_temp_oak, create_pipeline
from utils.post import process_images
from utils.power import init_power_manager

# Set base path and get camera trap ID (default: hostname)
BASE_PATH = Path.home() / "insect-detect"
CAM_ID = socket.gethostname()

# Create directory where data will be stored (images, metadata, configs)
DATA_PATH = BASE_PATH / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Create directory where logs will be stored
LOGS_PATH = BASE_PATH / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Set logging levels and format, write logs to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
                    filename=f"{LOGS_PATH}/{Path(__file__).stem}.log", encoding="utf-8")
logger = logging.getLogger()
logger.info("-------- Recording Logger initialized --------")
logging.getLogger("apscheduler").setLevel(logging.WARNING)  # decrease apscheduler logging level
logging.getLogger("tzlocal").setLevel(logging.ERROR)        # suppress timezone warning

# Parse configuration files
config_selector = parse_yaml(BASE_PATH / "configs" / "config_selector.yaml")
config_active = config_selector.config_active
config = parse_yaml(BASE_PATH / "configs" / config_active)
config_model = parse_json(BASE_PATH / "models" / config.detection.model.config)

# Constantly light LED to indicate recording is running
led = None
if config.led.enabled:
    led_gpio_pin = config.led.gpio_pin
    for _ in range(30):  # retry for 3 seconds as LED might still be used by other process
        try:
            led = LED(led_gpio_pin)
            break
        except Exception:
            time.sleep(0.1)
if led:
    led.on()

# Extract some frequently used configuration parameters
PWR_MGMT = config.powermanager.enabled
PWR_MGMT_MODEL = config.powermanager.model if PWR_MGMT else None
CHARGE_MIN = config.powermanager.charge_min if PWR_MGMT else None
CHARGE_CHECK = config.powermanager.charge_check if PWR_MGMT else None
TEMP_OAK_MAX = config.oak.temp_max
TEMP_OAK_CHECK = config.oak.temp_check
DISK_MIN = config.storage.disk_min
DISK_CHECK = config.storage.disk_check
CAP_INT_DET = config.recording.capture_interval.detection
CAP_INT_TL = config.recording.capture_interval.timelapse
EXP_REGION = config.detection.exposure_region.enabled
LABELS = config_model.mappings.labels

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
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
if PWR_MGMT:
    if (chargelevel_start != "USB_C_IN" and chargelevel_start < CHARGE_MIN) or chargelevel_start == "NA":
        logger.warning("Shut down without recording due to low charge level: %s%%", chargelevel_start)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

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
config_path = save_path / f"{timestamp_dir}_{Path(config_active).stem}.json"
config_model_path = save_path / f"{timestamp_dir}_{config.detection.model.config}"
json.dump(sanitize_config(config), config_path.open("w", encoding="utf-8"), indent=2)
json.dump(config_model, config_model_path.open("w", encoding="utf-8"), indent=2)

# Create depthai pipeline and set path to metadata .csv file
pipeline, sensor_res = create_pipeline(BASE_PATH, config, config_model,
                                       use_webapp_config=False, create_xin=EXP_REGION)
metadata_path = save_path / f"{timestamp_dir}_metadata.csv"

try:
    with (open(metadata_path, "a", buffering=1, encoding="utf-8") as metadata_file,
          dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device,  # start device in USB2 mode
          ThreadPoolExecutor(max_workers=3) as executor):

        # Create output queues to get the synchronized HQ frames and tracker + model output
        q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
        q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

        # Create input queue to send control commands to OAK camera (if exposure_region is enabled)
        q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False) if EXP_REGION else None

        # Write header to metadata .csv file
        metadata_writer = csv.DictWriter(metadata_file, fieldnames=[
            "cam_ID", "rec_ID", "timestamp", "lens_position", "iso_sensitivity", "exposure_time",
            "label", "confidence", "track_ID", "track_status", "x_min", "y_min", "x_max", "y_max"
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

        # Initialize variables for start of recording and capture/check events
        rec_start = datetime.now()
        start_time = time.monotonic()
        last_capture = start_time - CAP_INT_TL  # capture first frame immediately at start
        next_capture = start_time + CAP_INT_DET
        last_temp_check = start_time
        last_disk_check = start_time
        last_charge_check = start_time if PWR_MGMT else None
        chargelevel = chargelevel_start if PWR_MGMT else None
        chargelevels = []
        exposure_region_active = False

        try:
            # Run recording session until either:
            while (time.monotonic() < start_time + REC_TIME  # configured recording duration is reached
                   and not external_shutdown.is_set()        # recording is stopped by external trigger
                   and disk_free > DISK_MIN                  # free disk space drops below threshold
                   and temp_oak < TEMP_OAK_MAX               # OAK chip temperature exceeds threshold
                   and len(chargelevels) < 3):               # charge level drops below threshold for three times

                # Initialize tracking variables
                track_active = False
                track_id_max = -1
                track_id_max_bbox = None

                # Activate HQ frame capture events based on current time and configured intervals
                current_time = time.monotonic()
                triggered_capture = current_time >= next_capture
                timelapse_capture = current_time >= last_capture + CAP_INT_TL

                if q_frame.has() and (triggered_capture or timelapse_capture):
                    # Get MJPEG-encoded HQ frame and associated data (synced with tracker output)
                    timestamp = datetime.now()
                    timestamp_iso = timestamp.isoformat()
                    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")
                    frame_dai = q_frame.get()       # depthai.ImgFrame (type: BITSTREAM)
                    frame_hq = frame_dai.getData()  # frame data (bitstream in numpy array)
                    lens_pos = frame_dai.getLensPosition()
                    iso_sens = frame_dai.getSensitivity()
                    exp_time = frame_dai.getExposureTime().total_seconds() * 1000  # milliseconds

                    if q_track.has():
                        # Get tracker output (including passthrough model output)
                        tracklets = q_track.get().tracklets
                        for tracklet in tracklets:
                            # Check if tracklet is active (not "LOST" or "REMOVED")
                            tracklet_status = tracklet.status.name
                            if tracklet_status in {"TRACKED", "NEW"}:
                                track_active = True
                                track_id = tracklet.id
                                bbox = (tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin,
                                        tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax)

                                if tracklet_status == "TRACKED" and track_id > track_id_max:
                                    track_id_max = track_id
                                    track_id_max_bbox = bbox

                                # Save metadata from camera and tracker + model output to .csv file
                                metadata = {
                                    "cam_ID": CAM_ID,
                                    "rec_ID": rec_id,
                                    "timestamp": timestamp_iso,
                                    "lens_position": lens_pos,
                                    "iso_sensitivity": iso_sens,
                                    "exposure_time": round(exp_time, 2),
                                    "label": LABELS[tracklet.srcImgDetection.label],
                                    "confidence": round(tracklet.srcImgDetection.confidence, 2),
                                    "track_ID": track_id,
                                    "track_status": tracklet_status,
                                    "x_min": round(bbox[0], 4),
                                    "y_min": round(bbox[1], 4),
                                    "x_max": round(bbox[2], 4),
                                    "y_max": round(bbox[3], 4)
                                }
                                metadata_writer.writerow(metadata)

                        if EXP_REGION:
                            if track_id_max_bbox:
                                # Use model bbox from most recent active tracking ID to set auto exposure region
                                roi_x, roi_y, roi_w, roi_h = convert_bbox_roi(track_id_max_bbox, sensor_res)
                                exp_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h)
                                q_ctrl.send(exp_ctrl)
                                exposure_region_active = True
                            elif exposure_region_active:
                                # Reset auto exposure region to full frame if there is no active tracking ID
                                roi_x, roi_y, roi_w, roi_h = 1, 1, sensor_res[0] - 1, sensor_res[1] - 1
                                exp_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h)
                                q_ctrl.send(exp_ctrl)
                                exposure_region_active = False

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
            rec_stop_shutdown = external_shutdown.is_set()
            rec_stop_disk = disk_free < DISK_MIN
            rec_stop_temp_oak = temp_oak >= TEMP_OAK_MAX
            rec_stop_charge = PWR_MGMT and len(chargelevels) >= 3
            if rec_stop_shutdown:
                logger.warning("Recording %s stopped early by external trigger", rec_id)
            elif rec_stop_disk:
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
        if not rec_stop_shutdown and not rec_stop_disk and not rec_stop_charge:
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
        elif rec_stop_shutdown:
            logger.warning("Skipped post-processing as recording was stopped by external trigger")
        elif rec_stop_disk:
            logger.warning("Skipped post-processing as recording was stopped due to low disk space: %s MB", disk_free)
        elif rec_stop_charge:
            logger.warning("Skipped post-processing as recording was stopped due to low charge level: %s%%", chargelevel)

    if config.archive.enabled or config.upload.enabled:
        if not rec_stop_shutdown and not rec_stop_disk and not rec_stop_charge:
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
        elif rec_stop_shutdown:
            logger.warning("Skipped archiving/uploading as recording was stopped by external trigger")
        elif rec_stop_disk:
            logger.warning("Skipped archiving/uploading as recording was stopped due to low disk space: %s MB", disk_free)
        elif rec_stop_charge:
            logger.warning("Skipped archiving/uploading as recording was stopped due to low charge level: %s%%", chargelevel)

except KeyboardInterrupt:
    logger.warning("Recording %s stopped by Ctrl+C", rec_id)
except Exception:
    logger.exception("Error during initialization of recording %s", rec_id)
finally:
    if not external_shutdown.is_set():
        if config.recording.shutdown.enabled:
            # Shut down Raspberry Pi
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
