#!/usr/bin/env python3

'''
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- write info and error (stderr) messages (+ traceback) to log file ("script_log.log")
- shut down without recording if PiJuice battery charge level or free disk space (MB)
  are lower than the specified thresholds
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled LQ frames (e.g. 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs (on-device)
- synchronize tracker output (+ passthrough detections) from inference on LQ frames
  with HQ frames (e.g. 1920x1080 px) on-device using the respective sequence numbers
- create new folders for each day, recording interval and object class
- save detections (bounding box area) cropped from HQ frames to .jpg (1080p: ~12.5 fps)
- save corresponding metadata from model + tracker output (time, label, confidence,
  tracking ID, relative bbox coordinates, .jpg file path) to "metadata_{timestamp}.csv"
- duration of each recording interval conditional on PiJuice battery charge level
- write record info (recording ID, start/end time, duration, number of cropped detections,
  number of unique tracking IDs, free disk space and battery charge level) to "record_log.csv"
  and safely shut down RPi after recording interval is finished or if charge level drops
  below the specified threshold or if an error occurs
- optional arguments:
  "-4k" crop detections from (+ save HQ frames in) 4K resolution (default = 1080p)
        -> will slow down pipeline speed to ~3.4 fps (1080p: ~12.5 fps)
  "-crop [square, tight]" save cropped detections with aspect ratio 1:1 ("-crop square") or
                          keep original bbox size with variable aspect ratio ("-crop tight")
                          -> "-crop square" increases bbox size on both sides of the minimum
                             dimension, or only on one side if object is localized at frame
                             margin. default + recommended: increases classification accuracy
  "-raw" additionally save HQ frames to .jpg (e.g. for training data collection)
         -> will slow down pipeline speed to ~4.5 fps (4K sync: ~1.2 fps)
  "-overlay" additionally save HQ frames with overlay (bbox + info) to .jpg
             -> will slow down pipeline speed to ~4.5 fps (4K sync: ~1.2 fps)
  "-log" write RPi CPU + OAK chip temperature, RPi available memory (MB) +
         CPU utilization (%) and battery info to "info_log_{timestamp}.csv"

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
from pijuice import PiJuice


import requests
import RPi.GPIO as GPIO
import time

def capture():
    # Create folder to save images + metadata + logs (if not already present)
    Path("insect-detect/data").mkdir(parents=True, exist_ok=True)

    # Create logger and write info + error messages to log file
    logging.basicConfig(filename="insect-detect/data/script_log.log", encoding="utf-8",
                        format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger()
    sys.stderr.write = logger.error

    # Create folder to save images + metadata + logs (if not already present)
    Path("insect-detect/data").mkdir(parents=True, exist_ok=True)

    # Create logger and write info + error messages to log file
    logging.basicConfig(filename="insect-detect/data/script_log.log", encoding="utf-8",
                        format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger()
    sys.stderr.write = logger.error

    # Define optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-4k", "--four_k_resolution", action="store_true",
        help="crop detections from (+ save HQ frames in) 4K resolution; default = 1080p")
    parser.add_argument("-crop", "--crop_bbox", choices=["square", "tight"], default="square", type=str,
        help="save cropped detections with aspect ratio 1:1 ('-crop square') or \
            keep original bbox size with variable aspect ratio ('-crop tight')")
    parser.add_argument("-raw", "--save_raw_frames", action="store_true",
        help="additionally save full raw HQ frames in separate folder (e.g. for training data)")
    parser.add_argument("-overlay", "--save_overlay_frames", action="store_true",
        help="additionally save full HQ frames with overlay (bbox + info) in separate folder")
    parser.add_argument("-log", "--save_logs", action="store_true",
        help="save RPi CPU + OAK chip temperature, RPi available memory (MB) + \
            CPU utilization (%) and battery info to .csv file")
    args = parser.parse_args()

    if args.save_logs:
        from apscheduler.schedulers.background import BackgroundScheduler
        from gpiozero import CPUTemperature

    # Instantiate PiJuice
    pijuice = PiJuice(1, 0x14)

    # Continue script only if battery charge level and free disk space (MB) are higher than thresholds
    chargelevel_start = pijuice.status.GetChargeLevel().get("data", -1)
    disk_free = round(psutil.disk_usage("/").free / 1048576)
    if chargelevel_start < 10 or disk_free < 200:
        logger.info(f"Shut down without recording | Charge level: {chargelevel_start}%\n")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        time.sleep(5) # wait 5 seconds for RPi to shut down

    # Optional: Disable charging of PiJuice battery if charge level is higher than threshold
    #if chargelevel_start > 80:
    #    pijuice.config.SetChargingConfig({"charging_enabled": False})

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
    #cam_rgb.initialControl.setAutoFocusLensRange(142,146) # platform ~9.5 inches from the camera
    #cam_rgb.initialControl.setManualFocus(143) # platform ~9.5 inches from the camera
    cam_rgb.initialControl.setManualExposure(80000,400)
    #cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)


    if not args.four_k_resolution:
        cam_rgb.setIspScale(1, 2) # downscale 4K to 1080p HQ frames (1920x1080 px)
    cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames for model input
    cam_rgb.setPreviewKeepAspectRatio(False) # "squeeze" frames (16:9) to square (1:1)
    cam_rgb.setInterleaved(False) # planar layout
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    cam_rgb.setFps(10) # frames per second available for focus/exposure/model input

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

    def make_bbox_square(bbox):
        """Increase bbox size on both sides of the minimum dimension, or only on one side if localized at frame margin."""
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_diff = (max(bbox_width, bbox_height) - min(bbox_width, bbox_height)) // 2
        if bbox_width < bbox_height:
            if bbox[0] - bbox_diff < 0:
                det_crop = frame[bbox[1]:bbox[3], 0:bbox[2] + (bbox_diff * 2 - bbox[0])]
            elif not args.four_k_resolution and bbox[2] + bbox_diff > 1920:
                det_crop = frame[bbox[1]:bbox[3], bbox[0] - (bbox_diff * 2 - (1920 - bbox[2])):1920]
            elif args.four_k_resolution and bbox[2] + bbox_diff > 3840:
                det_crop = frame[bbox[1]:bbox[3], bbox[0] - (bbox_diff * 2 - (3840 - bbox[2])):3840]
            else:
                det_crop = frame[bbox[1]:bbox[3], bbox[0] - bbox_diff:bbox[2] + bbox_diff]
        else:
            if bbox[1] - bbox_diff < 0:
                det_crop = frame[0:bbox[3] + (bbox_diff * 2 - bbox[1]), bbox[0]:bbox[2]]
            elif not args.four_k_resolution and bbox[3] + bbox_diff > 1080:
                det_crop = frame[bbox[1] - (bbox_diff * 2 - (1080 - bbox[3])):1080, bbox[0]:bbox[2]]
            elif args.four_k_resolution and bbox[3] + bbox_diff > 2160:
                det_crop = frame[bbox[1] - (bbox_diff * 2 - (2160 - bbox[3])):2160, bbox[0]:bbox[2]]
            else:
                det_crop = frame[bbox[1] - bbox_diff:bbox[3] + bbox_diff, bbox[0]:bbox[2]]
        return det_crop


    latest_images = {}
    image_count = {}  # Dictionary to keep track of image count for each track.id
    webhook_url = "https://nytelyfe-402203.uc.r.appspot.com/upload" # Webhook URL

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
                    if args.crop_bbox == "square":
                        det_crop = make_bbox_square(bbox)
                    else:
                        det_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    label = labels[track.srcImgDetection.label]
                    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                    crop_path = f"{save_path}/cropped/{label}/{timestamp}_{track.id}_crop.jpg"
                    cv2.imwrite(crop_path, det_crop)
            
                    # Update the latest image for this track.id
                    latest_images[track.id] = crop_path
                    
                    # Update image count for this track.id
                    image_count[track.id] = image_count.get(track.id, 0) + 1
                    print(f"Image count for track.id {track.id}: {image_count[track.id]}")
                    
                    

                    if image_count[track.id] == 3:
                        try:
                            with open(crop_path, 'rb') as f:
                                #Open metadata CSV
                                #with open(f"{save_path}/metadata_{rec_start}.csv", 'rb') as metadata_file:
                                    # Prepare the files to be sent
                                files = {'file': f}
                                        #'metadata': ('metadata.csv', metadata_file)
                                
                                data = {
                                'accountID': 'Y7I3Jmp7dCXoank4WXKeTCSoPDp1'  # Replace with your actual account ID
                                }
                                response = requests.post(webhook_url, files=files, data=data)
                            
                                if response.status_code == 200:
                                    print(f"Successfully sent {crop_path} to webhook.")
                                else:
                                    print(f"Failed to send image to webhook. Status code: {response.status_code}")
                        except Exception as e:
                            print(f"An error occurred: {e}")

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
        Write recording ID, time, RPi CPU + OAK chip temperature, RPi available memory (MB) +
        CPU utilization (%) and PiJuice battery info + temp to .csv file.
        """
        with open(f"insect-detect/data/{rec_start[:8]}/info_log_{rec_start[:8]}.csv", "a",
                encoding="utf-8") as log_info_file:
            log_info = csv.DictWriter(log_info_file, fieldnames=
                ["rec_ID", "timestamp", "temp_pi", "temp_oak", "pi_mem_available", "pi_cpu_used",
                "power_input", "charge_status", "charge_level", "temp_batt", "voltage_batt_mV"])
            if log_info_file.tell() == 0:
                log_info.writeheader()
            try:
                temp_oak = round(device.getChipTemperature().average)
            except RuntimeError:
                temp_oak = "NA"
            try:
                logs_info = {
                    "rec_ID": rec_id,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H-%M-%S"),
                    "temp_pi": round(CPUTemperature().temperature),
                    "temp_oak": temp_oak,
                    "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
                    "pi_cpu_used": psutil.cpu_percent(interval=None),
                    "power_input": pijuice.status.GetStatus().get("data", {}).get("powerInput", "NA"),
                    "charge_status": pijuice.status.GetStatus().get("data", {}).get("battery", "NA"),
                    "charge_level": chargelevel,
                    "temp_batt": pijuice.status.GetBatteryTemperature().get("data", "NA"),
                    "voltage_batt_mV": pijuice.status.GetBatteryVoltage().get("data", "NA")
                }
            except IndexError:
                logs_info = {}
            log_info.writerow(logs_info)
            log_info_file.flush()

    # Connect to OAK device and start pipeline in USB2 mode
    with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

        # Write RPi + OAK + battery info to .csv log file at specified interval
        if args.save_logs:
            logging.getLogger("apscheduler").setLevel(logging.WARNING)
            scheduler = BackgroundScheduler()
            scheduler.add_job(save_logs, "interval", seconds=30, id="log")
            scheduler.start()

        # Create empty list to save charge level (if < 10) and set charge level
        lst_chargelevel = []
        chargelevel = chargelevel_start

        # Set recording time conditional on PiJuice battery charge level
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

        # Write info on start of recording to log file
        logger.info(f"Rec ID: {rec_id} | Rec time: {int(rec_time / 60)} min | Charge level: {chargelevel}%")

        # Create output queues to get the frames and tracklets + detections from the outputs defined above
        q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
        q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

        # Set start time of recording
        start_time = time.monotonic()

        try:
            # Record until recording time is finished or charge level dropped below threshold for 10 times
            while time.monotonic() < start_time + rec_time and len(lst_chargelevel) < 10:

                # Update charge level (return "99" if not readable and write to list if < 10)
                chargelevel = pijuice.status.GetChargeLevel().get("data", 99)
                if chargelevel < 10:
                    lst_chargelevel.append(chargelevel)

                # Get synchronized HQ frames + tracker output (passthrough detections)
                if q_frame.has():
                    frame = q_frame.get().getCvFrame()

                    if q_track.has():
                        tracks = q_track.get().tracklets

                        # Save cropped detections (slower if saving additional HQ frames)
                        store_data(frame, tracks)

                # Wait for 1 second
                time.sleep(1)

            # Write info on end of recording to log file and write record logs to .csv
            logger.info(f"Recording {rec_id} finished | Charge level: {chargelevel}%\n")
            record_log()

            # Enable charging of PiJuice battery if charge level is lower than threshold
            if chargelevel < 80:
                pijuice.config.SetChargingConfig({"charging_enabled": True})

            # Shutdown Raspberry Pi
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

        # Write info on error during recording to log file and write record logs to .csv
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(f"Error during recording {rec_id} | Charge level: {chargelevel}%\n")
            record_log()

            # Enable charging of PiJuice battery if charge level is lower than threshold
            if chargelevel < 80:
                pijuice.config.SetChargingConfig({"charging_enabled": True})

            # Shutdown Raspberry Pi
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

if __name__ == "__main__":
    capture()