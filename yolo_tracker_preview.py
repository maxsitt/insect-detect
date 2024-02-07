#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on stretched + downscaled LQ frames (default: 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on object motion speed and inference speed of the detection model
- show downscaled LQ frames + model/tracker output (bounding box, label, confidence,
  tracking ID, tracking status) + fps in a new window (e.g. via X11 forwarding)
- optional arguments:
  "-log" print available Raspberry Pi memory, RPi CPU utilization + temperature,
         OAK memory + CPU usage and OAK chip temperature to console
  "-ae"  use bounding box coordinates from detections to set auto exposure region

based on open source scripts available at https://github.com/luxonis
'''

import argparse
import json
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# Define optional argument
parser = argparse.ArgumentParser()
parser.add_argument("-log", "--print_logs", action="store_true",
    help="print RPi available memory, RPi CPU utilization + temperature, \
          OAK memory + CPU usage and OAK chip temperature to console")
parser.add_argument("-ae", "--bbox_ae_region", action="store_true",
    help="use bounding box coordinates from detections to set auto exposure region")
args = parser.parse_args()

if args.print_logs:
    import psutil
    from apscheduler.schedulers.background import BackgroundScheduler
    from gpiozero import CPUTemperature

# Set file paths to the detection model and corresponding config JSON
MODEL_PATH = Path("insect-detect/models/yolov5n_320_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("insect-detect/models/json/yolov5_v7_320.json")

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
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
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

# Create and configure object tracker node and define inputs + outputs
tracker = pipeline.create(dai.node.ObjectTracker)
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
#tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)  # better for low fps
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
nn.passthrough.link(tracker.inputTrackerFrame)
nn.passthrough.link(tracker.inputDetectionFrame)
nn.out.link(tracker.inputDetections)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
tracker.passthroughTrackerFrame.link(xout_rgb.input)

xout_tracker = pipeline.create(dai.node.XLinkOut)
xout_tracker.setStreamName("track")
tracker.out.link(xout_tracker.input)

if args.bbox_ae_region:
    # Create XLinkIn node to send control commands to OAK device (e.g. set auto exposure region)
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)


def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


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


def print_logs():
    """Print Raspberry Pi info to console."""
    print(f"\nAvailable RPi memory: {round(psutil.virtual_memory().available / 1048576)} MB")
    print(f"RPi CPU utilization:  {round(psutil.cpu_percent(interval=None))} %")
    print(f"RPi CPU temperature:  {round(CPUTemperature().temperature)} °C\n")


# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    if args.print_logs:
        # Print RPi + OAK info to console every second
        scheduler = BackgroundScheduler()
        scheduler.add_job(print_logs, "interval", seconds=1, id="log")
        scheduler.start()
        device.setLogLevel(dai.LogLevel.INFO)
        device.setLogOutputLevel(dai.LogLevel.INFO)

    # Create output queues to get the frames and tracklets (+ detections) from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    if args.bbox_ae_region:
        # Create input queue to send commands to the OAK device (e.g. set auto exposure region)
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

    # Create start_time and counter variable to measure fps
    start_time = time.monotonic()
    counter = 0

    while True:
        # Get LQ frames + tracker output (including detections) and show in new window together with fps
        if q_frame.has() and q_track.has():
            frame_lq = q_frame.get().getCvFrame()
            tracks = q_track.get().tracklets

            counter += 1
            fps = round(counter / (time.monotonic() - start_time), 2)

            for tracklet in tracks:
                # Get bounding box from passthrough detections
                xmin, ymin = tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin
                xmax, ymax = tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax
                bbox_det = frame_norm(frame_lq, (xmin, ymin, xmax, ymax))

                # Get bounding box from object tracker
                roi = tracklet.roi.denormalize(frame_lq.shape[1], frame_lq.shape[0])
                x1, y1 = int(roi.topLeft().x), int(roi.topLeft().y)
                x2, y2 = int(roi.bottomRight().x), int(roi.bottomRight().y)

                # Get metadata from tracker output (including passthrough detections)
                label = labels[tracklet.srcImgDetection.label]
                det_conf = round(tracklet.srcImgDetection.confidence, 2)
                track_id = tracklet.id
                track_status = tracklet.status.name

                if args.bbox_ae_region and tracklet == tracks[-1]:
                    # Use model bbox from latest tracking ID to set auto exposure region
                    bbox_set_exposure_region(xmin, ymin, xmax, ymax)

                cv2.putText(frame_lq, label, (bbox_det[0], bbox_det[3] + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame_lq, f"{det_conf}", (bbox_det[0], bbox_det[3] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame_lq, f"ID:{track_id}", (bbox_det[0], bbox_det[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_lq, track_status, (bbox_det[0], bbox_det[3] + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.rectangle(frame_lq, (bbox_det[0], bbox_det[1]), (bbox_det[2], bbox_det[3]),
                              (0, 0, 255), 2)  # model bbox
                cv2.rectangle(frame_lq, (x1, y1), (x2, y2), (0, 255, 130), 1)  # tracker bbox

            cv2.putText(frame_lq, f"fps: {fps}", (4, frame_lq.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("tracker_preview", frame_lq)

            #print(f"fps: {fps}")
            # streaming the frames via SSH (X11 forwarding) will slow down fps
            # comment out "cv2.imshow()" and print fps to console for true fps

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
