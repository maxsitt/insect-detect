#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled LQ frames (e.g. 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs (on-device)
- show downscaled LQ frames + model/tracker output (bounding box, label, confidence,
  tracking ID, tracking status) + fps in a new window (e.g. via X11 forwarding)
- optional argument:
  "-log" print available Raspberry Pi memory, RPi CPU utilization + temperature,
         OAK memory + CPU usage and OAK chip temperature to console

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
args = parser.parse_args()

if args.print_logs:
    import psutil
    from apscheduler.schedulers.background import BackgroundScheduler
    from gpiozero import CPUTemperature

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
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames for model input
cam_rgb.setPreviewKeepAspectRatio(False) # "squeeze" frames (16:9) to square (1:1)
cam_rgb.setInterleaved(False) # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(47) # frames per second available for focus/exposure/model input

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

# Create and configure object tracker node and define inputs + outputs
tracker = pipeline.create(dai.node.ObjectTracker)
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
#tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS) # better for low fps
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

# Define functions
def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def print_logs():
    """Print Raspberry Pi info to console."""
    print(f"\nAvailable RPi memory: {round(psutil.virtual_memory().available / 1048576)} MB")
    print(f"RPi CPU utilization:  {round(psutil.cpu_percent(interval=None))} %")
    print(f"RPi CPU temperature:  {round(CPUTemperature().temperature)} °C\n")

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Print RPi + OAK info to console every second
    if args.print_logs:
        scheduler = BackgroundScheduler()
        scheduler.add_job(print_logs, "interval", seconds=1, id="log")
        scheduler.start()
        device.setLogLevel(dai.LogLevel.INFO)
        device.setLogOutputLevel(dai.LogLevel.INFO)

    # Create output queues to get the frames and tracklets + detections from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    # Create start_time and counter variable to measure fps
    start_time = time.monotonic()
    counter = 0

    # Get LQ frames + tracker output (passthrough detections) and show in new window
    while True:
        if q_frame.has():
            frame = q_frame.get().getCvFrame()

            if q_track.has():
                tracks = q_track.get().tracklets
                counter += 1
                fps = round(counter / (time.monotonic() - start_time), 2)

                for track in tracks:
                    roi = track.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    bbox = frame_norm(frame, (track.srcImgDetection.xmin, track.srcImgDetection.ymin,
                                              track.srcImgDetection.xmax, track.srcImgDetection.ymax))
                    cv2.putText(frame, labels[track.srcImgDetection.label], (bbox[0], bbox[3] + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"{round(track.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"ID:{track.id}", (bbox[0], bbox[3] + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, track.status.name, (bbox[0], bbox[3] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2) # model bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 130), 1) # tracker bbox

                cv2.putText(frame, f"fps: {fps}", (4, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("tracker_preview", frame)
                #print(f"fps: {fps}")
                # streaming the frames via SSH (X11 forwarding) will slow down fps
                # comment out "cv2.imshow()" and print fps to console for true fps

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
