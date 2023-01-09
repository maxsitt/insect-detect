#!/usr/bin/env python3

'''
Author:       Maximilian Sittinger (https://github.com/maxsitt)
Website:      https://maxsitt.github.io/insect-detect-docs/
License:      GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- run a custom YOLOv5 object detection model (.blob format) on-device (Luxonis OAK)
- use 4K frames downscaled to full FOV LQ frames (e.g. 416x416) as model input
- use an object tracker (Intel DL Streamer) to track detected objects and set unique tracking IDs
- show a preview of 4K frames downscaled to full FOV LQ frames (e.g. 416x416) + model/tracker output
- optional argument:
  "-log" print available Raspberry Pi memory (MB) and RPi CPU utilization (percent) to console

compiled with open source scripts available at https://github.com/luxonis
'''

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-log", "--print-log", action="store_true",
    help="print RPi available memory (MB) + CPU utilization (percent)")
args = parser.parse_args()

if args.print_log:
    import psutil

# Set file paths to the detection model and config JSON
MODEL_PATH = Path("./insect-detect/models/yolov5s_416_openvino_2022.1_9shave.blob")
CONFIG_PATH = Path("./insect-detect/models/json/yolov5s_416.json")

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
cam_rgb.setPreviewSize(416, 416) # downscaled LQ frames for model input
cam_rgb.setInterleaved(False)
cam_rgb.setPreviewKeepAspectRatio(False) # squash full FOV frames to square
cam_rgb.setFps(20) # frames per second available for focus/exposure/model input

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
tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
#tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
#tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
nn.passthrough.link(tracker.inputTrackerFrame)
nn.passthrough.link(tracker.inputDetectionFrame)
nn.out.link(tracker.inputDetections)

xout_tracker = pipeline.create(dai.node.XLinkOut)
xout_tracker.setStreamName("track")
tracker.out.link(xout_tracker.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
tracker.passthroughTrackerFrame.link(xout_rgb.input)

# Define function to convert relative bounding box coordinates (0-1) to pixel coordinates
def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Connect to OAK device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Create output queues to get the frames and detections from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

    # Create start_time and counter variables to measure fps of the detection model + tracker
    start_time = time.monotonic()
    counter = 0

    # Get LQ preview frames and tracker output (detections + t.ID) and show in window
    while True:
        if args.print_log:
            print(f"Available RPi memory: {round(psutil.virtual_memory().available / 1048576)} MB")
            print(f"RPi CPU utilization:  {psutil.cpu_percent(interval=None)}%")
            print("\n")

        frame = q_frame.get().getCvFrame()
        track_out = q_track.get()

        if track_out is not None:
            tracklets_data = track_out.tracklets
            counter+=1

        if frame is not None:
            for t in tracklets_data:
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                bbox = frame_norm(frame, (t.srcImgDetection.xmin, t.srcImgDetection.ymin,
                                          t.srcImgDetection.xmax, t.srcImgDetection.ymax))
                cv2.putText(frame, labels[t.srcImgDetection.label], (bbox[0], bbox[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"{round(t.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"t.ID:{t.id}", (bbox[0], bbox[3] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,  0.6, (255, 255, 255), 1)
                cv2.putText(frame, t.status.name, (bbox[0], bbox[3] + 75),
                            cv2.FONT_HERSHEY_SIMPLEX,  0.4, (255, 255, 255), 1)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2) # model output
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 130), 1) # tracker output

            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - start_time)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("tracker_preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break
