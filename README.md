# DIY camera trap for automated insect monitoring

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)
[![License badge](https://img.shields.io/badge/license-GPLv3-yellowgreen)](https://choosealicense.com/licenses/gpl-3.0/)

This repository contains Python scripts and a [YOLOv5s](https://github.com/ultralytics/yolov5)
detection model ([.blob format](https://docs.luxonis.com/en/latest/pages/model_conversion/))
for testing and deploying the Insect Detect DIY camera trap for automated insect monitoring.
The camera trap system is composed of low-cost off-the-shelf hardware components
([Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/),
[Luxonis OAK-1](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1093.html),
[PiJuice Zero pHAT](https://uk.pi-supply.com/products/pijuice-zero)), combined with
open source software and can be easily assembled and set up with the
[provided instructions](https://maxsitt.github.io/insect-detect-docs/).

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/hardware/assets/images/insectdetect_diy_cameratrap.jpg" width="400">

## Installation

Instructions on how to install all required Python packages can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/software/pisetup/#oak-1-configuration) 📑.

## Processing pipeline

More information about the processing pipeline can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/detection/) 📑.

Processing pipeline for the `yolov5_tracker_save_hqsync.py` script that can be used for
continuous automated insect monitoring:

- The object tracker output (+ passthrough detections) from inference on LQ frames (e.g. 416x416) is synchronized
  with HQ frames (e.g. 3840x2160) in a script node on-device (OAK), using the respective sequence numbers.
- Detections (area of the bounding box) are cropped from the synced HQ frames and saved to .jpg.
- All relevant metadata from the detection model and tracker output (timestamp, label, confidence score, tracking ID,
  relative bbox coordinates, .jpg file path) is saved to a metadata .csv file for each cropped detection.

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_sync_pipeline.png" width="600">

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_frame_sync.gif" width="600">

## Detection models

| Model<br><sup>(.blob)   | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Precision<sup>val<br> | Recall<sup>val<br> | Speed<br><sup>OAK<br>(fps) |
| ----------------------- | --------------------- | -------------------- | ----------------- | --------------------- | ------------------ | -------------------------- |
| **YOLOv5n** (+ tracker) | 320                   | 53.9                 | 97.6              | 96.0                  | **96.6**           | **40**                     |
| YOLOv5n (+ tracker)     | 416                   | 58.2                 | 97.4              | **97.0**              | 95.0               | 30                         |
| YOLOv5s (+ tracker)     | 416                   | **63.4**             | **97.8**          | 96.6                  | 95.6               | 17                         |

**Table Notes**

- All models were trained to 300 epochs with batch size 32 and default settings with
  [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml)
  hyperparameters. Reproduce the model training with the provided
  [Google Colab notebook](https://colab.research.google.com/github/maxsitt/insect-detect-ml/blob/main/notebooks/YOLOv5_detection_training_OAK_conversion.ipynb).
- Trained on custom [dataset_320](https://universe.roboflow.com/maximilian-sittinger/insect_detect_detection/dataset/7) or
  [dataset_416](https://universe.roboflow.com/maximilian-sittinger/insect_detect_detection/dataset/4) with only 1 class ("insect").
- Model metrics (mAP, Precision, Recall) are shown for the original .pt model before conversion to ONNX -> OpenVINO -> .blob format.
- Speed (fps) is shown for the converted model in .blob format, running on the OAK device (same speed with object
  tracker). Set `cam_rgb.setFps()` to the respective fps shown for each model to reproduce the speed measurements.
- To reproduce the correct speed (fps) measurement while connected via SSH (X11 forwarding of the frames), print fps to the
  console and comment out `cv2.imshow()`, as this will significantly slow down the received message output and thereby fps.
  If you are using e.g. a Raspberry Pi 4 B connected to a screen, fps will be correctly shown in the livestream (see gif).

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/images/yolov5n_tracker_episyrphus_320.gif" width="400">

## License

All Python scripts are licensed under the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).

## Citation

You can cite this repository as:

```
Sittinger, M. (2022). Insect Detect - Software for automated insect monitoring
with a DIY camera trap system (v1.5). Zenodo. https://doi.org/10.5281/zenodo.7472238
```
