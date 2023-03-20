# Insect Detect - DIY camera trap for insect monitoring

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains Python scripts and [YOLOv5](https://github.com/ultralytics/yolov5),
[YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7)
and [YOLOv8](https://github.com/ultralytics/ultralytics) object detection models
([.blob format](https://docs.luxonis.com/en/latest/pages/model_conversion/)) for testing
and deploying the **Insect Detect** DIY camera trap for automated insect monitoring.

The camera trap system is composed of low-cost off-the-shelf hardware components
([Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/),
[Luxonis OAK-1](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1093.html),
[PiJuice Zero pHAT](https://uk.pi-supply.com/products/pijuice-zero)), combined with
open source software and can be easily assembled and set up with the
[provided instructions](https://maxsitt.github.io/insect-detect-docs/).

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/hardware/assets/images/insectdetect_diy_cameratrap.jpg" width="400">

---

## Installation

Instructions on how to install all required Python packages can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/software/pisetup/#oak-1-configuration) 📑.

Please make sure that you followed [all steps](https://maxsitt.github.io/insect-detect-docs/software/pisetup/)
to set up your Raspberry Pi before using the OAK-1 camera.

---

## Detection models

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Precision<sup>val<br> | Recall<sup>val<br> | Speed<br><sup>OAK<br>(fps) | params<br><sup>(M) |
| ----------- | --------------------- | -------------------- | ----------------- | --------------------- | ------------------ | -------------------------- | ------------------ |
| YOLOv5n     | 320                   | 53.8                 | 96.9              | 95.5                  | 96.1               | 41                         | 1.76               |
| YOLOv6n     | 320                   | 50.3                 | 95.1              | 96.9                  | 89.8               | 42                         | 4.63               |
| YOLOv7-tiny | 320                   | 53.2                 | 95.7              | 94.7                  | 94.2               | 38                         | 6.01               |
| YOLOv8n     | 320                   | 55.4                 | 94.4              | 92.2                  | 89.9               | 34                         | 3.01               |

**Table Notes**

- All models were trained to 300 epochs with batch size 32 and default hyperparameters. Reproduce the
  model training with the provided [Google Colab notebooks](https://github.com/maxsitt/insect-detect-ml#model-training).
- Trained on [Insect_Detect_detection](https://universe.roboflow.com/maximilian-sittinger/insect_detect_detection)
  dataset [version 7](https://universe.roboflow.com/maximilian-sittinger/insect_detect_detection/dataset/7),
  downscaled to 320x320 pixel with only 1 class ("insect").
- Model metrics (mAP, Precision, Recall) are shown for the original PyTorch (.pt) model before conversion to ONNX ->
  OpenVINO -> .blob format. Reproduce metrics by using the respective model validation method (e.g. `val.py` for YOLOv5).
- Speed (fps) is shown for the converted models (.blob 4 shaves), running on OAK-1 connected to RPi Zero 2 W (same speed with
  object tracker). Set `cam_rgb.setFps()` to the respective fps shown for each model to reproduce the speed measurements.
- While connected via SSH (X11 forwarding of the frames), print fps to the console and comment out `cv2.imshow()`,
  as forwarding the frames will significantly slow down the received message output and thereby fps. If you
  are using a Raspberry Pi 4 B connected to a screen, fps will be correctly shown in the livestream (see gif).

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/images/yolov5n_tracker_episyrphus_320.gif" width="320">

---

## Processing pipeline

More information about the processing pipeline can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/detection/) 📑.

Processing pipeline for the
[`yolo_tracker_save_hqsync.py`](https://github.com/maxsitt/insect-detect/blob/main/yolo_tracker_save_hqsync.py)
script that can be used for continuous automated insect monitoring:

- The object tracker output (+ passthrough detections) from inference on LQ frames (e.g. 320x320) is synchronized
  with HQ frames (e.g. 3840x2160) in a script node on-device (OAK), using the respective sequence numbers.
- Detections (area of the bounding box) are cropped from the synced HQ frames and saved to .jpg.
- All relevant metadata from the detection model and tracker output (timestamp, label, confidence score, tracking ID,
  relative bbox coordinates, .jpg file path) is saved to a metadata .csv file for each cropped detection.

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_sync_pipeline.png" width="800">

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_frame_sync.gif" width="800">

Check out the [classification instructions](https://maxsitt.github.io/insect-detect-docs/deployment/classification/)
and the [`insect-detect-ml`](https://github.com/maxsitt/insect-detect-ml) GitHub repo for more information on how to
classify the cropped detections. Take a look at the [analysis instructions](https://maxsitt.github.io/insect-detect-docs/deployment/analysis/)
for more information on how to post-process and analyze the combined metadata and classification results.

---

## License

All Python scripts are licensed under the GNU General Public License v3.0
([GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0)).

## Citation

You can cite this repository as:

```
Sittinger, M. (2022). Insect Detect - Software for automated insect monitoring
with a DIY camera trap system (v1.5). Zenodo. https://doi.org/10.5281/zenodo.7472238
```
