# Insect Detect - DIY camera trap for automated insect monitoring

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="540">

[![DOI PLOS ONE](https://img.shields.io/badge/PLOS%20ONE-10.1371%2Fjournal.pone.0295474-BD3094)](https://doi.org/10.1371/journal.pone.0295474)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)
[![DOI Zenodo](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)

This repository contains Python scripts and [YOLOv5](https://github.com/ultralytics/yolov5),
[YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7)
and [YOLOv8](https://github.com/ultralytics/ultralytics) object detection models
([.blob format](https://docs.luxonis.com/software/ai-inference/conversion/)) for testing
and deploying the **Insect Detect** DIY camera trap for automated insect monitoring.

The camera trap system is composed of low-cost off-the-shelf hardware components
([Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/),
[Luxonis OAK-1](https://docs.luxonis.com/hardware/products/OAK-1),
[Witty Pi 4 L3V7](https://www.uugear.com/product/witty-pi-4-l3v7/) or
[PiJuice Zero pHAT](https://uk.pi-supply.com/products/pijuice-zero)), combined with
open source software and can be easily assembled and set up with the
[provided instructions](https://maxsitt.github.io/insect-detect-docs/).

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/hardware/assets/images/2024_mount_camtrap_platform.jpg" width="400">

## Installation

> [!IMPORTANT]
> Please make sure that you followed [all steps](https://maxsitt.github.io/insect-detect-docs/software/pisetup/)
> to set up your Raspberry Pi.

Install all dependencies/packages and automatically run the required setup steps:

``` bash
wget -qO- https://raw.githubusercontent.com/maxsitt/insect-detect/main/insect_detect_install.sh | bash
```

**Optional:** Install and configure [Rclone](https://rclone.org/docs/) if you want to use the upload feature:

``` bash
wget -qO- https://rclone.org/install.sh | sudo bash
```

Check out the [**Usage**](https://maxsitt.github.io/insect-detect-docs/software/usage/)
documentation for more information.

---

## Detection models

| Model       | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Precision<sup>val<br> | Recall<sup>val<br> | Speed<sup>OAK<br>(fps) | params<br><sup>(M) |
| ----------- | --------------------- | -------------------- | ----------------- | --------------------- | ------------------ | ---------------------- | ------------------ |
| YOLOv5n     | 320                   | 53.8                 | 96.9              | 95.5                  | 96.1               | 49                     | 1.76               |
| YOLOv6n     | 320                   | 50.3                 | 95.1              | 96.9                  | 89.8               | 60                     | 4.63               |
| YOLOv7-tiny | 320                   | 53.2                 | 95.7              | 94.7                  | 94.2               | 52                     | 6.01               |
| YOLOv8n     | 320                   | 55.4                 | 94.4              | 92.2                  | 89.9               | 39                     | 3.01               |

<details>
<summary>Table Notes</summary>

- All models were trained to 300 epochs with batch size 32 and default hyperparameters. Reproduce the
  model training with the provided [Google Colab notebooks](https://github.com/maxsitt/insect-detect-ml#model-training).
- Trained on [Insect_Detect_detection](https://universe.roboflow.com/maximilian-sittinger/insect_detect_detection)
  dataset [version 7](https://universe.roboflow.com/maximilian-sittinger/insect_detect_detection/dataset/7),
  downscaled to 320x320 pixel with only 1 class ("insect").
- Model metrics (mAP, Precision, Recall) are shown for the original PyTorch (.pt) model before conversion to ONNX ->
  OpenVINO -> .blob format. Reproduce metrics by using the respective model validation method.

</details>

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/images/yolov5n_tracker_episyrphus_320.gif" width="320">

---

## Processing pipeline

All configuration parameters can be customized in the web app or by directly modifying the
[`config_custom.yaml`](https://github.com/maxsitt/insect-detect/tree/main/configs/config_custom.yaml)
file. You can generate multiple custom configuration files and select the active config either in
the web app or by modifying the
[`config_selector.yaml`](https://github.com/maxsitt/insect-detect/blob/main/configs/config_selector.yaml).

Processing pipeline for the
[`yolo_tracker_save_hqsync.py`](https://github.com/maxsitt/insect-detect/blob/main/yolo_tracker_save_hqsync.py)
script that can be used for automated insect monitoring:

- A custom **YOLO insect detection model** is run in real time on device (OAK) and uses a
  continuous stream of downscaled LQ frames as input.
- An **object tracker** uses the bounding box coordinates of detected insects to assign a unique
  tracking ID to each individual present in the frame and track its movement through time.
- The tracker + model output from inference on LQ frames is synchronized with
  **MJPEG-encoded HQ frames** (default: 3840x2160 px) on device (OAK).
- The HQ frames are saved to the microSD card at the configured **capture intervals** while
  an insect is detected (triggered capture) and independent of detections (time-lapse capture).
- Corresponding **metadata** from the detection model and tracker output is saved to a
  metadata .csv file for each detected and tracked insect (including timestamp, label,
  confidence score, tracking ID, tracking status and bounding box coordinates).
- The bounding box coordinates can be used to **crop detected insects** from the corresponding
  HQ frames and save them as individual .jpg images. Depending on the post-processing configuration,
  the original HQ frames are optionally deleted to save storage space.
- If a power management board (Witty Pi 4 L3V7 or PiJuice Zero) is connected and enabled in the
  configuration, **intelligent power management** is activated which includes battery charge level
  monitoring with conditional recording durations.
- With the default configuration, running the recording consumes **~3.8 W** of power.

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_sync_pipeline.png" width="800">

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_frame_sync_1080p.jpg" width="800">

More information about the processing pipeline can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/detection/) 📑.

Check out the [classification](https://maxsitt.github.io/insect-detect-docs/deployment/classification/)
instructions and the [`insect-detect-ml`](https://github.com/maxsitt/insect-detect-ml) GitHub repo for
information on how to classify the cropped detections with the provided classification model and script.

Take a look at the [post-processing](https://maxsitt.github.io/insect-detect-docs/deployment/post-processing/)
instructions for information on how to post-process the metadata with classification results.

---

## License

This repository is licensed under the terms of the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).

## Citation

If you use resources from this repository, please cite our paper:

```
Sittinger M, Uhler J, Pink M, Herz A (2024) Insect detect: An open-source DIY camera trap for automated insect monitoring. PLOS ONE 19(4): e0295474. https://doi.org/10.1371/journal.pone.0295474
```
