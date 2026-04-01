# Insect Detect - DIY camera trap for automated insect monitoring

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="540">

[![DOI PLOS ONE](https://img.shields.io/badge/PLOS%20ONE-10.1371%2Fjournal.pone.0295474-BD3094)](https://doi.org/10.1371/journal.pone.0295474)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)
[![DOI Zenodo](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)

This repository contains Python scripts and insect detection models for testing
and deploying the **Insect Detect** DIY camera trap for automated insect monitoring.

The camera trap system is composed of low-cost off-the-shelf hardware components
([Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/),
[Luxonis OAK-1](https://docs.luxonis.com/hardware/products/OAK-1),
[Witty Pi 4 L3V7](https://www.uugear.com/product/witty-pi-4-l3v7/)), combined with
open source software and can be easily assembled and set up with the
[provided instructions](https://maxsitt.github.io/insect-detect-docs/).

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/hardware/assets/images/2024_mount_camtrap_platform.jpg" width="400">

## Installation

> [!IMPORTANT]
> Please make sure that you followed [all steps](https://maxsitt.github.io/insect-detect-docs/software/pisetup/)
> to set up your Raspberry Pi.

Install the `insect-detect` software including all required packages and setup steps:

``` bash
wget -qO- https://raw.githubusercontent.com/maxsitt/insect-detect/main/install.sh | bash
```

**Optional:** Install and configure [Rclone](https://rclone.org/docs/) if you want to use the upload feature:

``` bash
wget -qO- https://rclone.org/install.sh | sudo bash
```

Check out the [**Usage**](https://maxsitt.github.io/insect-detect-docs/software/usage/)
documentation for more information.

---

## Detection models

> [!IMPORTANT]
> New detection models are trained with [`luxonis-train`](https://github.com/luxonis/luxonis-train)
> based on an updated training dataset version.

The new training dataset and model training approach will be published soon, together
with more details about the detection model.

> [!WARNING]
> Detection model training is currently being refactored and will be updated soon.
> YOLO detection models trained with the previous approach are not supported anymore
> after insect-detect v2.0.0.

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/images/yolov5n_tracker_episyrphus_320.gif" width="320">

---

## Processing pipeline

All configuration parameters can be customized in the web app or by directly modifying the
[`config.yaml`](https://github.com/maxsitt/insect-detect/blob/main/configs/config.yaml)
file. You can generate multiple custom configuration files and select the active config either in
the web app or by modifying the
[`config_selector.yaml`](https://github.com/maxsitt/insect-detect/blob/main/configs/config_selector.yaml).

Processing pipeline for the
[`capture.py`](https://github.com/maxsitt/insect-detect/blob/main/src/insectdetect/capture.py)
script that can be used for automated insect monitoring:

- A custom **insect detection model** is run in real time on device (OAK)
  and uses a continuous stream of downscaled frames as input.
- An **object tracker** uses the bounding box coordinates of detected insects
  to assign a unique tracking ID to each individual present in the frame and
  track its movement through time.
- The tracker + model output from inference on downscaled frames is synchronized with
  **MJPEG-encoded high-resolution frames** (default: 3840x2160 px) on device (OAK).
- The full frames are saved to the microSD card at the configured
  **capture intervals** while an insect is detected (triggered capture)
  and independent of detections (timelapse capture).
- Corresponding **metadata** from the detection model and tracker output
  is saved to a metadata .csv file for each detected and tracked insect
  (including timestamp, label, confidence score, tracking ID, tracking status
  and bounding box coordinates).
- The bounding box coordinates can be used to **crop detected insects** from
  the corresponding full frames and save them as individual .jpg images.
  Depending on the post-processing configuration, the original full frames are
  optionally deleted to save storage space.
- If a power management board (e.g. Witty Pi 4 L3V7) is connected and
  enabled in the configuration, **intelligent power management** is activated which
  includes battery charge level monitoring with conditional recording durations.
- With the default configuration, running the recording consumes around **4.2 W** of power.

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

``` text
Sittinger M, Uhler J, Pink M, Herz A (2024) Insect detect: An open-source DIY camera trap for automated insect monitoring. PLOS ONE 19(4): e0295474. https://doi.org/10.1371/journal.pone.0295474
```
