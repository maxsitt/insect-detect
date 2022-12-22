# DIY camera trap for automated insect monitoring

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)
![License badge](https://img.shields.io/badge/license-GPLv3-yellowgreen)

This repository contains Python scripts and a YOLOv5s detection model (.blob format) for
testing the DIY camera trap system for automated insect monitoring.

## Installation

Instructions on how to install all required Python packages can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/software/pisetup/#oak-1-configuration) 📑.

## Processing pipeline

More information about the processing pipeline can be found in the
[**Insect Detect Docs**](https://maxsitt.github.io/insect-detect-docs/deployment/detection/) 📑.

Processing pipeline for the `yolov5_tracker_save_hqsync.py` script:

- The object tracker output (+ passthrough detections) from inference on LQ frames (e.g. 416x416) is synchronized
  with HQ frames (e.g. 3840x2160) in a script node on-device, using the respective sequence numbers.
- Detections (area of the bounding box) are cropped from the synced HQ frames and saved to .jpg.
- All relevant metadata from the detection model and tracker output (timestamp, label, confidence score, tracking ID,
  relative bbox coordinates, .jpg file path) is saved to a metadata .csv file for each cropped detection.

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_sync_pipeline.png" width="500">

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/deployment/assets/images/hq_frame_sync.png" width="500">

## DIY camera trap

The DIY camera trap for automated insect monitoring is composed of low-cost off-the-shelf hardware components,
combined with completely open source software and can be easily assembled and set up with the
[provided instructions](https://maxsitt.github.io/insect-detect-docs/hardware/).
All Python scripts for testing the system, data collection and continuous automated monitoring can be adapted
to different use cases by changing only a few lines of code.

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/hardware/assets/images/insectdetect_diy_cameratrap.jpg" width="400">

## License

All Python scripts are licensed under the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).

## Citation

Until the corresponding paper will be published, please cite this project as:

``` text
Sittinger, M. (2022). Insect Detect - Software for automated insect monitoring
with a DIY camera trap system. Zenodo. https://doi.org/10.5281/zenodo.7472238
```

[![DOI](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)
