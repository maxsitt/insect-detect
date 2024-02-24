#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- show downscaled frames + fps in a new window (e.g. via X11 forwarding)
  -> "isp_scale=(1, 4)" will downscale 1920x1080 px to 480x270 px

based on open source scripts available at https://github.com/luxonis/depthai/tree/main/depthai_sdk
'''

from depthai_sdk import OakCamera

with OakCamera(usb_speed="usb2") as oak:
#with OakCamera(usb_speed="usb2", rotation=180) as oak:
    cam_rgb = oak.camera("RGB", resolution="1080p", fps=8)
    cam_rgb.config_color_camera(isp_scale=(1, 4), interleaved=False, color_order="BGR")

    visualizer = oak.visualize(cam_rgb.out.main, fps=True)

    oak.start(blocking=False)

    while oak.running():
        oak.poll()
