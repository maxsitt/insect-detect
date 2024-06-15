"""Utility functions for bounding box adjustment and data storage.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    create_signal_handler(): Create signal handler for a received signal.
    frame_norm(): Convert relative bounding box coordinates (0-1) to pixel coordinates.
    make_bbox_square(): Adjust bounding box dimensions to make it square.
    zip_data(): Store data in an uncompressed .zip file for each day and delete original directory.

frame_norm() is based on open source scripts available at https://github.com/luxonis
"""

import shutil
from zipfile import ZipFile

import numpy as np


def create_signal_handler(external_shutdown):
    """Create signal handler for a received signal."""
    def signal_handler(sig, frame):
        """Handle a received signal by raising a SystemExit exception."""
        external_shutdown.set()
        raise SystemExit

    return signal_handler


def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]

    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def make_bbox_square(frame, bbox):
    """Adjust bounding box dimensions to make it square.

    Increase bounding box size on both sides of the minimum dimension,
    or only on one side if bbox is localized at the frame margin.
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_diff = abs(bbox_width - bbox_height) // 2

    if bbox_width < bbox_height:
        if bbox[0] - bbox_diff < 0:
            bbox[0] = 0
            bbox[2] = bbox[2] + bbox_diff * 2 - bbox[0]
        elif bbox[2] + bbox_diff > frame.shape[1]:
            bbox[0] = bbox[0] - bbox_diff * 2 + frame.shape[1] - bbox[2]
            bbox[2] = frame.shape[1]
        else:
            bbox[0] = bbox[0] - bbox_diff
            bbox[2] = bbox[2] + bbox_diff
    else:
        if bbox[1] - bbox_diff < 0:
            bbox[1] = 0
            bbox[3] = bbox[3] + bbox_diff * 2 - bbox[1]
        elif bbox[3] + bbox_diff > frame.shape[0]:
            bbox[1] = bbox[1] - bbox_diff * 2 + frame.shape[0] - bbox[3]
            bbox[3] = frame.shape[0]
        else:
            bbox[1] = bbox[1] - bbox_diff
            bbox[3] = bbox[3] + bbox_diff

    return bbox


def zip_data(save_path):
    """Store data in an uncompressed .zip file for each day and delete original directory."""
    with ZipFile(f"{save_path.parent}.zip", "a") as zip_file:
        for file in save_path.rglob("*"):
            zip_file.write(file, file.relative_to(save_path.parent))

    shutil.rmtree(save_path.parent, ignore_errors=True)
