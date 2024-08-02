"""Utility functions for bounding box adjustment and data storage.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    create_signal_handler(): Create signal handler for a received signal.
    frame_norm(): Convert relative bounding box coordinates (0-1) to pixel coordinates.
    make_bbox_square(): Adjust bounding box dimensions to make it square.
    archive_data(): Archive all captured data + logs and manage disk space.

frame_norm() is based on open source scripts available at https://github.com/luxonis
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import psutil


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


def archive_data(data_path, cam_id, low_diskspace=1000):
    """Archive all captured data + logs and manage disk space.

    Directories (images + metadata) are saved to uncompressed .zip files,
    log files are copied to archive directory. Original data is deleted
    starting from the oldest directory if the remaining free disk space
    drops below the specified threshold.
    """
    archive_path = data_path.parent / "data_archived" / cam_id
    archive_path.mkdir(parents=True, exist_ok=True)

    dirs_orig = []
    for file_or_dir in data_path.iterdir():
        if file_or_dir.is_dir():
            dirs_orig.append(file_or_dir)
            zip_path = archive_path / (file_or_dir.name + ".zip")
            subprocess.run(["zip", "-q", "-r", "-u", "-0", zip_path, "."],
                           cwd=file_or_dir, check=False)
        elif file_or_dir.is_file():
            subprocess.run(["rsync", "-a", "-u", file_or_dir, archive_path], check=True)
    dirs_orig.sort()

    cronjob_log_file = Path.home() / "insect-detect" / "cronjob_log.log"
    if cronjob_log_file.exists():
        subprocess.run(["rsync", "-a", "-u", cronjob_log_file, archive_path], check=True)

    disk_free = round(psutil.disk_usage("/").free / 1048576)
    while dirs_orig and disk_free < low_diskspace:
        shutil.rmtree(dirs_orig[0], ignore_errors=True)
        dirs_orig.pop(0)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

    return archive_path
