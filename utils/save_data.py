"""Utility functions to save images and metadata.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    save_crop_metadata(): Save cropped detection to .jpg and corresponding metadata to .csv.
    save_full_frame(): Save full frame to .jpg.
    save_overlay_frame(): Save full frame with overlays to .jpg.

partly based on open source scripts available at https://github.com/luxonis
"""

import csv
from datetime import datetime

import cv2

from utils.general import make_bbox_square


def save_crop_metadata(cam_id, rec_id, frame, bbox, label, det_conf, track_id,
                       bbox_orig, rec_start_format, save_path, crop="square"):
    """Save cropped detection to .jpg and corresponding metadata to .csv."""
    timestamp = datetime.now()
    if crop == "square":
        bbox = make_bbox_square(frame, bbox.copy())
    bbox_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    timestamp_crop = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")
    path_crop = f"{save_path}/crop/{label}/{timestamp_crop}_ID{track_id}_crop.jpg"
    cv2.imwrite(path_crop, bbox_crop)

    metadata = {
        "cam_ID": cam_id,
        "rec_ID": rec_id,
        "timestamp": timestamp.isoformat(),
        "label": label,
        "confidence": det_conf,
        "track_ID": track_id,
        "x_min": round(bbox_orig[0], 4),
        "y_min": round(bbox_orig[1], 4),
        "x_max": round(bbox_orig[2], 4),
        "y_max": round(bbox_orig[3], 4),
        "file_path": path_crop
    }

    with open(save_path / f"{rec_start_format}_metadata.csv", "a", encoding="utf-8") as metadata_file:
        metadata_writer = csv.DictWriter(metadata_file, fieldnames=metadata.keys())
        if metadata_file.tell() == 0:
            metadata_writer.writeheader()
        metadata_writer.writerow(metadata)


def save_full_frame(frame, save_path):
    """Save full frame to .jpg."""
    if frame is not None:
        timestamp_full = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        path_full = f"{save_path}/full/{timestamp_full}_full.jpg"
        cv2.imwrite(path_full, frame)


def save_overlay_frame(frame, bbox, label, det_conf, track_id,
                       tracklet, tracks, save_path, res_4k=False):
    """Save full frame with overlays to .jpg."""
    text_pos = (48, 98, 164) if res_4k else (28, 55, 92)
    font_size = (1.7, 1.6, 2) if res_4k else (0.9, 0.8, 1.1)
    thickness = 3 if res_4k else 2

    cv2.putText(frame, label, (bbox[0], bbox[3] + text_pos[0]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size[0], (255, 255, 255), thickness)
    cv2.putText(frame, f"{det_conf}", (bbox[0], bbox[3] + text_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size[1], (255, 255, 255), thickness)
    cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[3] + text_pos[2]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size[2], (255, 255, 255), thickness)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness)

    if tracklet == tracks[-1]:
        timestamp_overlay = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        path_overlay = f"{save_path}/overlay/{timestamp_overlay}_overlay.jpg"
        cv2.imwrite(path_overlay, frame)
