"""Utility functions for image post-processing.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    make_bbox_square(): Adjust bounding box dimensions to make it square.
    process_images(): Process images with specified methods.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def make_bbox_square(img_width, img_height, bbox):
    """Adjust bounding box dimensions to make it square.

    Increase bounding box size on both sides of the minimum dimension,
    or only on one side if bbox is localized at the frame margin.
    """
    bbox_sq = bbox.copy()

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_diff = abs(bbox_width - bbox_height) // 2

    if bbox_width < bbox_height:
        if bbox[0] - bbox_diff < 0:
            bbox_sq[0] = 0
            bbox_sq[2] = bbox[2] + bbox_diff * 2 - bbox[0]
        elif bbox[2] + bbox_diff > img_width:
            bbox_sq[0] = bbox[0] - bbox_diff * 2 + img_width - bbox[2]
            bbox_sq[2] = img_width
        else:
            bbox_sq[0] = bbox[0] - bbox_diff
            bbox_sq[2] = bbox[2] + bbox_diff
    else:
        if bbox[1] - bbox_diff < 0:
            bbox_sq[1] = 0
            bbox_sq[3] = bbox[3] + bbox_diff * 2 - bbox[1]
        elif bbox[3] + bbox_diff > img_height:
            bbox_sq[1] = bbox[1] - bbox_diff * 2 + img_height - bbox[3]
            bbox_sq[3] = img_height
        else:
            bbox_sq[1] = bbox[1] - bbox_diff
            bbox_sq[3] = bbox[3] + bbox_diff

    return bbox_sq


def process_images(save_path, processing_methods, crop_method="square"):
    """Process images with specified methods."""

    def process_image(timestamp, group):
        """Process image to crop and/or overlay bounding boxes."""
        img_processed = False
        timestamp_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d_%H-%M-%S-%f")
        img_path = save_path / f"{timestamp_str}.jpg"
        try:
            img = cv2.imread(str(img_path))
        except Exception:
            return

        bboxes = group[["x_min", "y_min", "x_max", "y_max"]].values  # normalized bounding boxes
        bboxes = (np.clip(bboxes, 0, 1) * norm_vals).astype(int)     # convert to pixel coordinates
        labels = group["label"].values
        track_ids = group["track_ID"].values

        if crop_bboxes:
            for idx, bbox in enumerate(bboxes):
                crop_path = save_path / "crop" / labels[idx] / f"{timestamp_str}_ID{track_ids[idx]}_crop.jpg"
                if crop_method == "square":
                    bbox_sq = make_bbox_square(img_width, img_height, bbox)
                    crop = img[bbox_sq[1]:bbox_sq[3], bbox_sq[0]:bbox_sq[2]]
                else:
                    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cv2.imwrite(str(crop_path), crop)
                img_processed = True

        if draw_overlays:
            det_confs = group["confidence"].values
            overlay_path = save_path / "overlay" / f"{timestamp_str}_overlay.jpg"

            for idx, bbox in enumerate(bboxes):
                cv2.putText(img, f"{labels[idx]}", (bbox[0], bbox[3] + text_pos[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size[0], (255, 255, 255), thickness)
                cv2.putText(img, f"{det_confs[idx]}", (bbox[0], bbox[3] + text_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size[1], (255, 255, 255), thickness)
                cv2.putText(img, f"ID:{track_ids[idx]}", (bbox[0], bbox[3] + text_pos[2]),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size[2], (255, 255, 255), thickness)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness)
            cv2.imwrite(str(overlay_path), img)
            img_processed = True

        if img_processed:
            processed_images.add(img_path)

    # Initialize processing parameters
    processed_images = set()
    crop_bboxes = "crop" in processing_methods
    draw_overlays = "overlay" in processing_methods
    delete_original_images = "delete" in processing_methods

    # Load metadata and group by timestamp (= image)
    metadata_path = next(save_path.glob("*metadata.csv"))
    metadata = pd.read_csv(metadata_path, encoding="utf-8")
    metadata_grouped = metadata.groupby("timestamp")  # grouped metadata per image

    # Create directories for processed images
    if crop_bboxes:
        labels_unique = metadata["label"].unique()
        for label in labels_unique:
            (save_path / "crop" / label).mkdir(parents=True, exist_ok=True)
    if draw_overlays:
        (save_path / "overlay").mkdir(parents=True, exist_ok=True)

    # Get dimensions from sample image to convert normalized bounding boxes to pixel coordinates
    img_sample = cv2.imread(str(next(save_path.glob("*.jpg"))))
    img_height, img_width = img_sample.shape[:2]
    norm_vals = np.array([img_width, img_height, img_width, img_height])

    # Set overlay configuration based on image width
    text_pos = (48, 98, 164) if img_width > 2000 else (28, 55, 92)
    font_size = (1.7, 1.6, 2) if img_width > 2000 else (0.9, 0.8, 1.1)
    thickness = 3 if img_width > 2000 else 2

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_image, timestamp, group)
            for timestamp, group in metadata_grouped
        ]
        for future in futures:
            future.result()

    if delete_original_images:
        # Delete original HQ frames after processing
        for img_path in processed_images:
            img_path.unlink(missing_ok=True)
