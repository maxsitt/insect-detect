"""Utility functions for post-processing of captured images.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    OverlayParams: Rendering parameters for drawing bounding boxes and text overlays.
    ImageParams:   Image dimensions and normalization values for coordinate math.

Functions:
    make_bbox_square(): Adjust bounding box dimensions to make it square.
    process_images():   Process images in real-time as metadata arrives in queue.
"""

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import cv2.typing
import numpy as np
from numpy.typing import NDArray

from insectdetect.config import AppConfig

# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlayParams:
    """Rendering parameters for drawing bounding boxes and text overlays."""
    font: int
    font_size: tuple[float, float]
    text_thickness: int
    outline_thickness: int
    text_gap: int
    label_height: int
    track_height: int
    total_text_height: int
    box_thickness: int


class ImageParams(TypedDict):
    """Image dimensions and normalization values for coordinate math."""
    width: int
    height: int
    norm_vals: NDArray[np.int32]


def _build_overlay_params(img_w: int) -> OverlayParams:
    """Build overlay rendering parameters based on image width.

    Args:
        img_w: Image width in pixels.

    Returns:
        OverlayParams with font, sizes, thicknesses and gaps for overlay rendering.
    """
    big = img_w > 2000
    font = cv2.FONT_HERSHEY_DUPLEX
    font_size: tuple[float, float] = (1.3, 1.5) if big else (0.9, 1.1)
    text_thickness = 2 if big else 1
    text_gap = 10 if big else 6
    label_h = cv2.getTextSize("insect 0.99", font, font_size[0], text_thickness)[0][1]
    track_h = cv2.getTextSize("ID: 999", font, font_size[1], text_thickness)[0][1]
    return OverlayParams(
        font=font,
        font_size=font_size,
        text_thickness=text_thickness,
        outline_thickness=text_thickness + 4,
        text_gap=text_gap,
        label_height=label_h,
        track_height=track_h,
        total_text_height=label_h + text_gap + track_h + 20,
        box_thickness=3 if big else 2,
    )


def _draw_overlay(
    img: cv2.typing.MatLike,
    bboxes: NDArray[np.int32],
    detections: list[dict[str, object]],
    labels: list[str],
    track_ids: list[object],
    ovl: OverlayParams,
    img_h: int
) -> None:
    """Draw bounding boxes and label/track ID text onto the image in-place.

    Args:
        img:        Image array to draw on (modified in-place), as returned by cv2.imread().
        bboxes:     Bounding boxes as [[x_min, y_min, x_max, y_max], ...] in pixels.
        detections: List of detection metadata dicts for this frame.
        labels:     List of label strings, one per detection.
        track_ids:  List of track IDs, one per detection.
        ovl:        Overlay rendering parameters.
        img_h:      Image height in pixels.
    """
    for idx, (x0, y0, x1, y1) in enumerate(bboxes):
        label_text = f"{labels[idx]} {detections[idx]['confidence']:.2f}"
        track_text = f"ID: {track_ids[idx]}"

        # Adaptive positioning: below bbox if space allows, otherwise above
        if y1 + ovl.total_text_height < img_h * 0.95:
            label_pos = (x0, y1 + ovl.label_height + 10)
            track_pos = (x0, y1 + ovl.label_height + ovl.text_gap + ovl.track_height + 10)
        else:
            label_pos = (x0, y0 - ovl.text_gap - ovl.track_height - 10)
            track_pos = (x0, y0 - 10)

        # Draw text with black outline then white fill for visibility on any background
        for text, pos, size in ((label_text, label_pos, ovl.font_size[0]),
                                (track_text, track_pos, ovl.font_size[1])):
            cv2.putText(img, text, pos, ovl.font, size, (0, 0, 0),
                        ovl.outline_thickness, cv2.LINE_AA)
            cv2.putText(img, text, pos, ovl.font, size, (255, 255, 255),
                        ovl.text_thickness, cv2.LINE_AA)

        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), ovl.box_thickness)


def make_bbox_square(
    img_width: int,
    img_height: int,
    bbox: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Adjust bounding box dimensions to make it square.

    Expands the shorter side of the bounding box symmetrically while keeping
    the result within the image bounds.

    Args:
        img_width:  Image width in pixels.
        img_height: Image height in pixels.
        bbox:       Bounding box as [x_min, y_min, x_max, y_max] in pixels.

    Returns:
        Square bounding box as [x_min, y_min, x_max, y_max] in pixels.
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    if bbox_width == bbox_height:
        return bbox

    bbox_sq = bbox.copy()
    expansion_per_side = abs(bbox_width - bbox_height) // 2

    if bbox_width < bbox_height:
        x0_new = max(0, bbox[0] - expansion_per_side)
        x1_new = min(img_width, bbox[2] + (bbox_height - (bbox[2] - x0_new)))
        if x1_new - x0_new < bbox_height:
            x0_new = max(0, x1_new - bbox_height)
        bbox_sq[0] = x0_new
        bbox_sq[2] = min(x0_new + bbox_height, img_width)
    else:
        y0_new = max(0, bbox[1] - expansion_per_side)
        y1_new = min(img_height, bbox[3] + (bbox_width - (bbox[3] - y0_new)))
        if y1_new - y0_new < bbox_width:
            y0_new = max(0, y1_new - bbox_width)
        bbox_sq[1] = y0_new
        bbox_sq[3] = min(y0_new + bbox_width, img_height)

    return bbox_sq


def process_images(
    metadata_queue: queue.Queue[list[dict[str, object]]],
    session_path: Path,
    config: AppConfig,
    stop_event: threading.Event
) -> None:
    """Process images in real-time as metadata arrives in queue.

    This function is designed to run in a separate thread. It continuously
    monitors the metadata queue and processes images as they become available.
    Stops cleanly when stop_event is set and the queue is drained.

    Args:
        metadata_queue: Queue receiving lists of detection metadata from the recording thread.
        session_path:   Recording session directory where processed images are saved.
        config:         AppConfig with processing and recording settings.
        stop_event:     Threading event signalling that recording has stopped.
    """
    crop_enabled = config.processing.crop.enabled
    crop_method = config.processing.crop.method
    overlay_enabled = config.processing.overlay.enabled
    delete_original = config.processing.delete.enabled
    check_interval = config.recording.interval.detection

    img_params: ImageParams | None = None
    ovl_params: OverlayParams | None = None
    crop_encode_params    = [cv2.IMWRITE_JPEG_QUALITY, 95]
    overlay_encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    labels_seen: set[str] = set()
    processed_count = 0
    crops_path = session_path / "crops"
    overlays_path = session_path / "overlays"

    if crop_enabled:
        crops_path.mkdir(exist_ok=True)
    if overlay_enabled:
        overlays_path.mkdir(exist_ok=True)

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ImgWriter") as write_executor:
            while True:
                timeout = min(check_interval, 5) if metadata_queue.empty() else 0.1

                try:
                    detections = metadata_queue.get(timeout=timeout)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if not detections:
                    logger.warning("Received empty detections list, skipping")
                    continue

                img_filename = str(detections[0]["filename"])
                img_stem = Path(img_filename).stem
                img_path = session_path / img_filename

                # Wait for image file to be written by the recording thread
                max_wait_time = 2.0
                wait_interval = 0.05
                deadline = time.monotonic() + max_wait_time
                while not img_path.exists() and time.monotonic() < deadline:
                    time.sleep(wait_interval)

                if not img_path.exists():
                    logger.warning("Skipping %s - file not found after %.1fs",
                                   img_filename, max_wait_time)
                    continue

                try:
                    # Read image with retries in case the write is still completing
                    img = None
                    img_path_str = str(img_path)
                    max_read_attempts = 5
                    for attempt in range(max_read_attempts):
                        img = cv2.imread(img_path_str)
                        if img is not None:
                            break
                        if attempt < max_read_attempts - 1:
                            logger.debug("Read attempt %d failed for %s, retrying...",
                                         attempt + 1, img_filename)
                            time.sleep(0.2)

                    if img is None:
                        logger.error("Failed to read image %s after %d attempts",
                                     img_filename, max_read_attempts)
                        continue

                    # Initialize image and overlay parameters on first successfully read frame
                    if img_params is None:
                        img_h, img_w = img.shape[:2]
                        img_params = ImageParams(
                            width=img_w,
                            height=img_h,
                            norm_vals=np.array([img_w, img_h, img_w, img_h], dtype=np.int32),
                        )
                        if overlay_enabled:
                            ovl_params = _build_overlay_params(img_w)
                    assert img_params is not None

                    bboxes: NDArray[np.int32] = (np.array(
                        [[d["x_min"], d["y_min"], d["x_max"], d["y_max"]] for d in detections],
                        dtype=np.float32
                    ) * img_params["norm_vals"]).astype(np.int32)
                    labels = [str(d["label"]) for d in detections]
                    track_ids = [d["track_id"] for d in detections]

                    if crop_enabled:
                        for idx, bbox in enumerate(bboxes):
                            label = labels[idx]
                            if label not in labels_seen:
                                (crops_path / label).mkdir(parents=True, exist_ok=True)
                                labels_seen.add(label)

                            if crop_method == "square":
                                bbox = make_bbox_square(
                                    img_params["width"],
                                    img_params["height"],
                                    bbox
                                )

                            crop_filename = f"{img_stem}_ID{track_ids[idx]}_crop.jpg"
                            crop_path = crops_path / label / crop_filename
                            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
                            try:
                                write_executor.submit(cv2.imwrite, str(crop_path), crop,
                                                      crop_encode_params)
                            except RuntimeError:
                                logger.warning("Skip writing crops for %s - executor shutdown",
                                               img_filename)
                                break

                    if overlay_enabled:
                        assert ovl_params is not None
                        _draw_overlay(
                            img, bboxes, detections, labels, track_ids,
                            ovl_params, img_params["height"]
                        )

                        overlay_filename = f"{img_stem}_overlay.jpg"
                        overlay_path = overlays_path / overlay_filename
                        try:
                            write_executor.submit(cv2.imwrite, str(overlay_path), img,
                                                  overlay_encode_params)
                        except RuntimeError:
                            logger.warning("Skip writing overlay for %s - executor shutdown",
                                           img_filename)

                    if delete_original:
                        img_path.unlink(missing_ok=True)

                    processed_count += 1

                except Exception:
                    logger.exception("Error processing image %s", img_filename)

    except Exception:
        logger.exception("Error during image processing")
    finally:
        logger.info("Image processing finished: %d frames processed", processed_count)
