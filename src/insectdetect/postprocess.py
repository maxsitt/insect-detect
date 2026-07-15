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
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import cv2
import cv2.typing
import numpy as np
from numpy.typing import NDArray

from insectdetect.config import AppConfig
from insectdetect.constants import IMAGE_DRAIN_TIMEOUT

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Minimum time (seconds) between processing images, to limit RPi CPU usage and power consumption
_THROTTLE_INTERVAL = 2.0


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
    metadata_queue: queue.Queue[tuple[list[dict[str, object]], Future[None]]],
    session_path: Path,
    config: AppConfig,
    stop_event: threading.Event,
    abort_event: threading.Event
) -> None:
    """Process images in real-time as metadata arrives in queue.

    This function is designed to run in a separate thread. It continuously
    monitors the metadata queue and processes images as they become available.
    Stops cleanly when stop_event is set and the queue is drained (bounded by
    IMAGE_DRAIN_TIMEOUT), or immediately once abort_event is set.

    Args:
        metadata_queue: Queue receiving (detections, save_future) tuples from the recording thread.
        session_path:   Recording session directory where processed images are saved.
        config:         AppConfig with processing and recording settings.
        stop_event:     Threading event signalling that recording has stopped.
        abort_event:    Threading event signalling an immediate stop (e.g. Ctrl+C), skipping
                        the graceful drain of any images still queued.
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
    skipped_count = 0
    crops_path = session_path / "crops"
    overlays_path = session_path / "overlays"
    drain_deadline: float | None = None

    if crop_enabled:
        crops_path.mkdir(exist_ok=True)
    if overlay_enabled:
        overlays_path.mkdir(exist_ok=True)

    # Manage executor manually to avoid blocking shutdown if a write is stuck
    write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ImgWriter")
    writer_stuck = False
    last_io_future: Future[Any] | None = None

    def _log_io_result(fut: Future[Any], filename: str, op: str) -> None:
        """Log any exception or unsuccessful result from a I/O operation."""
        try:
            result = fut.result()
        except CancelledError:
            return
        except Exception:
            logger.exception("%s failed for %s", op.capitalize(), filename)
            return
        if result is False:
            logger.error("%s returned failure for %s", op.capitalize(), filename)

    def _submit_io(func: Any, *args: Any, filename: str, op: str, **kwargs: Any) -> tuple[Future[Any], str]:
        """Submit a I/O operation, tracking and logging its outcome."""
        nonlocal last_io_future
        fut = write_executor.submit(func, *args, **kwargs)
        fut.add_done_callback(lambda f: _log_io_result(f, filename, op))
        last_io_future = fut
        return fut, op

    def _io_succeeded(fut: Future[Any], img_filename: str, op: str = "write") -> bool:
        """Wait briefly for a submitted I/O operation and flag the executor as stuck on timeout."""
        nonlocal writer_stuck
        try:
            result = fut.result(timeout=5.0)
        except TimeoutError:
            if not writer_stuck:
                writer_stuck = True
                logger.error(
                    "%s timed out for %s and disk I/O appears stuck, disabling "
                    "image processing for the rest of this session", op.capitalize(), img_filename
                )
            return False
        except Exception:
            return False
        return result is not False

    try:
        while True:
            if abort_event.is_set():
                remaining = metadata_queue.qsize()
                if remaining:
                    logger.warning(
                        "Post-processing aborted: %d image(s) could not be processed", remaining
                    )
                break

            if stop_event.is_set():
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + IMAGE_DRAIN_TIMEOUT
                elif time.monotonic() >= drain_deadline:
                    remaining = metadata_queue.qsize()
                    if remaining:
                        logger.warning(
                            "Post-processing drain timed out after %d seconds: %d image(s) "
                            "could not be processed", int(IMAGE_DRAIN_TIMEOUT), remaining
                        )
                    break

            # Throttle queue.get() timeout to avoid busy-waiting when the queue is empty
            timeout = min(check_interval, 5) if metadata_queue.empty() else 0.1

            try:
                detections, save_future = metadata_queue.get(timeout=timeout)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if not detections:
                logger.warning("Received empty detections list, skipping")
                skipped_count += 1
                continue

            try:
                img_filename = str(detections[0]["filename"])
                img_stem = Path(img_filename).stem
                img_path = session_path / img_filename
            except Exception:
                logger.exception("Malformed detections metadata, skipping")
                skipped_count += 1
                continue

            if writer_stuck:
                # Skip the save wait, image decode and bbox computation if writer thread is stuck
                skipped_count += 1
                continue

            try:
                save_future.result(timeout=5.0)
            except TimeoutError:
                logger.error("Save timed out for %s, skipping", img_filename)
                skipped_count += 1
                continue
            except Exception:
                logger.exception("Save failed for %s, skipping", img_filename)
                skipped_count += 1
                continue

            cycle_start = time.monotonic()

            try:
                write_futures: list[tuple[Future[Any], str]] = []
                write_ok = True

                img = cv2.imread(str(img_path))
                if img is None:
                    logger.error("Failed to read %s after confirmed save", img_filename)
                    skipped_count += 1
                else:
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
                            write_futures.append(_submit_io(
                                cv2.imwrite, str(crop_path), crop, crop_encode_params,
                                filename=crop_filename, op="crop write"))

                    if overlay_enabled:
                        assert ovl_params is not None
                        _draw_overlay(
                            img, bboxes, detections, labels, track_ids,
                            ovl_params, img_params["height"]
                        )

                        overlay_filename = f"{img_stem}_overlay.jpg"
                        overlay_path = overlays_path / overlay_filename
                        write_futures.append(_submit_io(
                            cv2.imwrite, str(overlay_path), img, overlay_encode_params,
                            filename=overlay_filename, op="overlay write"))

                    for fut, op in write_futures:
                        if writer_stuck:
                            write_ok = False
                            break
                        if not _io_succeeded(fut, img_filename, op):
                            write_ok = False

                    if write_ok:
                        processed_count += 1
                    else:
                        skipped_count += 1

                    if delete_original and not writer_stuck and write_ok:
                        del_future, del_op = _submit_io(
                            img_path.unlink, missing_ok=True, filename=img_filename, op="delete")
                        _io_succeeded(del_future, img_filename, del_op)

                if not stop_event.is_set():
                    elapsed = time.monotonic() - cycle_start
                    remaining_pause = _THROTTLE_INTERVAL - elapsed
                    if remaining_pause > 0:
                        stop_event.wait(timeout=remaining_pause)

            except Exception:
                logger.exception("Error processing image %s", img_filename)

    except Exception:
        logger.exception("Error during image processing")
    finally:
        # shutdown(wait=True) below has no timeout, so pre-check the last future with a
        # bounded wait first. Also catches futures left unchecked by an exception between
        # submission and the _io_succeeded() check above, which would leave writer_stuck unset
        if last_io_future is not None and not writer_stuck:
            try:
                last_io_future.result(timeout=5.0)
            except TimeoutError:
                writer_stuck = True
            except Exception:
                pass  # already logged via the done callback
        write_executor.shutdown(wait=not writer_stuck, cancel_futures=writer_stuck)
        logger.info(
            "Image processing finished: %d frames processed, %d skipped",
            processed_count, skipped_count
        )
