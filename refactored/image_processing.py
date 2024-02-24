import numpy as np

def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def make_bbox_square(frame, bbox, resolution):
    """Increase bbox size on both sides of the minimum dimension, or only on one side if localized at frame margin."""
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_diff = (max(bbox_width, bbox_height) - min(bbox_width, bbox_height)) // 2
    if bbox_width < bbox_height:
        if bbox[0] - bbox_diff < 0:
            det_crop = frame[bbox[1]:bbox[3], 0:bbox[2] + (bbox_diff * 2 - bbox[0])]
        elif resolution and bbox[2] + bbox_diff > resolution[0]:  # Assuming resolution is a tuple (width, height)
            det_crop = frame[bbox[1]:bbox[3], bbox[0] - (bbox_diff * 2 - (resolution[0] - bbox[2])):resolution[0]]
        else:
            det_crop = frame[bbox[1]:bbox[3], bbox[0] - bbox_diff:bbox[2] + bbox_diff]
    else:
        if bbox[1] - bbox_diff < 0:
            det_crop = frame[0:bbox[3] + (bbox_diff * 2 - bbox[1]), bbox[0]:bbox[2]]
        elif resolution and bbox[3] + bbox_diff > resolution[1]:
            det_crop = frame[bbox[1] - (bbox_diff * 2 - (resolution[1] - bbox[3])):resolution[1], bbox[0]:bbox[2]]
        else:
            det_crop = frame[bbox[1] - bbox_diff:bbox[3] + bbox_diff, bbox[0]:bbox[2]]
    return det_crop
