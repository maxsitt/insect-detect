"""Utility functions for Luxonis OAK camera control.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    convert_bbox_roi(): Convert bounding box coordinates to ROI (region of interest).
    convert_cm_lens_position(): Convert centimeter value to OAK lens position value.
"""

# Create dictionary containing centimeter values and corresponding OAK lens positions
CM_LENS_POSITIONS = {
    8: 255, 9: 210, 10: 200, 11: 190, 12: 180, 13: 175, 14: 170, 15: 165, 16: 162, 17: 160,
    18: 158, 19: 156, 20: 154, 21: 152, 22: 150, 23: 148, 24: 146, 25: 144, 26: 142, 27: 141,
    28: 140, 29: 139, 30: 138, 31: 137, 32: 136, 34: 135, 36: 134, 38: 133, 40: 132, 42: 131,
    45: 130, 48: 129, 52: 128, 56: 127, 60: 126, 64: 125, 68: 124, 72: 123, 76: 122, 80: 121
}
CM_KEYS = tuple(CM_LENS_POSITIONS.keys())


def convert_bbox_roi(bbox, sensor_res):
    """Convert bounding box coordinates to ROI (region of interest)."""
    def clamp(val, min_val, max_val):
        """Clamp a value between a minimum and a maximum value."""
        return max(min_val, min(val, max_val))

    xmin, ymin, xmax, ymax = [clamp(coord, 0.001, 0.999) for coord in bbox]
    roi_x, roi_y = int(xmin * sensor_res[0]), int(ymin * sensor_res[1])
    roi_w, roi_h = int((xmax - xmin) * sensor_res[0]), int((ymax - ymin) * sensor_res[1])

    return roi_x, roi_y, roi_w, roi_h


def convert_cm_lens_position(distance_cm):
    """Convert centimeter value to OAK lens position value."""
    if distance_cm in CM_KEYS:
        return CM_LENS_POSITIONS[distance_cm]

    closest_cm = min(CM_KEYS, key=lambda k: abs(k - distance_cm))

    return CM_LENS_POSITIONS[closest_cm]
