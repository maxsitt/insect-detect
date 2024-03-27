"""Utility functions to control the OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    set_focus_range(): Convert closest cm values to lens position values and set auto focus range.
    bbox_set_exposure_region(): Use bounding box coordinates to set auto exposure region.

partly based on open source scripts available at https://github.com/luxonis
"""

import depthai as dai


def set_focus_range(cm_min, cm_max):
    """Convert closest cm values to lens position values and set auto focus range."""
    cm_lenspos = {
        6: 250,
        8: 220,
        10: 190,
        12: 170,
        14: 160,
        16: 150,
        20: 140,
        25: 135,
        30: 130,
        40: 125,
        60: 120
    }

    closest_cm_min = min(cm_lenspos.keys(), key=lambda k: abs(k - cm_min))
    closest_cm_max = min(cm_lenspos.keys(), key=lambda k: abs(k - cm_max))
    lenspos_min = cm_lenspos[closest_cm_max]
    lenspos_max = cm_lenspos[closest_cm_min]

    af_ctrl = dai.CameraControl().setAutoFocusLensRange(lenspos_min, lenspos_max)

    return af_ctrl


def bbox_set_exposure_region(bbox, sensor_res):
    """Use bounding box coordinates to set auto exposure region."""
    xmin_roi = max(0.001, bbox[0])
    ymin_roi = max(0.001, bbox[1])
    xmax_roi = min(0.999, bbox[2])
    ymax_roi = min(0.999, bbox[3])

    roi_x = int(xmin_roi * sensor_res[0])
    roi_y = int(ymin_roi * sensor_res[1])
    roi_width = int((xmax_roi - xmin_roi) * sensor_res[0])
    roi_height = int((ymax_roi - ymin_roi) * sensor_res[1])

    ae_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_width, roi_height)

    return ae_ctrl
