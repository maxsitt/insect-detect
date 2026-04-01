"""Utility functions for data management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    save_encoded_frame(): Save MJPEG-encoded frame to .jpg file.
    archive_data(): Copy captured data + logs to archive directory and manage disk space.
    upload_data(): Upload archived data to cloud storage provider via rclone.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Literal

import depthai as dai
import psutil


def save_encoded_frame(
    frame: dai.ImgFrame,
    session_path: Path,
    file_stem: str,
    trigger: Literal["detection", "timelapse"] = "detection"
) -> None:
    """Save MJPEG-encoded frame to .jpg file.

    Args:
        frame:        depthai.ImgFrame message (type: BITSTREAM).
        session_path: Recording session directory where the frame is saved.
        file_stem:    Filename stem (without extension) for the saved .jpg file.
        trigger:      Capture trigger type ('detection' or 'timelapse').
    """
    if trigger == "timelapse":
        img_path = session_path / "timelapse" / f"{file_stem}_timelapse.jpg"
    else:
        img_path = session_path / f"{file_stem}.jpg"
    with open(img_path, "wb", buffering=1024 * 1024) as jpg:
        frame.getData().tofile(jpg)


def archive_data(data_path: Path, device_id: str, disk_low: int = 5000) -> Path:
    """Copy captured data + logs to archive directory and manage disk space.

    Detection, timelapse, overlay and cropped frames are stored in separate
    uncompressed zip files per session. Metadata, log and config files are
    copied directly via rsync. The oldest original data directories are deleted
    when free disk space drops below the 'disk_low' threshold (in MB).

    Args:
        data_path: Root data directory containing daily/session subdirectories.
        device_id: Camera trap ID (hostname), used as archive subdirectory name.
        disk_low:  Free disk space threshold (MB) below which old data is deleted.
                   Corresponds to 'storage.archive.disk_low' in config.

    Returns:
        Path to the archive directory for this camera ID.
    """
    archive_path = data_path.parent / "data_archived" / device_id
    archive_path.mkdir(parents=True, exist_ok=True)

    # Copy root-level files (e.g. session ID counter) to archive root
    for file_path in data_path.glob("*.*"):
        if file_path.is_file():
            subprocess.run(["rsync", "-a", "-u", file_path, archive_path], check=False)

    # Copy log files to archive logs subdirectory
    logs_path = data_path.parent / "logs"
    if logs_path.exists() and logs_path.is_dir():
        archive_logs_path = archive_path / "logs"
        archive_logs_path.mkdir(exist_ok=True)
        for log_file in logs_path.glob("*.log"):
            subprocess.run(["rsync", "-a", "-u", log_file, archive_logs_path], check=False)

    # Iterate over daily date directories, sorted oldest-first for disk management
    dirs_orig = sorted([d for d in data_path.iterdir() if d.is_dir()])
    for date_dir in dirs_orig:
        archive_date_dir = archive_path / date_dir.name
        archive_date_dir.mkdir(exist_ok=True)

        # Copy date-level files (e.g. daily summary CSVs) directly
        for file_path in date_dir.glob("*.*"):
            if file_path.is_file():
                subprocess.run(["rsync", "-a", "-u", file_path, archive_date_dir], check=False)

        for timestamp_dir in date_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            archive_timestamp_dir = archive_date_dir / timestamp_dir.name
            archive_timestamp_dir.mkdir(exist_ok=True)

            # Pack full frames (excluding crops/overlays/timelapse subdirectories) into zip
            if next(timestamp_dir.glob("*.jpg"), None):
                full_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_full.zip"
                subprocess.run([
                    "zip", "-q", "-r", "-u", "-0", full_zip_path, ".", "-i", "*.jpg",
                    "-x", "crops/*", "-x", "overlays/*", "-x", "timelapse/*"
                ], cwd=timestamp_dir, check=False)

            # Pack cropped detection frames into zip
            crop_dir = timestamp_dir / "crops"
            if crop_dir.is_dir() and next(crop_dir.glob("**/*.jpg"), None):
                crop_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_crops.zip"
                subprocess.run([
                    "zip", "-q", "-r", "-u", "-0", crop_zip_path, ".", "-i", "*.jpg",
                ], cwd=crop_dir, check=False)

            # Pack overlay frames into zip
            overlay_dir = timestamp_dir / "overlays"
            if overlay_dir.is_dir() and next(overlay_dir.glob("*.jpg"), None):
                overlay_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_overlays.zip"
                subprocess.run([
                    "zip", "-q", "-r", "-u", "-0", overlay_zip_path, ".", "-i", "*.jpg",
                ], cwd=overlay_dir, check=False)

            # Pack timelapse frames into zip
            timelapse_dir = timestamp_dir / "timelapse"
            if timelapse_dir.is_dir() and next(timelapse_dir.glob("*.jpg"), None):
                timelapse_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_timelapse.zip"
                subprocess.run([
                    "zip", "-q", "-r", "-u", "-0", timelapse_zip_path, ".", "-i", "*.jpg",
                ], cwd=timelapse_dir, check=False)

            # Copy non-image files (metadata CSV, log, config JSON) directly
            for file_path in timestamp_dir.glob("*.*"):
                if file_path.is_file() and file_path.suffix.lower() != ".jpg":
                    subprocess.run(["rsync", "-a", "-u", file_path, archive_timestamp_dir], check=False)

    # Delete oldest original data directories until free disk space exceeds threshold
    disk_free = round(psutil.disk_usage("/").free / 1048576)
    while dirs_orig and disk_free < disk_low:
        shutil.rmtree(dirs_orig.pop(0), ignore_errors=True)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

    return archive_path


def upload_data(
    data_path: Path,
    archive_path: Path,
    content: Literal["all", "full", "crops", "timelapse", "metadata"] = "crops"
) -> None:
    """Upload archived data to cloud storage provider via rclone.

    Selects which archived zip files to upload based on 'content'.
    All options include metadata, log and config files.
    Corresponds to 'storage.upload.content' in config.

    Content options:
    - all:       upload all data except overlay frames
    - full:      upload only full frame zip files
    - crops:     upload only cropped detection zip files (default)
    - timelapse: upload only timelapse frame zip files
    - metadata:  upload only metadata and log files, no images

    Args:
        data_path:    Root data directory (used for rclone log file path).
        archive_path: Path to the archive directory for this camera ID.
        content:      Content selection string from 'storage.upload.content'.
    """
    # This example uses MinIO as storage provider
    # Rclone configuration and providers: https://rclone.org/docs/
    # Rclone copy options: https://rclone.org/commands/rclone_copy/#options

    #rclone_config_path = Path.home() / ".config" / "rclone" / "rclone.conf"  # default config path

    exclude_patterns: dict[str, list[str]] = {
        "full":      ["**/*_crops.zip", "**/*_timelapse.zip", "**/*_overlays.zip"],
        "crops":     ["**/*_full.zip", "**/*_timelapse.zip", "**/*_overlays.zip"],
        "timelapse": ["**/*_full.zip", "**/*_crops.zip", "**/*_overlays.zip"],
        "metadata":  ["**/*.zip", "**/*.jpg"],
        "all":       ["**/*_overlays.zip"],
    }
    exclude_filter = exclude_patterns.get(content, exclude_patterns["all"])

    rclone_cmd: list[str] = [
        "rclone", "copy",
        #f"--config={rclone_config_path}",  # use custom rclone config file (optional)
        #"--progress",          # show transfer progress in terminal (optional)
        "--update",             # skip files that are newer on the destination
        "--transfers", "1",     # limit number of parallel file transfers
        "--buffer-size", "4M",  # limit memory buffer size (MB) when reading file(s)
        "--bwlimit", "4M",      # limit bandwidth (MB/s)
        f"--log-file={data_path / 'rclone.log'}",
        "--log-level=INFO",     # set log level to DEBUG for more detailed output
    ]

    for pattern in exclude_filter:
        rclone_cmd.extend(["--exclude", pattern])

    rclone_cmd.extend([str(archive_path.parent), "minio:bucket"])

    subprocess.run(rclone_cmd, check=False)
