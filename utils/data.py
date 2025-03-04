"""Utility functions for data management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    save_encoded_frame(): Save MJPEG-encoded frame to .jpg file.
    archive_data(): Archive all captured data + logs/configs and manage disk space.
    upload_data(): Upload archived data to cloud storage provider.
"""

import shutil
import subprocess
from pathlib import Path

import psutil


def save_encoded_frame(save_path, timestamp_str, frame):
    """Save MJPEG-encoded frame to .jpg file."""
    with open(save_path / f"{timestamp_str}.jpg", "wb") as jpg:
        jpg.write(frame)


def archive_data(data_path, cam_id, low_diskspace=5000):
    """Archive all captured data + logs/configs and manage disk space.

    Store full frames, overlay frames and cropped detections in separate
    uncompressed .zip files, copy metadata/log/config files to archive
    directory. Original data is deleted starting from the oldest directory
    if the remaining free disk space drops below the configured threshold.
    """
    archive_path = data_path.parent / "data_archived" / cam_id
    archive_path.mkdir(parents=True, exist_ok=True)

    cronjob_log = Path.home() / "insect-detect" / "cronjob_log.log"
    if cronjob_log.exists():
        subprocess.run(["rsync", "-a", "-u", cronjob_log, archive_path], check=False)

    for file_path in data_path.glob("*.*"):
        if file_path.is_file():
            subprocess.run(["rsync", "-a", "-u", file_path, archive_path], check=False)

    dirs_orig = sorted([d for d in data_path.iterdir() if d.is_dir()])
    for date_dir in dirs_orig:
        archive_date_dir = archive_path / date_dir.name
        archive_date_dir.mkdir(exist_ok=True)

        for file_path in date_dir.glob("*.*"):
            if file_path.is_file():
                subprocess.run(["rsync", "-a", "-u", file_path, archive_date_dir], check=False)

        for timestamp_dir in date_dir.iterdir():
            if timestamp_dir.is_dir():
                archive_timestamp_dir = archive_date_dir / timestamp_dir.name
                archive_timestamp_dir.mkdir(exist_ok=True)

                if next(timestamp_dir.glob("*.jpg"), None):
                    full_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_full.zip"
                    subprocess.run([
                        "zip", "-q", "-r", "-u", "-0", full_zip_path, ".", "-i", "*.jpg",
                        "-x", "crop/*", "-x", "overlay/*"
                    ], cwd=timestamp_dir, check=False)

                crop_dir = timestamp_dir / "crop"
                if crop_dir.exists() and crop_dir.is_dir() and next(crop_dir.glob("**/*.jpg"), None):
                    crop_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_crop.zip"
                    subprocess.run([
                        "zip", "-q", "-r", "-u", "-0", crop_zip_path, ".", "-i", "*.jpg",
                    ], cwd=crop_dir, check=False)

                overlay_dir = timestamp_dir / "overlay"
                if overlay_dir.exists() and overlay_dir.is_dir() and next(overlay_dir.glob("*.jpg"), None):
                    overlay_zip_path = archive_timestamp_dir / f"{timestamp_dir.name}_overlay.zip"
                    subprocess.run([
                        "zip", "-q", "-r", "-u", "-0", overlay_zip_path, ".", "-i", "*.jpg",
                    ], cwd=overlay_dir, check=False)

                for file_path in timestamp_dir.glob("*.*"):
                    if file_path.is_file() and file_path.suffix.lower() != ".jpg":
                        subprocess.run(["rsync", "-a", "-u", file_path, archive_timestamp_dir], check=False)

    disk_free = round(psutil.disk_usage("/").free / 1048576)
    while dirs_orig and disk_free < low_diskspace:
        shutil.rmtree(dirs_orig.pop(0), ignore_errors=True)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

    return archive_path


def upload_data(data_path, archive_path, content="crop"):
    """Upload archived data to cloud storage provider."""
    # This example uses MinIO as storage provider
    # Rclone configuration and providers: https://rclone.org/docs/
    # Rclone copy options: https://rclone.org/commands/rclone_copy/#options

    #rclone_config_path = Path.home() / ".config" / "rclone" / "rclone.conf"  # default config path

    if content == "full":
        exclude_filter = ["**/*_crop.zip", "**/*_overlay.zip"]
    elif content == "crop":
        exclude_filter = ["**/*_full.zip", "**/*_overlay.zip"]
    elif content == "metadata":
        exclude_filter = ["**/*.zip", "**/*.jpg"]
    else:
        exclude_filter = ["**/*_overlay.zip"]  # upload all data except overlay frames

    rclone_cmd = [
        "rclone", "copy",
        #f"--config={rclone_config_path}",  # use custom rclone config file (optional)
        #"--progress",          # show transfer progress in terminal (optional)
        "--update",             # skip files that are newer on the destination
        "--transfers", "1",     # limit number of parallel file transfers
        "--buffer-size", "4M",  # limit memory buffer size (MB) when reading file(s)
        "--bwlimit", "4M",      # limit bandwidth (MB/s)
        f"--log-file={data_path / 'rclone.log'}",
        "--log-level=INFO"      # set log level to DEBUG for more detailed output
    ]

    for pattern in exclude_filter:
        rclone_cmd.extend(["--exclude", pattern])

    rclone_cmd.extend([str(archive_path.parent), "minio:bucket"])

    subprocess.run(rclone_cmd, check=False)
