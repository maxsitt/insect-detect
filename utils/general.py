"""Utility functions for data management and general functionality.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    create_signal_handler(): Create signal handler for a received signal.
    save_encoded_frame(): Save MJPEG-encoded frame to .jpg file.
    archive_data(): Archive all captured data + logs and manage disk space.
    upload_data(): Upload archived data to cloud storage provider.
"""

import shutil
import subprocess
from pathlib import Path

import psutil


def create_signal_handler(external_shutdown):
    """Create signal handler for a received signal."""
    def signal_handler(sig, frame):
        """Handle a received signal by raising a SystemExit exception."""
        external_shutdown.set()
        raise SystemExit

    return signal_handler


def save_encoded_frame(save_path, timestamp_str, frame):
    """Save MJPEG-encoded frame to .jpg file."""
    with open(save_path / f"{timestamp_str}.jpg", "wb") as jpg:
        jpg.write(frame)


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


def upload_data(data_path, archive_path):
    """Upload archived data to cloud storage provider."""
    # this example uses MinIO as storage provider
    # for more providers and config setup see https://rclone.org/#providers
    # for more options see https://rclone.org/commands/rclone_copy/#options
    subprocess.run(["rclone", "copy",
                    #"--progress",           # show transfer progress in terminal (optional)
                    "--update",             # skip files that are newer on the destination
                    "--transfers", "1",     # number of parallel file transfers
                    "--buffer-size", "4M",  # memory buffer size when reading file(s)
                    "--bwlimit", "4M",      # bandwidth limit in MB/s
                    f"--log-file={data_path / 'rclone.log'}",
                    "--log-level=INFO",
                    archive_path.parent, "minio:bucket"],
                    check=False)
