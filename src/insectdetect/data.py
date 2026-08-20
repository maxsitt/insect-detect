"""Utility functions for data management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    session_content_stats(): Get the number of files and their size per content category.
    list_sessions(): Enumerate recording sessions with file counts, sizes and archive status.
    iter_session_entries(): Collect the files of a recording session that match a content selection.
    build_zip_stream(): Queue files into an uncompressed zip archive that is generated while it is streamed.
    archive_data(): Copy data of a recording session + logs to archive directory and manage disk space.
    upload_data(): Upload the archived data of a recording session to a cloud storage provider via rclone.
"""

import logging
import shutil
import subprocess
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from zipstream import ZipStream

from insectdetect.constants import (
    ARCHIVE_PATH,
    DATA_PATH,
    HOSTNAME,
    LOGS_PATH,
    SUBDIR_NAMES,
    ContentSelection,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Summary of a single recording session directory.

    Attributes:
        date:     Name of the date directory ('YYYY-MM-DD').
        session:  Name of the session directory ('YYYY-MM-DD_HH-MM-SS').
        path:     Path to the session directory.
        counts:   Number of files per content category.
        sizes:    Total byte size per content category.
        archived: True if a complete zip file exists for every non-empty image category.
    """

    date: str
    session: str
    path: Path
    counts: dict[str, int] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    archived: bool = False

    @property
    def total_files(self) -> int:
        """Total number of files across all content categories."""
        return sum(self.counts.values())

    @property
    def total_bytes(self) -> int:
        """Total byte size across all content categories."""
        return sum(self.sizes.values())


def _sum_file_sizes(file_paths: Iterable[Path]) -> tuple[int, int]:
    """Return the number of files and their total byte size, skipping inaccessible files."""
    count = 0
    total = 0
    for file_path in file_paths:
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
        count += 1
    return count, total


def session_content_stats(session_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Get the number of files and their size per content category.

    Args:
        session_path: Path to the recording session directory.

    Returns:
        Tuple of (file counts, byte sizes), both keyed by content category.
    """
    counts: dict[str, int] = {"full": 0, "metadata": 0}
    sizes: dict[str, int] = {"full": 0, "metadata": 0}

    # Detection frames and metadata/config files are stored directly in the session directory
    for file_path in session_path.glob("*.*"):
        try:
            file_size = file_path.stat().st_size
        except OSError:
            continue
        category = "full" if file_path.suffix.lower() == ".jpg" else "metadata"
        counts[category] += 1
        sizes[category] += file_size

    # Cropped, timelapse and overlay frames are stored in dedicated subdirectories
    for category in SUBDIR_NAMES:
        counts[category], sizes[category] = _sum_file_sizes(
            (session_path / category).glob("**/*.jpg")
        )

    return counts, sizes


def list_sessions() -> list[SessionInfo]:
    """Enumerate recording sessions with file counts, sizes and archive status.

    Returns:
        List of session summaries, sorted newest first.
    """
    if not DATA_PATH.is_dir():
        return []

    sessions: list[SessionInfo] = []

    for date_dir in sorted((d for d in DATA_PATH.iterdir() if d.is_dir()), reverse=True):
        for session_dir in sorted((s for s in date_dir.iterdir() if s.is_dir()), reverse=True):
            counts, sizes = session_content_stats(session_dir)

            sessions.append(SessionInfo(date=date_dir.name, session=session_dir.name,
                                        path=session_dir, counts=counts, sizes=sizes,
                                        archived=_is_session_archived(session_dir)))

    return sessions


def iter_session_entries(
    session_path: Path,
    content: ContentSelection = "all"
) -> list[tuple[str, Path]]:
    """Collect the files of a recording session that match a content selection.

    The metadata files and config snapshots are included in every selection.
    Does not stat() the files, as ZipStream.add_path() already does so.

    Args:
        session_path: Path to the recording session directory.
        content:      Content category to select.

    Returns:
        List of (archive member name, file path) tuples, sorted within each category.
    """
    entries: list[tuple[str, Path]] = []
    include_all = content == "all"

    # Detection frames and metadata/config files are stored directly in the session directory
    for file_path in sorted(session_path.glob("*.*")):
        if file_path.suffix.lower() != ".jpg" or include_all or content == "full":
            entries.append((file_path.name, file_path))

    # Cropped, timelapse and overlay frames are stored in dedicated subdirectories,
    # of which 'all' includes all but the overlay frames (as does upload_data())
    if include_all:
        categories = tuple(c for c in SUBDIR_NAMES if c != "overlays")
    else:
        categories = (content,) if content in SUBDIR_NAMES else ()

    for category in categories:
        category_path = session_path / category
        base_path = session_path if include_all else category_path
        for file_path in sorted(category_path.glob("**/*.jpg")):
            entries.append((file_path.relative_to(base_path).as_posix(), file_path))

    return entries


def build_zip_stream(entries: list[tuple[str, Path]], sized: bool = True) -> ZipStream:
    """Queue files into an uncompressed zip archive that is generated while it is streamed.

    Args:
        entries: List of (archive member name, file path) tuples, as returned by iter_session_entries().
        sized:   If True, the archive size is calculated while queueing the files and
                 can be read via len() to serve a 'Content-Length' header. The queued
                 files must not change size afterwards, as a mismatch raises a
                 RuntimeError mid-stream. Set to False for files that are still being
                 written to, e.g. the log file of the running web app.

    Returns:
        ZipStream that yields consecutive byte chunks of the archive when iterated.
    """
    zip_stream = ZipStream(sized=sized)  # compress_type defaults to ZIP_STORED (no compression)
    for arcname, file_path in entries:
        zip_stream.add_path(file_path, arcname, recurse=False)

    return zip_stream


def _zip_images(zip_path: Path, source_path: Path, exclude: Iterable[str] = ()) -> bool:
    """Pack all .jpg files of a directory into an uncompressed zip file.

    An existing zip file is updated instead of being rewritten, so archiving
    a session twice only adds files that are missing or newer.

    Args:
        zip_path:    Path to the zip file to create or update.
        source_path: Path to the directory containing the .jpg files.
        exclude:     Glob patterns (relative to 'source_path') to exclude.

    Returns:
        True if the zip file was written successfully.
    """
    zip_cmd = ["zip", "-q", "-r", "-u", "-0", zip_path, ".", "-i", "*.jpg"]
    for pattern in exclude:
        zip_cmd.extend(["-x", pattern])

    # Exit code 12 means "nothing to do" and is returned if the zip file is already up to date
    returncode = subprocess.run(zip_cmd, cwd=source_path, check=False).returncode
    if returncode not in (0, 12):
        logger.error("Could not archive '%s' (zip exit code: %s)", source_path, returncode)
        return False

    return True


def _archived_session_path(session_path: Path) -> Path:
    """Return the path of a recording session inside the archive directory."""
    return ARCHIVE_PATH / session_path.parent.name / session_path.name


def _is_session_archived(session_path: Path) -> bool:
    """Check if a complete zip file exists for every non-empty image category of a session."""
    archive_session_path = _archived_session_path(session_path)
    if not archive_session_path.is_dir():
        return False

    categories = ["full"] if next(session_path.glob("*.jpg"), None) else []
    categories += [category for category in SUBDIR_NAMES
                   if next((session_path / category).glob("**/*.jpg"), None)]

    return all(zipfile.is_zipfile(archive_session_path / f"{session_path.name}_{category}.zip")
               for category in categories)


def archive_data(session_path: Path, disk_low: int = 5000) -> bool:
    """Copy data of a recording session + logs to archive directory and manage disk space.

    Only the given recording session is archived, which keeps the processing time after
    a session short and independent of the amount of data that is still on the microSD card.

    Detection, timelapse, overlay and cropped frames are stored in separate uncompressed
    zip files. Metadata, log and config files are copied directly via rsync. The oldest
    original date directories are deleted when free disk space drops below the 'disk_low'
    threshold (in MB), but only if all of their sessions were archived.

    Args:
        session_path: Path to the recording session directory to archive.
        disk_low:     Free disk space threshold (MB) below which original data is deleted.

    Returns:
        True if all zip files of the session were written successfully.
    """
    archive_session_path = _archived_session_path(session_path)
    archive_session_path.mkdir(parents=True, exist_ok=True)
    archive_failed = False

    # Copy root-level files to archive root
    for file_path in DATA_PATH.glob("*.*"):
        if file_path.is_file():
            subprocess.run(["rsync", "-a", "-u", file_path, ARCHIVE_PATH], check=False)

    # Copy log files to archive logs subdirectory
    if LOGS_PATH.is_dir():
        archive_logs_path = ARCHIVE_PATH / "logs"
        archive_logs_path.mkdir(exist_ok=True)
        for log_file in LOGS_PATH.glob("*.log"):
            subprocess.run(["rsync", "-a", "-u", log_file, archive_logs_path], check=False)

    # Pack full frames into zip, excluding the category subdirectories
    if next(session_path.glob("*.jpg"), None):
        full_zip_path = archive_session_path / f"{session_path.name}_full.zip"
        if not _zip_images(full_zip_path, session_path,
                           exclude=[f"{name}/*" for name in SUBDIR_NAMES]):
            archive_failed = True

    # Pack cropped, overlay and timelapse frames into one zip file per category
    for category in SUBDIR_NAMES:
        category_path = session_path / category
        if next(category_path.glob("**/*.jpg"), None):
            zip_path = archive_session_path / f"{session_path.name}_{category}.zip"
            if not _zip_images(zip_path, category_path):
                archive_failed = True

    # Copy non-image files (e.g. metadata CSV, config JSON) directly
    for file_path in session_path.glob("*.*"):
        if file_path.is_file() and file_path.suffix.lower() != ".jpg":
            subprocess.run(["rsync", "-a", "-u", file_path, archive_session_path], check=False)

    # Never delete original data if a zip file could not be written, as the archived
    # copy might be incomplete (recording is stopped once 'storage.disk_min' is reached)
    disk_free = round(psutil.disk_usage("/").free / 1048576)
    if archive_failed:
        if disk_free < disk_low:
            logger.error("Kept all original data despite low disk space (%s MB), as archiving failed",
                         disk_free)
        return False

    # Delete oldest original date directories until free disk space exceeds threshold,
    # skipping directories that contain sessions which were never archived
    for date_dir in sorted(d for d in DATA_PATH.iterdir() if d.is_dir()):
        if disk_free >= disk_low:
            break
        if not all(_is_session_archived(s) for s in date_dir.iterdir() if s.is_dir()):
            logger.warning("Kept '%s' despite low disk space (%s MB), as it was not fully archived",
                           date_dir.name, disk_free)
            continue
        shutil.rmtree(date_dir, ignore_errors=True)
        disk_free = round(psutil.disk_usage("/").free / 1048576)

    return True


def _rclone_copy(
    source_path: Path,
    remote_path: str,
    exclude: Iterable[str] = (),
    max_depth: int | None = None,
    no_traverse: bool = False
) -> bool:
    """Copy a local directory to a remote destination via rclone.

    Args:
        source_path: Path to the local directory to copy.
        remote_path: Remote destination in rclone syntax ('<remote>:<bucket>/<path>').
        exclude:     Glob patterns (relative to 'source_path') to exclude.
        max_depth:   Maximum directory depth to descend into (None for unlimited).
        no_traverse: If True, the destination is not listed. Set to True when copying
                     few files to a destination that already contains many files.

    Returns:
        True if rclone finished successfully.
    """
    rclone_cmd: list[str] = [
        "rclone", "copy",
        "--update",             # skip files that are newer on the destination
        "--transfers", "1",     # limit number of parallel file transfers (default: 4)
        "--checkers", "2",      # limit number of parallel file checks (default: 8)
        "--buffer-size", "4M",  # limit memory buffer size (MiB) per transfer (default: 16 MiB)
        "--bwlimit", "4M",      # limit bandwidth (MiB/s, default: unlimited)
        "--contimeout", "30s",  # limit time to establish a connection (default: 1 min)
        "--timeout", "60s",     # limit idle time of an established connection (default: 5 min)
        f"--log-file={LOGS_PATH / 'rclone.log'}",  # write log messages to file instead of stdout
        "--log-level=INFO",     # set log level to INFO (DEBUG for more detailed output)
    ]

    if max_depth is not None:
        rclone_cmd.extend(["--max-depth", str(max_depth)])
    if no_traverse:
        rclone_cmd.append("--no-traverse")
    for pattern in exclude:
        rclone_cmd.extend(["--exclude", pattern])

    rclone_cmd.extend([str(source_path), remote_path])

    if subprocess.run(rclone_cmd, check=False).returncode != 0:
        logger.error("Could not upload '%s' to '%s' - see rclone.log", source_path, remote_path)
        return False

    return True


def upload_data(session_path: Path, content: ContentSelection = "crops") -> bool:
    """Upload the archived data of a recording session to a cloud storage provider via rclone.

    Only the archived session is uploaded, together with the log files and the metadata
    files at archive root. Sessions that were recorded before uploading was enabled
    are never uploaded.

    Selects which archived zip files to upload based on 'content'.
    All options include metadata, log and config files.

    Content options:
    - all:       upload all data except overlay frames
    - full:      upload only full frame zip files
    - crops:     upload only cropped detection zip files (default)
    - overlays:  upload only overlay frame zip files
    - timelapse: upload only timelapse frame zip files
    - metadata:  upload only metadata and log files, no images

    Args:
        session_path: Path to the recording session directory that was archived.
        content:      Content selection string (from 'config.storage.upload.content').

    Returns:
        True if both the session and the metadata upload finished successfully.
    """
    # This example uses MinIO as storage provider
    # The remote layout mirrors the archive directory ('<bucket>/<device_id>/<date>/<session>')
    remote_root = f"minio:bucket/{HOSTNAME}"

    exclude_patterns: dict[str, list[str]] = {
        "all":       ["*_overlays.zip"],
        "full":      ["*_crops.zip", "*_timelapse.zip", "*_overlays.zip"],
        "crops":     ["*_full.zip", "*_timelapse.zip", "*_overlays.zip"],
        "overlays":  ["*_full.zip", "*_crops.zip", "*_timelapse.zip"],
        "timelapse": ["*_full.zip", "*_crops.zip", "*_overlays.zip"],
        "metadata":  ["*.zip", "*.jpg"],
    }
    exclude_filter = exclude_patterns.get(content, exclude_patterns["all"])

    # Upload the archived recording session
    archive_session_path = _archived_session_path(session_path)
    session_relpath = archive_session_path.relative_to(ARCHIVE_PATH).as_posix()
    session_ok = _rclone_copy(archive_session_path, f"{remote_root}/{session_relpath}",
                              exclude=exclude_filter)

    # Upload the metadata files at archive root and the log files one level below, without
    # descending into the session directories (already covered by the upload above)
    metadata_ok = _rclone_copy(ARCHIVE_PATH, remote_root, max_depth=2, no_traverse=True)

    return session_ok and metadata_ok
