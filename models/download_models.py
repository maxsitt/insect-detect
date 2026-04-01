"""Download and verify model archives defined in models/models.json.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    compute_sha256(): Compute and return the SHA-256 checksum of a file.
    download_file(): Download a file from the given URL to a local destination path.
    download_models(): Download and verify all model archives defined in models/models.json.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Models JSON registry file path
MODELS_PATH = Path(__file__).resolve().parent
MODELS_JSON = MODELS_PATH / "models.json"


def compute_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute and return the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, timeout: int = 60) -> None:
    """Download a file from the given URL to a local destination path.

    Shows download progress in MB and percentage. Removes the destination
    file if the download fails to avoid leaving partial files on disk.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "insect-detect"})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            dest_path.open("wb") as f
        ):
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({downloaded / total * 100:.0f}%)",
                        end="", flush=True
                    )
        print()
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise


def download_models() -> bool:
    """Download and verify all model archives defined in models/models.json.

    Skips models that are already present. Returns True if all models were
    downloaded and verified successfully, False if any model failed.
    """
    if not MODELS_JSON.exists():
        raise FileNotFoundError(f"Models registry not found: '{MODELS_JSON}'")

    registry = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    models = registry.get("models", [])

    if not models:
        print("No models defined in 'models.json'.")
        return True

    success = True
    for model in models:
        name: str = model["name"]
        url: str = model["url"]
        expected_sha256: str = model["sha256"]

        archive_name = url.split("/")[-1]
        archive_path = MODELS_PATH / archive_name

        if archive_path.exists():
            print(f"Model '{name}' is already present, skipping.")
            continue

        print(f"Downloading model '{name}'...")
        try:
            download_file(url, archive_path)
        except Exception as e:
            print(f"ERROR: Failed to download '{name}': {e}")
            success = False
            continue

        print("Verifying checksum...")
        actual = compute_sha256(archive_path)
        if actual != expected_sha256:
            print(
                f"ERROR: SHA-256 mismatch for '{archive_name}'.\n"
                f"  expected: {expected_sha256}\n"
                f"  got:      {actual}"
            )
            archive_path.unlink()
            success = False
            continue

        print(f"Model '{name}' downloaded successfully.")

    return success


if __name__ == "__main__":
    sys.exit(0 if download_models() else 1)
