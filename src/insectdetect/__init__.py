"""Software package for automated insect monitoring with the Insect Detect camera trap.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Entry Points (via pyproject.toml scripts):
    startup: uv run startup
    capture: uv run capture
    webapp:  uv run webapp

Modules:
    startup:     Run configured startup sequence at boot (requires enabled systemd service).
    capture:     Capture detection-triggered images and save model/tracker metadata from OAK camera.
    webapp:      Stream OAK camera live feed and configure settings via browser-based web app.
    config:      Classes and functions for configuration file management.
    data:        Utility functions for data management.
    metrics:     Utility functions for system metrics and information logging.
    network:     Utility functions for network management via NetworkManager.
    oak:         Utility functions for OAK camera pipeline creation and metadata conversion.
    postprocess: Utility functions for post-processing of captured images.
    power:       Utility functions for power management initialization and state handling.
    wittypi:     Utility class for reading status information from Witty Pi 4 L3V7.
"""
