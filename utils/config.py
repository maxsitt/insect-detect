"""Utility functions for parsing of configuration files.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Class:
    DotDict: Dictionary that allows dot notation access to nested keys.
Functions:
    parse_json(): Load JSON configuration file and return dot notation accessible dictionary.
    parse_yaml(): Load YAML configuration file and return dot notation accessible dictionary.
"""

import json

import yaml


class DotDict(dict):
    """Dictionary that allows dot notation access to nested keys."""
    def __getattr__(self, key):
        value = self[key]
        return DotDict(value) if isinstance(value, dict) else value


def parse_json(config_path):
    """Load JSON configuration file and return dot notation accessible dictionary."""
    with open(config_path, "r", encoding="utf-8") as config:
        return DotDict(json.load(config))


def parse_yaml(config_path):
    """Load YAML configuration file and return dot notation accessible dictionary."""
    with open(config_path, "r", encoding="utf-8") as config:
        return DotDict(yaml.safe_load(config))
