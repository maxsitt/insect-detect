"""Utility functions for configuration file management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Class:
    DotDict: Dictionary that allows dot notation access to nested keys.
Functions:
    parse_json(): Load JSON configuration file and return dot notation accessible dictionary.
    parse_yaml(): Load YAML configuration file and return dot notation accessible dictionary.
    check_config_changes(): Check if an updated config section has any changes to the original.
    update_config_selector(): Update the config selector file to point to the active configuration.
    update_nested_dict(): Update nested dictionary recursively. Replace 'None' with default value.
"""

import json

import ruamel.yaml
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


def check_config_changes(original, updates, section):
    """Check if an updated config section has any changes to the original."""
    return json.dumps(dict(original[section]), sort_keys=True) != json.dumps(updates[section], sort_keys=True)


def update_config_selector(base_path, config_active):
    """Update the config selector file to point to the active configuration."""
    ruamel_yaml = ruamel.yaml.YAML()
    ruamel_yaml.width = 150  # maximum line width before wrapping
    ruamel_yaml.preserve_quotes = True  # preserve all comments

    config_selector_path = base_path / "configs" / "config_selector.yaml"

    with open(config_selector_path, "r", encoding="utf-8") as file:
        config_selector = ruamel_yaml.load(file)

    config_selector["config_active"] = config_active

    with open(config_selector_path, "w", encoding="utf-8") as file:
        ruamel_yaml.dump(config_selector, file)


def update_nested_dict(template, updates, defaults):
    """Update nested dictionary recursively. Replace 'None' with default value."""
    for key, value in updates.items():
        if (isinstance(value, dict) and
            isinstance(template.get(key), dict) and
            isinstance(defaults.get(key), dict)):
            update_nested_dict(template[key], value, defaults[key])
        else:
            template[key] = value if value is not None else defaults[key]
