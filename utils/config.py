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
    check_config_changes(): Check if an updated config has any changes to the original.
    update_config_selector(): Update the config selector file to point to the active configuration.
    update_config_file(): Update existing or create new config based on template and save to file.
    update_nested_dict(): Update nested dictionary recursively. Replace 'None' with default value for required fields.
    sanitize_config(): Mask sensitive information in config (e.g. passwords).
"""

import copy
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


def check_config_changes(original, updates):
    """Check if an updated config has any changes to the original."""
    return json.dumps(dict(original), sort_keys=True) != json.dumps(dict(updates), sort_keys=True)


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


def update_config_file(config_path, config_template_path, config_updates, config, optional_fields=None):
    """Update existing or create new config based on template and save to file."""
    ruamel_yaml = ruamel.yaml.YAML()
    ruamel_yaml.width = 150  # maximum line width before wrapping
    ruamel_yaml.preserve_quotes = True  # preserve all comments
    ruamel_yaml.boolean_representation = ["false", "true"]  # ensure lowercase representation
    ruamel_yaml.indent(mapping=2, sequence=4, offset=2)  # indentation for nested structures

    with open(config_template_path, "r", encoding="utf-8") as file:
        config_template = ruamel_yaml.load(file)

    update_nested_dict(config_template, dict(config_updates), dict(config), optional_fields)

    with open(config_path, "w", encoding="utf-8") as file:
        ruamel_yaml.dump(config_template, file)


def update_nested_dict(template, updates, defaults, optional_fields=None, path=""):
    """Update nested dictionary recursively. Replace 'None' with default value for required fields."""
    if optional_fields is None:
        optional_fields = set()
    optional_field_names = {"ssid", "password"}  # not required (can always be empty/None)

    for key, value in updates.items():
        current_path = f"{path}.{key}" if path else key
        if (isinstance(value, dict) and
            isinstance(template.get(key), dict) and
            isinstance(defaults.get(key), dict)):
            update_nested_dict(template[key], value, defaults[key], optional_fields, current_path)
        else:
            is_optional = current_path in optional_fields or key in optional_field_names
            if value is None and not is_optional:
                template[key] = defaults[key]
            else:
                template[key] = value


def sanitize_config(config):
    """Mask sensitive information in config (e.g. passwords)."""
    sanitized = copy.deepcopy(dict(config))

    if "network" in sanitized:
        if "wifi" in sanitized["network"]:
            for wifi in sanitized["network"]["wifi"]:
                if "password" in wifi:
                    wifi["password"] = "[REDACTED]"
        if "hotspot" in sanitized["network"] and "password" in sanitized["network"]["hotspot"]:
            sanitized["network"]["hotspot"]["password"] = "[REDACTED]"

    return sanitized
