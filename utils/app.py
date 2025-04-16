"""Utility functions for NiceGUI-based web app.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    create_duration_inputs(): Create hours and minutes input fields for a specific duration type.
    convert_duration(): Convert minutes to a dictionary with hours, minutes, and total values.
    grid_separator(): Create a horizontal separator line for a 2-column grid layout.
    on_duration_change(): Convert hours and minutes to total minutes and update config.
    validate_number(): Validate that number is within the required range.
"""

from nicegui import app, ui


def create_duration_inputs(duration_type, label_text, tooltip_text=None):
    """Create hours and minutes input fields for a specific duration type."""
    if duration_type == "default":
        duration_setting = app.state.rec_durations["default"]
    else:
        duration_setting = app.state.rec_durations["battery"][duration_type]

    label = ui.label(label_text).classes("font-bold")
    if tooltip_text:
        label.tooltip(tooltip_text)

    with ui.row(align_items="center").classes("w-full gap-2"):
        (ui.number(label="Hours", placeholder=1, min=0, max=24, precision=0, step=1,
                   on_change=lambda e: on_duration_change(e, duration_type, "hours"),
                   validation={"Required value between 0-24": lambda v: validate_number(v, 0, 24)})
         .bind_value(duration_setting, "hours")).classes("flex-1")
        (ui.number(label="Minutes", placeholder=0, min=0, max=59, precision=0, step=1,
                   on_change=lambda e: on_duration_change(e, duration_type, "minutes"),
                   validation={"Required value between 0-59": lambda v: validate_number(v, 0, 59)})
         .bind_value(duration_setting, "minutes")).classes("flex-1")


def convert_duration(total_minutes):
    """Convert minutes to a dictionary with hours, minutes, and total values."""
    return {"hours": total_minutes // 60, "minutes": total_minutes % 60, "total": total_minutes}


def grid_separator():
    """Create a horizontal separator line for a 2-column grid layout."""
    with ui.row().classes("w-full col-span-2 py-0 my-0"):
        ui.element("div").classes("w-full border-t border-gray-700")


async def on_duration_change(e, duration_type, field_type):
    """Convert hours and minutes to total minutes and update config."""
    if e.value is not None:
        if duration_type == "default":
            duration_setting = app.state.rec_durations["default"]
            config_target = app.state.config_updates["recording"]["duration"]
        else:
            duration_setting = app.state.rec_durations["battery"][duration_type]
            config_target = app.state.config_updates["recording"]["duration"]["battery"]

        duration_setting[field_type] = int(e.value)
        hours = duration_setting.get("hours", 0) or 0  # default to 0 if not set
        minutes = duration_setting.get("minutes", 0) or 0
        total_minutes = (hours * 60) + minutes

        duration_setting["total"] = total_minutes
        if duration_type == "default":
            config_target["default"] = total_minutes
        else:
            config_target[duration_type] = total_minutes


def validate_number(value, min_value, max_value, multiple=None):
    """Validate that number is within the required range."""
    if multiple is None:
        return value is not None and (min_value <= value <= max_value)
    return value is not None and (min_value <= value <= max_value) and value % multiple == 0
