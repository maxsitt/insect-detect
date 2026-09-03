"""Focus sweep and temperature-compensated manual focus correction for the OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

The OAK lens barrel and VCM actuator expand as the chip heats up during a recording session,
which moves the plane of best focus away from the camera and progressively blurs the images.
The lens position code is roughly linear in diopters and the expansion is roughly linear in
temperature, so the correction is linear in chip temperature.

The compensation has three parts:
    1. Anchor:      A (lens position, chip temperature) pair. The reference pair is set in
                    the web app, an optional focus sweep refines it into the working anchor.
    2. Feedforward: lens_pos = anchor_pos + coeff * (chip_temp - anchor_temp),
                    applied periodically during a recording session.
    3. Closed loop: An optional focus sweep at session start re-anchors the pair. It corrects
                    slow mechanical drift, and can improve a focus position set in the web app.

Sharpness is measured as the size in bytes of the MJPEG-encoded frame: at fixed encoder quality
the compressed size tracks high-frequency image content (no decoding required on the Raspberry Pi).

Compensation stays inactive until a focus position has been set in the web app:
a lens position without the chip temperature it was set at cannot be corrected.

Classes:
    ManualFocus:     Persisted focus reference, working anchor and session state.
    SweepSample:     One measured position of a focus sweep.
    SweepResult:     Outcome of a focus sweep, including all samples and guard results.
    PeakFit:         Located sharpness peak of a focus sweep and the guard inputs.
    FocusController: Holds the anchor, runs the focus sweep and applies the lens position correction.

Functions:
    load_manual_focus():   Load the persisted manual focus file.
    save_manual_focus():   Write the persisted manual focus file.
    resolve_lens_pos():    Return the lens position to start the pipeline with in 'manual' focus mode.
    record_webapp_focus(): Store a focus position set from the web app as the new reference.
    get_frame():           Get the next synchronized frame from the pipeline.
    wait_for_lens_pos():   Drain frames until one reports the commanded lens position.
"""

import json
import logging
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import depthai as dai

from insectdetect.config import AppConfig
from insectdetect.constants import (
    FOCUS_ANCHOR_MAX_DRIFT,
    FOCUS_SWEEP_MAX_ISO,
    FOCUS_SWEEP_MIN_GAP,
    FOCUS_SWEEP_SPAN,
    LENS_POS_DEFAULT,
    LENS_POS_MAX,
    LENS_POS_MIN,
    MANUAL_FOCUS_PATH,
)
from insectdetect.oak import get_chip_temp, send_camctrl

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Step size of the focus sweep, in lens position units (must stay at 1)
FOCUS_SWEEP_STEP: int = 1

# Number of frames discarded after each lens move, to absorb VCM settle + pipeline latency
FOCUS_SETTLE_FRAMES: int = 2

# Maximum time (seconds) to wait for a single frame during a sweep before giving up
FOCUS_FRAME_TIMEOUT: int = 5

# Consecutive frames with identical ISO and exposure required before locking AE/AWB + timeout
FOCUS_AE_SETTLE_FRAMES: int = 3
FOCUS_AE_SETTLE_TIMEOUT: int = 20

# Half-width (lens position units) of the parabola fit around the sharpest measured position
FOCUS_FIT_WINDOW: int = 4

# Minimum R^2 (goodness of fit) of the fitted parabola, to tell a real sharpness peak from noise
FOCUS_MIN_FIT_R2: float = 0.5

# Maximum brightness change during a sweep (fraction of the mean frame size) before it is rejected,
# as a sudden change (cloud, shadow, insect) tilts the fitted curve and moves the peak
FOCUS_MAX_DRIFT_FRAC: float = 0.02

# Required height of the sharpness peak as (max-min)/max of the frame sizes: a flat curve means no
# peak was found. Grows with the width of the resolved peak (K), bounded by the min and max values.
FOCUS_PEAK_RATIO_K: float = 0.0004
FOCUS_MIN_PEAK_RATIO: float = 0.002
FOCUS_MAX_PEAK_RATIO: float = 0.012

# Weight of a new sweep result when updating an anchor that came from a previous sweep,
# so a single bad sweep cannot move the anchor far
FOCUS_ANCHOR_SMOOTHING: float = 0.3

# Extra margin (lens position units) on top of rounding before the lens is actually moved,
# so small temperature fluctuations do not move the lens back and forth
FOCUS_DEADBAND: float = 0.6


@dataclass
class ManualFocus:
    """Persisted focus reference, working anchor and session state."""
    lens_pos: int | None = None           # reference lens position set in the web app
    temp: float | None = None             # chip temperature when the reference was set
    anchor_lens_pos: float | None = None  # working anchor, starts as the reference and is refined by sweeps
    anchor_temp: float | None = None      # chip temperature the working anchor was measured at
    anchor_source: str | None = None      # how the working anchor was set ('webapp' or 'sweep')
    session_end: str | None = None        # ISO timestamp of the last session end, used for the sweep gap gate
    updated: str | None = None            # ISO timestamp of the last write


@dataclass
class SweepSample:
    """One measured position of a focus sweep."""
    commanded_pos: int                 # lens position the sweep moved the lens to
    actual_pos: int                    # lens position the frame reports, to verify the move completed
    jpeg_bytes: int                    # size of the encoded frame, used as the sharpness metric
    iso: int                           # ISO sensitivity of the frame, constant while AE is locked
    exp_ms: float                      # exposure time (ms) of the frame, constant while AE is locked
    t_rel: float = 0.0                 # seconds since the first sample of the sweep
    is_bookend: bool = False           # repeat of the first position, used to measure drift
    jpeg_bytes_detrended: float = 0.0  # jpeg_bytes with the measured drift removed


@dataclass
class SweepResult:
    """Outcome of a focus sweep, including all samples and guard results."""
    center_pos: int                    # lens position the sweep was centred on (temperature-corrected)
    span: int                          # half-width of the sweep in lens position units
    temp_mid: float | None = None      # chip temperature sampled at the sweep midpoint
    samples: list[SweepSample] = field(default_factory=list)  # all measured samples, bookend included
    best_pos: int | None = None        # rounded vertex if inside the swept range, else the argmax
    best_pos_fit: float | None = None  # parabola vertex before rounding to a lens position
    edge_hit: bool = False             # peak sits on the boundary of the swept range
    accepted: bool = False             # whether all acceptance guards passed
    reason: str = ""                   # why the sweep was rejected, or 'ok' when accepted


@dataclass
class PeakFit:
    """Located sharpness peak of a focus sweep and the guard inputs."""
    best_pos: int | None = None    # rounded vertex if it lies inside the swept range, else the argmax
    argmax_pos: int | None = None  # lens position of the largest measured frame size
    vertex: float | None = None    # unrounded parabola vertex, None if the fit failed
    fit_ok: bool = False           # parabola fitted and its vertex lies inside the swept range
    r2: float = 0.0                # how much of the measured variation the parabola explains
    peak_ratio: float = 0.0        # (max-min)/max of the measured frame sizes
    edge_hit: bool = False         # largest size on the boundary, so no peak was resolved


def load_manual_focus(path: Path = MANUAL_FOCUS_PATH) -> ManualFocus:
    """Load the persisted manual focus file.

    Args:
        path: Path to the manual focus .json file.

    Returns:
        ManualFocus with the persisted values, or an empty one if missing or unreadable.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = set(ManualFocus.__dataclass_fields__)
        return ManualFocus(**{k: v for k, v in data.items() if k in known})
    except FileNotFoundError:
        logger.debug("No manual focus file found at %s", path)
    except Exception:
        logger.exception("Could not read manual focus file %s, starting fresh", path)
    return ManualFocus()


def save_manual_focus(state: ManualFocus, path: Path = MANUAL_FOCUS_PATH) -> None:
    """Write the persisted manual focus file.

    Writing to a sibling temp file and replacing the target in a single step means a
    power loss can only leave the old file or the temp file behind, never a half-written one.

    Args:
        state: ManualFocus to write.
        path:  Path to the manual focus .json file.
    """
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        state.updated = datetime.now().isoformat()  # noqa: DTZ005
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(asdict(state), file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        tmp.replace(path)
        logger.debug("Manual focus saved: %s", asdict(state))
    except Exception:
        logger.exception("Could not write manual focus file %s", path)
        tmp.unlink(missing_ok=True)


def resolve_lens_pos(path: Path = MANUAL_FOCUS_PATH) -> int:
    """Return the lens position to start the pipeline with in 'manual' focus mode.

    Args:
        path: Path to the manual focus .json file.

    Returns:
        Working anchor position, falling back to the reference position and then to
        LENS_POS_DEFAULT if no focus position has been set in the web app yet.
    """
    state = load_manual_focus(path)
    lens_pos = state.anchor_lens_pos if state.anchor_lens_pos is not None else state.lens_pos
    if lens_pos is None:
        logger.info("No manual focus position set, using the default %s", LENS_POS_DEFAULT)
        return LENS_POS_DEFAULT
    return _clamp(round(lens_pos))


def record_webapp_focus(lens_pos: int, temp: float | None) -> None:
    """Store a focus position set from the web app as the new reference.

    The OAK chip temperature is the second half of the anchor: without it the position cannot be
    corrected later. A new reference also resets the working anchor, superseding any earlier sweep.

    Args:
        lens_pos: Lens position set from the web app focus slider.
        temp:     OAK chip temperature in degrees Celsius, or None if unavailable.
    """
    if temp is None:
        logger.warning(
            "Focus position %s set from web app, but no OAK chip temperature available: "
            "no reference stored, so thermal compensation stays inactive", lens_pos
        )
        return

    state = load_manual_focus()
    state.lens_pos = lens_pos
    state.temp = round(temp, 2)
    state.anchor_lens_pos = float(lens_pos)
    state.anchor_temp = round(temp, 2)
    state.anchor_source = "webapp"
    save_manual_focus(state)
    logger.info("Focus reference set from web app: lens_pos=%s temp=%.1f °C", lens_pos, temp)


def _clamp(value: int, low: int = LENS_POS_MIN, high: int = LENS_POS_MAX) -> int:
    """Clamp a lens position to the valid range."""
    return max(low, min(high, value))


def get_frame(q_sync: dai.MessageQueue, timeout_s: float = FOCUS_FRAME_TIMEOUT) -> dai.ImgFrame | None:
    """Get the next synchronized frame from the pipeline.

    Args:
        q_sync:    depthai output queue for the Sync node.
        timeout_s: Maximum time to wait for a frame.

    Returns:
        Next available ImgFrame (MJPEG bitstream), or None on timeout/closed queue.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if q_sync.isClosed():
            logger.warning("q_sync closed while waiting for a frame")
            return None
        try:
            group = cast(dai.MessageGroup | None, q_sync.tryGet())
        except Exception:
            logger.exception("Error getting synced frame from q_sync")
            return None
        if group is not None:
            frame = cast(dai.ImgFrame | None, group["frames"])
            if frame is not None:
                return frame
        time.sleep(0.02)
    logger.warning("Timed out after %s s waiting for a frame", timeout_s)
    return None


def wait_for_lens_pos(q_sync: dai.MessageQueue, lens_pos: int, timeout_s: float = 5.0) -> bool:
    """Drain frames until one reports the commanded lens position.

    Args:
        q_sync:    depthai output queue for the Sync node.
        lens_pos:  Commanded lens position to wait for.
        timeout_s: Maximum time to wait.

    Returns:
        True if a frame reported the commanded position, False on timeout.
    """
    deadline = time.monotonic() + timeout_s
    last: int | None = None
    while time.monotonic() < deadline:
        frame = get_frame(q_sync, timeout_s=max(0.1, deadline - time.monotonic()))
        if frame is None:
            break
        last = frame.getLensPosition()
        if last == lens_pos:
            return True
    logger.warning(
        "Lens position %s not confirmed within %.0f s (last frame reported %s): the first images "
        "of this session may still be at the previous position", lens_pos, timeout_s, last
    )
    return False


def _required_peak_ratio(resolved_span: float) -> float:
    """Return the minimum relative peak height required to accept a sweep.

    A sweep centred off the optimum measures less of the curve on one side, so it rises less than
    a centred one. Scaling by what was actually resolved keeps those sweeps acceptable - they are
    the ones that need correcting. On the boundary this returns the floor, and the edge-hit guard
    rejects the sweep anyway.

    Args:
        resolved_span: Half-width in lens position units resolved around the peak.

    Returns:
        Minimum acceptable (max-min)/max ratio.
    """
    return min(FOCUS_MAX_PEAK_RATIO,
               max(FOCUS_MIN_PEAK_RATIO, FOCUS_PEAK_RATIO_K * resolved_span * resolved_span))


def _fit_parabola_vertex(xs: list[float], ys: list[float]) -> tuple[float | None, float]:
    """Fit y = a*x^2 + b*x + c by least squares and return the vertex position and fit quality.

    Args:
        xs: Lens positions.
        ys: Sharpness values (JPEG bytes, drift-corrected).

    Returns:
        Tuple of (vertex position, R^2). The vertex is None if no peak could be fitted; R^2 is
        what separates a sharpness curve from noise (see FOCUS_MIN_FIT_R2).
    """
    # Minimum of five points required to make R^2 meaningful
    if len(xs) < 5:
        return None, 0.0

    # Center the lens position values to reduce rounding errors
    x_center = statistics.fmean(xs)
    x_centered = [x - x_center for x in xs]

    # Least squares reduces to three 'normal equations' for a, b and c, held as a 3x3 matrix
    # of sums of powers of x with the right-hand side appended as a fourth column
    power_sums = [sum(x ** k for x in x_centered) for k in range(5)]
    sum_x2y = sum(x * x * y for x, y in zip(x_centered, ys))
    sum_xy = sum(x * y for x, y in zip(x_centered, ys))
    sum_y = sum(ys)

    augmented_matrix = [[power_sums[4], power_sums[3], power_sums[2], sum_x2y],
                        [power_sums[3], power_sums[2], power_sums[1], sum_xy],
                        [power_sums[2], power_sums[1], power_sums[0], sum_y]]

    # Gauss-Jordan elimination with partial pivoting: the largest leading value is swapped to the
    # top at each step to limit rounding error, and a near-zero one means there is no solution
    for i in range(3):
        pivot = max(range(i, 3), key=lambda r: abs(augmented_matrix[r][i]))
        if abs(augmented_matrix[pivot][i]) < 1e-12:
            return None, 0.0
        augmented_matrix[i], augmented_matrix[pivot] = augmented_matrix[pivot], augmented_matrix[i]
        for r in range(3):
            if r != i:
                factor = augmented_matrix[r][i] / augmented_matrix[i][i]
                for c in range(i, 4):
                    augmented_matrix[r][c] -= factor * augmented_matrix[i][c]

    # Elimination leaves the matrix diagonal, so each row now yields one coefficient directly
    a = augmented_matrix[0][3] / augmented_matrix[0][0]
    b = augmented_matrix[1][3] / augmented_matrix[1][1]
    c = augmented_matrix[2][3] / augmented_matrix[2][2]

    # R^2: the error left after fitting, against the error of simply guessing the mean every time
    mean_y = statistics.fmean(ys)
    ss_res = sum((y - (a * x * x + b * x + c)) ** 2 for x, y in zip(x_centered, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if a >= 0:
        return None, r2  # opens upward: a minimum, not a sharpness maximum
    return x_center - b / (2 * a), r2  # the vertex of a parabola sits at x = -b / 2a


def _fit_peak(samples: list[SweepSample]) -> PeakFit:
    """Locate the sharpness peak of a focus sweep.

    The argmax alone is unreliable: within the depth of field the curve is flat and adjacent
    positions differ by less than the frame-to-frame noise. A parabola fitted around the argmax
    locates the vertex from the sloped sides instead, which is far more stable.

    Args:
        samples: Measured sweep samples, ordered by commanded position, the bookend excluded and
                 the drift already removed (see _finish_sweep).

    Returns:
        PeakFit with the located peak, falling back to the argmax where the parabola is unusable.
    """
    # Fewer than three samples cannot have an interior peak: every index would be a boundary
    if len(samples) < 3:
        return PeakFit()

    sizes = [s.jpeg_bytes_detrended for s in samples]
    positions = [s.commanded_pos for s in samples]
    swept_min, swept_max = min(positions), max(positions)

    # Rough peak: simply the largest measured size. On the boundary it means the curve was
    # still rising where the sweep ended, so there is no top inside the sweep to resolve
    size_max, size_min = max(sizes), min(sizes)
    argmax_idx = sizes.index(size_max)
    argmax_pos = positions[argmax_idx]
    fit = PeakFit(
        best_pos=argmax_pos,
        argmax_pos=argmax_pos,
        peak_ratio=(size_max - size_min) / size_max if size_max > 0 else 0.0,
        edge_hit=argmax_idx in {0, len(samples) - 1}
    )

    # Refine it with a parabola over the samples around the argmax only,
    # so that a competing focus plane further out cannot tilt the fit
    window = [i for i, p in enumerate(positions) if abs(p - argmax_pos) <= FOCUS_FIT_WINDOW]
    fit.vertex, fit.r2 = _fit_parabola_vertex([float(positions[i]) for i in window],
                                              [sizes[i] for i in window])

    # A vertex beyond the measured positions is an extrapolation, not a measurement: keep the argmax
    if fit.vertex is None or not (swept_min <= fit.vertex <= swept_max):
        return fit

    fit.best_pos = round(fit.vertex)
    fit.fit_ok = True
    return fit


class FocusController:
    """Holds the anchor, runs the focus sweep and applies the lens position correction."""

    def __init__(self, config: AppConfig, state: ManualFocus) -> None:
        """Initialize the controller from config and the persisted manual focus file.

        Args:
            config: AppConfig containing all configuration settings.
            state:  ManualFocus loaded from disk.
        """
        compensation = config.camera.focus.compensation
        self.state = state
        self.coeff = compensation.coeff
        self.sweep_enabled = compensation.sweep
        self.temp: float | None = None
        self.cold_start = False
        self.gap_s: float | None = None

        # Reference lens position-temperature pair set in the web app
        self.ref_pos: int | None = state.lens_pos
        self.ref_temp: float | None = state.temp

        # Carried as a float: a sweep resolves sub-code precision and target_position() rounds at
        # the end, so rounding here would bias every correction of the session by up to half a code
        anchor = state.anchor_lens_pos if state.anchor_lens_pos is not None else state.lens_pos
        self.base_pos: float = float(anchor) if anchor is not None else float(LENS_POS_DEFAULT)
        self.base_temp: float | None = state.anchor_temp
        self.applied_pos: int = _clamp(round(self.base_pos))

        if config.camera.focus.mode != "manual":
            self.disabled_reason = (f"camera.focus.mode is '{config.camera.focus.mode}', "
                                    f"not 'manual'")
            self.enabled = False
            logger.info("Focus compensation is INACTIVE: %s", self.disabled_reason)
            return

        if not compensation.enabled:
            self.disabled_reason = "camera.focus.compensation.enabled is false"
        elif self.ref_pos is None or self.ref_temp is None:
            self.disabled_reason = "no (position, temperature) reference to correct from"
        else:
            self.disabled_reason = ""
        self.enabled = not self.disabled_reason

        logger.info(
            "FocusController initialized | enabled=%s coeff=%.3f sweep=%s reference=(%s, %s °C) "
            "anchor=(%.2f, %s °C) source=%s",
            self.enabled, self.coeff, self.sweep_enabled, self.ref_pos, self.ref_temp,
            self.base_pos, self.base_temp, state.anchor_source
        )
        if anchor is None:
            logger.warning("No focus position set in the web app, using the default lens "
                           "position %s", LENS_POS_DEFAULT)
        if not self.enabled:
            logger.warning("Focus compensation is INACTIVE: %s", self.disabled_reason)

    def _record_thermal_state(self, temp_now: float | None) -> None:
        """Record the current chip temperature and evaluate the gap since the previous session.

        A missing temperature is carried as None rather than substituted: the correction is a
        difference against the anchor temperature, so a placeholder would move the lens far.

        Args:
            temp_now: Current chip temperature in degrees Celsius, or None if unavailable.
        """
        self.gap_s = None
        if self.state.session_end:
            try:
                last_end = datetime.fromisoformat(self.state.session_end)
                self.gap_s = (datetime.now() - last_end).total_seconds()  # noqa: DTZ005
            except ValueError:
                logger.warning("Could not parse persisted session_end: %s", self.state.session_end)

        self.cold_start = self.gap_s is not None and self.gap_s >= FOCUS_SWEEP_MIN_GAP
        self.temp = temp_now

        if temp_now is None:
            logger.warning(
                "No OAK chip temperature at session start: focus correction stays inactive "
                "until one can be read"
            )
        else:
            logger.info(
                "Session thermal state | temp=%.1f °C gap=%s s cold_start=%s",
                temp_now, round(self.gap_s) if self.gap_s is not None else "unknown",
                self.cold_start
            )

    def update_temp(self, temp_now: float | None) -> None:
        """Update the current chip temperature, ignoring a failed reading.

        Args:
            temp_now: Current chip temperature in degrees Celsius, or None if unavailable.
        """
        if temp_now is not None:
            self.temp = temp_now

    def _corrected_pos(self) -> float:
        """Lens position the anchor predicts for the current chip temperature, unrounded.

        Returns:
            Corrected position, or the anchor position itself if no temperature is available.
        """
        if self.base_temp is None or self.temp is None:
            return self.base_pos
        return self.base_pos + self.coeff * (self.temp - self.base_temp)

    def target_position(self) -> tuple[int, bool]:
        """Compute the temperature-corrected target lens position, with the deadband applied.

        A deadband keeps the lens still while the unrounded target hovers near the rounding
        boundary, so temperature noise cannot flip the position back and forth.

        Returns:
            Tuple of (target_lens_position, clipped_to_valid_range).
        """
        raw = self._corrected_pos()
        if self.base_temp is None or self.temp is None:
            return _clamp(round(raw)), False
        if abs(raw - self.applied_pos) <= 0.5 + FOCUS_DEADBAND:
            return self.applied_pos, False
        target = round(raw)
        clamped = _clamp(target)
        return clamped, clamped != target

    def set_anchor(self, lens_pos: float, temp: float, source: str) -> None:
        """Store a new working anchor pair, blended with the previous one and clamped.

        A sweep result is blended with an existing sweep-derived anchor rather than replacing it.
        The position keeps the sub-code precision of the sweep fit, it is rounded at the end.

        Args:
            lens_pos: Anchored lens position, fractional where a sweep resolved it.
            temp:     Chip temperature at which the anchor was measured.
            source:   How the anchor was set ('webapp' or 'sweep').
        """
        if (
            source == "sweep"
            and self.state.anchor_source == "sweep"
            and self.base_temp is not None
            and 0.0 < FOCUS_ANCHOR_SMOOTHING < 1.0
        ):
            # Project previous position to this sweep's temperature before blending
            projected = self.base_pos + self.coeff * (temp - self.base_temp)
            blended = (1.0 - FOCUS_ANCHOR_SMOOTHING) * projected + FOCUS_ANCHOR_SMOOTHING * lens_pos
            logger.info(
                "Anchor smoothing | measured=%.2f previous=%.2f projected=%.2f (to %.1f °C) "
                "blended=%.2f (w=%.2f)",
                lens_pos, self.base_pos, projected, temp, blended, FOCUS_ANCHOR_SMOOTHING
            )
            lens_pos = blended

        lens_pos = self._clamp_to_reference(lens_pos, temp)
        self.base_pos = lens_pos
        self.base_temp = temp
        self.state.anchor_lens_pos = round(lens_pos, 2)
        self.state.anchor_temp = round(temp, 2)
        self.state.anchor_source = source
        logger.info("Anchor set | lens_pos=%.2f temp=%.1f °C source=%s", lens_pos, temp, source)

    def _clamp_to_reference(self, lens_pos: float, temp: float) -> float:
        """Bound the working anchor to FOCUS_ANCHOR_MAX_DRIFT around the projected reference.

        The reference is projected to the anchor temperature first: two lens positions are only
        comparable at the same temperature, since a hot anchor legitimately sits above a cold one
        for the very same focus plane.

        Args:
            lens_pos: Candidate anchor position.
            temp:     Chip temperature the candidate was measured at.

        Returns:
            The candidate, bounded to the allowed drift and to the valid range.
        """
        if self.ref_pos is None or self.ref_temp is None:
            return float(_clamp(round(lens_pos)))

        expected = self.ref_pos + self.coeff * (temp - self.ref_temp)
        low = max(float(LENS_POS_MIN), expected - FOCUS_ANCHOR_MAX_DRIFT)
        high = min(float(LENS_POS_MAX), expected + FOCUS_ANCHOR_MAX_DRIFT)
        clamped = max(low, min(high, lens_pos))
        if clamped != lens_pos:
            logger.warning(
                "Anchor %.2f pinned to drift clamp %.2f (reference %s at %.1f °C projects to %.2f "
                "at %.1f °C, max drift %s). Repeated clamping means the focus position set in the "
                "web app no longer matches the scene and should be set again.",
                lens_pos, clamped, self.ref_pos, self.ref_temp, expected, temp,
                FOCUS_ANCHOR_MAX_DRIFT
            )
        return clamped

    def sweep_allowed(self, iso: int | None = None, exp_ms: float | None = None) -> tuple[bool, str]:
        """Check the timing and light-level guards before running the focus sweep.

        Call once without arguments to evaluate the cheap timing gates before spending time
        on auto exposure, then again with converged values to apply the light guard.

        Args:
            iso:    ISO sensitivity of a probe frame, or None if not yet measured.
            exp_ms: Exposure time (ms) of a probe frame, or None if not yet measured.

        Returns:
            Tuple of (allowed, reason).
        """
        if not self.enabled:
            return False, self.disabled_reason
        if not self.sweep_enabled:
            return False, "camera.focus.compensation.sweep is false"
        if self.gap_s is None:
            return False, "no completed recording session yet (first run, or the last one was cut short)"
        if not self.cold_start:
            return False, (f"only {round(self.gap_s)} s since the previous session "
                           f"(< {FOCUS_SWEEP_MIN_GAP} s), lens not thermally settled")
        if iso is not None and iso > FOCUS_SWEEP_MAX_ISO:
            return False, f"scene too dark (ISO {iso} > {FOCUS_SWEEP_MAX_ISO})"

        light = (f"ISO {iso}, {exp_ms:.1f} ms" if iso is not None and exp_ms is not None
                 else "light level not yet measured")
        return True, f"cold start, {light}"

    def _set_lens_pos(self, q_camctrl: dai.InputQueue, pos: int) -> None:
        """Send a manual focus command for the given lens position."""
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        ctrl.setManualFocus(pos)
        send_camctrl(q_camctrl, ctrl)

    def _set_ae_awb_lock(self, q_camctrl: dai.InputQueue, locked: bool) -> None:
        """Lock or unlock auto exposure and auto white balance during a sweep."""
        try:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(locked)
            ctrl.setAutoWhiteBalanceLock(locked)
            send_camctrl(q_camctrl, ctrl)
            logger.debug("AE/AWB lock set to %s", locked)
        except Exception:
            logger.exception("Could not set AE/AWB lock to %s (continuing without lock)", locked)

    def _wait_for_ae_settle(self, q_sync: dai.MessageQueue) -> tuple[int | None, float | None]:
        """Wait until auto exposure stops adjusting, so the lock freezes a converged value.

        Args:
            q_sync: depthai output queue for the Sync node.

        Returns:
            Tuple of (iso, exposure_ms) of the last frame, or (None, None) on timeout.
        """
        deadline = time.monotonic() + FOCUS_AE_SETTLE_TIMEOUT
        started = time.monotonic()
        last: tuple[int, float] | None = None
        stable = 0

        while time.monotonic() < deadline:
            frame = get_frame(q_sync)
            if frame is None:
                logger.warning("AE settle wait aborted: no frames available")
                return (last[0], last[1]) if last else (None, None)
            current = (frame.getSensitivity(),
                       round(frame.getExposureTime().total_seconds() * 1000, 2))
            stable = stable + 1 if current == last else 1
            last = current
            if stable >= FOCUS_AE_SETTLE_FRAMES:
                logger.info("AE settled in %.1f s | iso=%s exp=%.2f ms",
                            time.monotonic() - started, current[0], current[1])
                return current

        logger.warning(
            "AE did not settle within %s s (last: iso=%s exp=%s ms), locking anyway",
            FOCUS_AE_SETTLE_TIMEOUT, last[0] if last else None, last[1] if last else None
        )
        return (last[0], last[1]) if last else (None, None)

    def run_sweep(
        self,
        q_sync: dai.MessageQueue,
        q_camctrl: dai.InputQueue,
        temp_probe: Callable[[], float | None] | None = None
    ) -> SweepResult:
        """Sweep the lens around the anchor position and return the measured peak.

        Args:
            q_sync:     depthai output queue for the Sync node.
            q_camctrl:  depthai input queue for CameraControl messages.
            temp_probe: Optional callable returning the current chip temperature, sampled at
                        the sweep midpoint as the chip keeps warming while the sweep runs.

        Returns:
            SweepResult with all samples, the fitted peak and the guard outcomes.
        """
        # Centre on the position the anchor predicts for the current temperature
        center = _clamp(round(self._corrected_pos()))
        result = SweepResult(center_pos=center, span=FOCUS_SWEEP_SPAN)
        # Clamping at the hardware limits can map several steps onto one position, hence the set
        positions = sorted({
            _clamp(p)
            for p in range(center - FOCUS_SWEEP_SPAN,
                           center + FOCUS_SWEEP_SPAN + 1,
                           FOCUS_SWEEP_STEP)
        })
        sweep_start = time.monotonic()
        logger.info(
            "Focus sweep starting | center=%s (anchor %.2f) span=+/-%s positions=%s temp=%s",
            center, self.base_pos, FOCUS_SWEEP_SPAN, len(positions),
            f"{self.temp:.1f} °C" if self.temp is not None else "unknown"
        )

        # Let AE converge before locking, so the lock freezes a settled exposure
        self._wait_for_ae_settle(q_sync)
        self._set_ae_awb_lock(q_camctrl, True)
        t_origin: float | None = None

        def _measure(pos: int, bookend: bool = False) -> SweepSample | None:
            """Move to a position, wait out the settle frames and measure one frame.

            Args:
                pos:     Lens position to move to.
                bookend: Whether this measurement repeats the first position.

            Returns:
                SweepSample with the measured frame, or None if a frame timed out.
            """
            nonlocal t_origin
            self._set_lens_pos(q_camctrl, pos)
            for _ in range(FOCUS_SETTLE_FRAMES):
                if get_frame(q_sync) is None:
                    return None
            frame = get_frame(q_sync)
            if frame is None:
                return None
            now = time.monotonic()
            if t_origin is None:
                t_origin = now
            sample = SweepSample(
                commanded_pos=pos,
                actual_pos=frame.getLensPosition(),
                jpeg_bytes=int(frame.getData().size),
                iso=frame.getSensitivity(),
                exp_ms=round(frame.getExposureTime().total_seconds() * 1000, 2),
                t_rel=round(now - t_origin, 3),
                is_bookend=bookend
            )
            if sample.actual_pos != pos:
                logger.warning(
                    "Focus sweep: commanded lens position %s but frame reports %s "
                    "- increase FOCUS_SETTLE_FRAMES (currently %s)",
                    pos, sample.actual_pos, FOCUS_SETTLE_FRAMES
                )
            logger.debug(
                "Focus sweep sample%s | commanded=%s actual=%s bytes=%s iso=%s exp=%.2f ms t=%.2fs",
                " (bookend)" if bookend else "", pos, sample.actual_pos, sample.jpeg_bytes,
                sample.iso, sample.exp_ms, sample.t_rel
            )
            return sample

        midpoint = len(positions) // 2
        try:
            for index, pos in enumerate(positions):
                sample = _measure(pos)
                if sample is None:
                    result.reason = "frame timeout during measurement"
                    return self._finish_sweep(result, q_camctrl, sweep_start)
                result.samples.append(sample)
                if index == midpoint and temp_probe is not None:
                    result.temp_mid = temp_probe()

            # Bookend: repeat the first position to measure the drift over the sweep duration
            if len(result.samples) >= 2:
                bookend = _measure(positions[0], bookend=True)
                if bookend is not None:
                    result.samples.append(bookend)
        finally:
            # Always release the lock, or an aborted sweep leaves the exposure frozen for the session
            self._set_ae_awb_lock(q_camctrl, False)

        return self._finish_sweep(result, q_camctrl, sweep_start)

    def _finish_sweep(
        self,
        result: SweepResult,
        q_camctrl: dai.InputQueue,
        sweep_start: float
    ) -> SweepResult:
        """Remove the measured drift, evaluate the sweep guards, restore the lens position and log the outcome.

        Args:
            result:      Partially filled SweepResult to evaluate.
            q_camctrl:   depthai input queue for CameraControl messages.
            sweep_start: monotonic timestamp when the sweep started.

        Returns:
            The evaluated SweepResult.
        """
        # Remove the drift measured by the bookend before fitting: frame size ramps during a sweep,
        # and since positions are visited in order that ramp biases the peak in its direction
        measured = [s for s in result.samples if not s.is_bookend]
        bookend = next((s for s in result.samples if s.is_bookend), None)
        drift = 0.0
        drift_frac: float | None = None
        if bookend is not None and measured and bookend.t_rel > measured[0].t_rel:
            drift = ((bookend.jpeg_bytes - measured[0].jpeg_bytes)
                     / (bookend.t_rel - measured[0].t_rel))
            mean_size = statistics.fmean(s.jpeg_bytes for s in measured)
            if mean_size > 0:
                span_s = measured[-1].t_rel - measured[0].t_rel
                drift_frac = abs(drift) * span_s / mean_size
        if measured:
            t_start = measured[0].t_rel
            for s in result.samples:
                s.jpeg_bytes_detrended = round(s.jpeg_bytes - drift * (s.t_rel - t_start), 1)

        fit = _fit_peak(measured)

        # How much curve was resolved around the peak, which sets the required peak ratio
        # Only use a vertex that _fit_peak kept (a rejected one can sit outside the range and zero out both)
        peak = fit.vertex if fit.fit_ok else fit.argmax_pos
        if peak is not None and measured:
            swept = [s.commanded_pos for s in measured]
            resolved = max(0.0, min(peak - min(swept), max(swept) - peak))
        else:
            resolved = float(result.span)
        required = _required_peak_ratio(resolved)

        result.best_pos = fit.best_pos
        result.best_pos_fit = round(fit.vertex, 2) if fit.vertex is not None else None
        result.edge_hit = fit.edge_hit

        if fit.vertex is not None and not fit.fit_ok and fit.r2 >= FOCUS_MIN_FIT_R2:
            logger.info(
                "Focus sweep: parabola vertex outside the swept range (R2=%.3f), using the argmax "
                "(%s). Expected when the anchor sits off the true optimum.", fit.r2, fit.argmax_pos
            )
        elif (fit.best_pos is not None and fit.argmax_pos is not None
                and abs(fit.best_pos - fit.argmax_pos) > 1):
            logger.warning(
                "Focus sweep: fitted vertex %.2f differs from argmax %s by more than one code, "
                "using the fitted value. Expected on a flat or drifting curve top (%+.1f B/s).",
                fit.vertex if fit.vertex is not None else float("nan"), fit.argmax_pos, drift
            )

        if fit.best_pos is None:
            result.accepted = False
            result.reason = result.reason or "not enough samples"
        elif fit.vertex is None:
            # No downward parabola could be fitted, so the curve has no maximum to locate
            result.accepted = False
            result.reason = "no peak in the swept range (fitted curve has no maximum)"
        elif fit.r2 < FOCUS_MIN_FIT_R2:
            # On a featureless scene the parabola explains almost none of the variation, and
            # whichever sample happened to be noisiest would become a spurious peak
            result.accepted = False
            result.reason = "no sharpness curve in the swept range"
        elif drift_frac is not None and drift_frac > FOCUS_MAX_DRIFT_FRAC:
            # Drift this large usually means a step change in the scene, which de-trending
            # turns into a false tilt on the fitted parabola and moves the vertex
            result.accepted = False
            result.reason = "scene changed during the sweep"
        elif fit.peak_ratio < required:
            # A curve this flat has no peak that stands out from the frame-to-frame noise
            result.accepted = False
            result.reason = f"flat sharpness curve (+/-{resolved:.1f} positions resolved around the peak)"
        elif fit.edge_hit:
            # The guards above ask whether there is a curve, not whether its top was inside the
            # swept range: a ramp has the largest possible peak ratio and, being one side of a
            # parabola, a high R2. Its optimum lies further out than the anchor may move anyway
            result.accepted = False
            result.reason = (f"sharpness still rising at {fit.argmax_pos}, the edge of the swept "
                             f"range {measured[0].commanded_pos}-{measured[-1].commanded_pos}")
        else:
            result.accepted = True
            result.reason = "ok"

        # Restore the lens position to the measured peak if the sweep was accepted, else to where it was
        target = fit.best_pos if result.accepted and fit.best_pos is not None else result.center_pos
        self._set_lens_pos(q_camctrl, target)
        self.applied_pos = target

        log = logger.info if result.accepted else logger.warning
        log(
            "Focus sweep %s in %.1f s | center=%s -> lens_pos=%s (fit=%s argmax=%s) | temp=%s | "
            "R2=%.3f (min %.2f) peak=%.4f (min %.4f) drift=%.2f%% (max %.2f%%)%s",
            "accepted" if result.accepted else "rejected",
            time.monotonic() - sweep_start, result.center_pos, target,
            result.best_pos_fit, fit.argmax_pos,
            f"{result.temp_mid:.1f} °C" if result.temp_mid is not None else "unknown",
            fit.r2, FOCUS_MIN_FIT_R2, fit.peak_ratio, required,
            (drift_frac or 0.0) * 100, FOCUS_MAX_DRIFT_FRAC * 100,
            "" if result.accepted else f" | {result.reason}"
        )
        return result

    def set_up_session(
        self,
        q_sync: dai.MessageQueue,
        q_camctrl: dai.InputQueue,
        q_syslog: dai.MessageQueue
    ) -> None:
        """Anchor the focus for this recording session and apply the temperature correction.

        Runs the optional focus sweep, then moves the lens to the temperature-corrected position
        and waits for it to take effect. The correction matters most when the sweep is skipped:
        on a warm restart the anchor can be tens of degrees colder than the camera is now, which
        would otherwise leave the lens badly off until the first temperature check.

        Args:
            q_sync:    depthai output queue for the Sync node.
            q_camctrl: depthai input queue for CameraControl messages.
            q_syslog:  depthai output queue for SystemInformation messages.
        """
        self._record_thermal_state(get_chip_temp(q_syslog, wait_s=2.0))

        # Timing gates first, so a sweep that cannot run costs no exposure measurement
        allowed, reason = self.sweep_allowed()
        if allowed:
            # Gate on the converged exposure, not on whatever the next frame happens to report
            iso, exp_ms = self._wait_for_ae_settle(q_sync)
            allowed, reason = self.sweep_allowed(iso, exp_ms)

        if not allowed:
            logger.info("Focus sweep skipped: %s", reason)
        else:
            logger.info("Focus sweep allowed: %s", reason)
            result = self.run_sweep(
                q_sync, q_camctrl, temp_probe=lambda: get_chip_temp(q_syslog, self.temp)
            )
            if result.accepted and result.best_pos is not None:
                # temp_mid is the temperature the sweep actually measured at; a missed reading
                # falls back to a fresh one rather than discarding an otherwise good sweep
                anchor_temp = (result.temp_mid if result.temp_mid is not None
                               else get_chip_temp(q_syslog, self.temp, wait_s=2.0))
                anchor_pos = (result.best_pos_fit if result.best_pos_fit is not None
                              else float(result.best_pos))
                if anchor_temp is not None:
                    self.set_anchor(anchor_pos, anchor_temp, source="sweep")
                else:
                    logger.warning("Focus sweep accepted but no chip temperature available: "
                                   "keeping the previous anchor")

        # Seed the lens with the corrected position for the temperature the camera is at right now
        self.update_temp(get_chip_temp(q_syslog, self.temp, wait_s=2.0))
        target, clipped = self.target_position()
        if clipped:
            logger.warning("Focus start position clipped to the hardware lens position range")
        self._set_lens_pos(q_camctrl, target)
        wait_for_lens_pos(q_sync, target)
        self.applied_pos = target
        logger.info("Focus start position | lens_pos=%s temp=%s anchor=(%.2f, %s)", target,
                    f"{self.temp:.1f} °C" if self.temp is not None else "unknown",
                    self.base_pos,
                    f"{self.base_temp:.1f} °C" if self.base_temp is not None else "unset")

    def apply_correction(self, q_camctrl: dai.InputQueue, temp_now: float | None) -> None:
        """Move the lens to the corrected position if it drifted further than the deadband.

        Args:
            q_camctrl: depthai input queue for CameraControl messages.
            temp_now:  Current chip temperature in degrees Celsius, or None if unavailable.
        """
        self.update_temp(temp_now)
        target, clipped = self.target_position()
        if target == self.applied_pos:
            return

        self._set_lens_pos(q_camctrl, target)
        logger.info("Focus correction | temp=%.1f °C (anchor %.1f °C) | lens_pos=%s -> %s (%+d)",
                    temp_now if temp_now is not None else float("nan"),
                    self.base_temp if self.base_temp is not None else float("nan"),
                    self.applied_pos, target, target - self.applied_pos)
        self.applied_pos = target
        if clipped:
            logger.warning("Focus target clipped to the hardware lens position range")

    def persist(self, session_end: datetime | None = None) -> None:
        """Write the working anchor and session state to the manual focus file.

        Args:
            session_end: Session end timestamp used by the gap gate of the next session.
        """
        if session_end is not None:
            self.state.session_end = session_end.isoformat()
        save_manual_focus(self.state)
