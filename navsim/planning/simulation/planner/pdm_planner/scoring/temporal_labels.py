"""Aggregate simulator event scores, not cumulative/final trajectory scores."""
import numpy as np

FUTURE_WINDOWS = ((0.0, 0.5), (0.5, 2.5), (2.5, 4.0))


def aggregate_window_scores(nc_steps, dac_steps, interval_length, windows=FUTURE_WINDOWS):
    """Return [proposals, windows, NC/DAC], excluding t=0 by definition.

    Scores at each simulation step are local, so an early collision does not
    automatically label every later window as a failure. Repeated at-fault
    contact can still produce events in several windows.
    """
    nc_steps, dac_steps = np.asarray(nc_steps), np.asarray(dac_steps)
    if nc_steps.shape != dac_steps.shape or nc_steps.ndim != 2 or interval_length <= 0:
        raise ValueError('Expected matching [proposals, steps] arrays and positive step size.')
    times = np.arange(nc_steps.shape[1]) * interval_length
    result = []
    for start, end in windows:
        if end <= start or end > times[-1] + 1e-8:
            raise ValueError('Window must lie within the simulated horizon.')
        mask = (times > start + 1e-8) & (times <= end + 1e-8)
        if not mask.any():
            raise ValueError('Window contains no simulator steps.')
        result.append(np.stack((nc_steps[:, mask].min(1), dac_steps[:, mask].min(1)), -1))
    return np.stack(result, axis=1)


def aggregate_progress_scores(progress_steps, reference_progress, interval_length,
                              windows=FUTURE_WINDOWS):
    """Return soft progress labels [proposals, windows].

    Each label is the progress made inside a time window, normalized by the
    cached reference progress.  This mirrors DrivoR's soft progress target
    while preserving the three temporal windows used by NC/DAC.
    """
    progress_steps = np.asarray(progress_steps, dtype=np.float64)
    reference_progress = np.asarray(reference_progress, dtype=np.float64)
    if progress_steps.ndim != 2 or interval_length <= 0:
        raise ValueError('Expected [proposals, steps] progress and positive step size.')
    if reference_progress.ndim == 0:
        reference_progress = np.full(progress_steps.shape[0], float(reference_progress))
    if reference_progress.shape != (progress_steps.shape[0],):
        raise ValueError('Reference progress must be scalar or [proposals].')
    times = np.arange(progress_steps.shape[1]) * interval_length
    result = []
    denominator = np.maximum(reference_progress, 1e-6)
    for start, end in windows:
        if end <= start or end > times[-1] + 1e-8:
            raise ValueError('Window must lie within the simulated horizon.')
        start_idx = int(round(start / interval_length))
        end_idx = int(round(end / interval_length))
        if end_idx >= progress_steps.shape[1] or end_idx <= start_idx:
            raise ValueError('Window contains no simulator steps.')
        delta = np.maximum(progress_steps[:, end_idx] - progress_steps[:, start_idx], 0.0)
        result.append(np.clip(delta / denominator, 0.0, 1.0))
    return np.stack(result, axis=1)
