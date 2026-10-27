"""Read standard NAVSIM caches and DrivoR's compact training caches.

Compact caches store safety-masked reference progress instead of the reference
trajectory, and polygon coordinates instead of shapely polygons. Conversion is
in memory; shared caches are never modified.
"""
import lzma
import pickle
import numpy as np
from shapely.geometry import Polygon
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import PDMResults
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.temporal_labels import aggregate_progress_scores
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex


class CompactMetricCache:
    pass


def restore_drivable_map(tokens, map_types, geometries, *args):
    polygons = [g if hasattr(g, 'geom_type') else Polygon(g) for g in geometries]
    return PDMDrivableMap(tokens, map_types, polygons, *args)


class TrainingCacheReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'navsim.planning.metric_caching.train_metric_chache' and name == 'MetricCache':
            return CompactMetricCache
        if module == PDMDrivableMap.__module__ and name == 'PDMDrivableMap':
            return restore_drivable_map
        return super().find_class(module, name)


def load_training_metric_cache(path):
    with lzma.open(path, 'rb') as stream:
        return TrainingCacheReader(stream).load()


class TrainingPDMScorer(PDMScorer):
    reference_progress = None

    def _aggregate_scores(self):
        if self.reference_progress is None:
            return super()._aggregate_scores()
        mask = self._multi_metrics.prod(axis=0)
        progress = self._progress_raw * mask
        # Equivalent to scoring each candidate against the cached PDM reference,
        # not normalizing against other candidates in the training batch.
        denominator = np.maximum(progress, self.reference_progress)
        normalized = np.ones_like(progress)
        nontrivial = denominator > self._config.progress_distance_threshold
        normalized[nontrivial] = progress[nontrivial] / denominator[nontrivial]
        normalized[(~nontrivial) & (mask == 0)] = 0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized
        weights = self._config.weighted_metrics_array
        return mask * (self._weighted_metrics * weights[:, None]).sum(0) / weights.sum()


def score_training_candidate(cache, trajectory, simulator, scorer):
    """Return PDM result, NC/DAC windows, and soft EP windows."""
    if hasattr(cache, 'trajectory'):
        scorer.reference_progress = None
        result = pdm_score(cache, trajectory, simulator.proposal_sampling, simulator, scorer)
        reference = getattr(cache, 'pdm_progress', scorer._progress_raw[0])
        reference = float(np.asarray(reference).reshape(-1)[0])
        ep = aggregate_progress_scores(scorer._progress_steps, reference,
                                       simulator.proposal_sampling.interval_length)
        return result, scorer.temporal_safety_scores()[1], ep
    progress = np.asarray(cache.pdm_progress)
    if progress.size != 1:
        raise ValueError(f'Expected scalar reference progress, got {progress.shape}; cannot infer horizon.')
    scorer.reference_progress = float(progress.reshape(-1)[0])
    if not np.isfinite(scorer.reference_progress) or scorer.reference_progress < 0:
        raise ValueError('Invalid cached reference progress.')
    initial = cache.ego_state
    states = get_trajectory_as_array(transform_trajectory(trajectory, initial),
                                    simulator.proposal_sampling, initial.time_point)
    simulated = simulator.simulate_proposals(states[None], initial)
    scores = scorer.score_proposals(simulated, cache.observation, cache.centerline,
                                    cache.route_lane_ids, cache.drivable_area_map)
    multi, weighted = scorer._multi_metrics, scorer._weighted_metrics
    result = PDMResults(multi[MultiMetricIndex.NO_COLLISION, 0], multi[MultiMetricIndex.DRIVABLE_AREA, 0],
                        weighted[WeightedMetricIndex.PROGRESS, 0], weighted[WeightedMetricIndex.TTC, 0],
                        weighted[WeightedMetricIndex.COMFORTABLE, 0], weighted[WeightedMetricIndex.DRIVING_DIRECTION, 0],
                        scores[0])
    ep = aggregate_progress_scores(
        scorer._progress_steps, scorer.reference_progress,
        simulator.proposal_sampling.interval_length)
    return result, scorer.temporal_safety_scores()[0], ep


def student_temporal_targets(result, window_scores, progress_steps, initial_safety,
                             interval_length=0.1):
    """Make segment labels consistent with official full-horizon scores.

    Include t=0 in the first safety window. EP labels are contributions to the
    official normalized EP, not independent fractions of reference distance.
    At standstill, distribute a nonzero full EP by window duration (a label
    convention, not physical motion). No simulator invocation occurs here.
    """
    from navsim.planning.simulation.planner.pdm_planner.scoring.temporal_labels import FUTURE_WINDOWS
    windows = np.asarray(window_scores, dtype=np.float64).copy()
    windows[0] = np.minimum(windows[0], initial_safety)
    progress = np.asarray(progress_steps, dtype=np.float64)
    increments = np.asarray([max(progress[int(round(end / interval_length))]
                                 - progress[int(round(start / interval_length))], 0.0)
                             for start, end in FUTURE_WINDOWS])
    if increments.sum() > 1e-6:
        fractions = increments / increments.sum()
    else:
        durations = np.asarray([end - start for start, end in FUTURE_WINDOWS])
        fractions = durations / durations.sum()
    ep = fractions * result.ego_progress
    full = np.asarray([result.no_at_fault_collisions, result.drivable_area_compliance,
                       result.ego_progress, result.time_to_collision_within_bound, result.comfort])
    if not np.allclose(windows.min(0), full[:2], atol=1e-6):
        raise ValueError('Temporal safety labels disagree with full simulator metrics.')
    expected_pdm = full[0] * full[1] * (5 * full[2] + 5 * full[3] + 2 * full[4]) / 12
    if not np.isclose(expected_pdm, result.score, atol=1e-6):
        raise ValueError('Student scorer requires NAVSIM-v1 PDM weights (EP=5,TTC=5,C=2,DDC=0).')
    return dict(window_scores=windows, ep_scores=ep, full_scores=full, pdm_score=result.score)


def score_student_candidate(cache, trajectory, simulator, scorer):
    """One simulation supplies all segment, full-metric and PDMS targets."""
    result, windows, _ = score_training_candidate(cache, trajectory, simulator, scorer)
    # Standard caches score reference + candidate; compact caches score only the
    # candidate. The candidate is the last proposal in both cases.
    return student_temporal_targets(
        result, windows, scorer._progress_steps[-1],
        np.asarray([scorer._nc_step_scores[-1, 0], scorer._dac_step_scores[-1, 0]]),
        simulator.proposal_sampling.interval_length)
