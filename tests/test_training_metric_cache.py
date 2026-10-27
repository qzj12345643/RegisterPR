import unittest
from types import SimpleNamespace
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.worlddrive.training_metric_cache import TrainingPDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex


class CompactCacheTest(unittest.TestCase):
    def test_progress_normalization_matches_reference_plus_candidate(self):
        sampling = TrajectorySampling(time_horizon=4, interval_length=.1)
        for reference, candidate, safety in [(16.7, 10, 1), (10, 20, 1), (0, 0, 1), (16.7, 10, 0), (16.7, 10, .5)]:
            standard = PDMScorer(sampling)
            compact = TrainingPDMScorer(sampling)
            for scorer, count in [(standard, 2), (compact, 1)]:
                scorer._multi_metrics = np.ones((len(MultiMetricIndex), count))
                scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, -1] = safety
                scorer._weighted_metrics = np.ones((len(WeightedMetricIndex), count))
            standard._progress_raw = np.array([reference, candidate])
            compact._progress_raw = np.array([candidate])
            compact.reference_progress = reference
            np.testing.assert_allclose(standard._aggregate_scores()[-1], compact._aggregate_scores()[0])
            np.testing.assert_allclose(standard._weighted_metrics[WeightedMetricIndex.PROGRESS, -1],
                                       compact._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])

    def test_targets_without_static_anchor_reward_cache(self):
        from navsim.agents.worlddrive.worlddrive_features import TrajectoryTargetBuilder
        target = np.zeros((8, 3), dtype=np.float32)
        scene = SimpleNamespace(get_future_trajectory=lambda **kwargs: SimpleNamespace(poses=target))
        result = TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=.5)).compute_targets(scene)
        np.testing.assert_array_equal(result['trajectory'].numpy(), target)


if __name__ == '__main__':
    unittest.main()
