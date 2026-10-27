"""Run in a NAVSIM/nuPlan environment; geometry classification is controlled."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_scorer as module


class Observation:
    collided_track_ids = []
    red_light_token = 'red_light'
    unique_objects = {
        'agent': SimpleNamespace(tracked_object_type=next(iter(module.AGENT_TYPES))),
        'object': SimpleNamespace(tracked_object_type=object()),
    }

    def __getitem__(self, step):
        token = 'object' if step == 5 else 'agent'
        return SimpleNamespace(
            tokens=[token],
            query=lambda *args, **kwargs: np.array([[0], [0]]) if step in (5, 25) else np.empty((2, 0), dtype=int),
            # get_collision_type is mocked, but Python evaluates lookup arguments.
        ) if step not in (5, 25) else CollisionFrame(token)


class CollisionFrame:
    def __init__(self, token):
        self.tokens = [token]

    def query(self, *args, **kwargs):
        return np.array([[0], [0]])

    def __getitem__(self, token):
        return None


class ScorerTimelineTest(unittest.TestCase):
    def test_records_local_events_without_changing_scalar_metrics(self):
        scorer = module.PDMScorer.__new__(module.PDMScorer)
        scorer._num_proposals = 1
        scorer.proposal_sampling = SimpleNamespace(num_poses=40, interval_length=.1)
        scorer._observation = Observation()
        scorer._states = np.zeros((1, 41, 11))
        scorer._ego_polygons = np.empty((1, 41), dtype=object)
        scorer._ego_areas = np.zeros((1, 41, len(module.EgoAreaIndex)), dtype=bool)
        scorer._ego_areas[0, 26, module.EgoAreaIndex.NON_DRIVABLE_AREA] = True
        scorer._multi_metrics = np.ones((len(module.MultiMetricIndex), 1))
        scorer._collision_time_idcs = np.full(1, np.inf)
        with patch.object(module, 'get_collision_type', return_value=module.CollisionType.ACTIVE_FRONT_COLLISION):
            scorer._calculate_no_at_fault_collision()
        scorer._calculate_drivable_area_compliance()
        np.testing.assert_array_equal(scorer.temporal_safety_scores()[0], [[.5, 1], [0, 1], [1, 0]])
        self.assertEqual(scorer._multi_metrics[module.MultiMetricIndex.NO_COLLISION, 0], 0)
        self.assertEqual(scorer._multi_metrics[module.MultiMetricIndex.DRIVABLE_AREA, 0], 0)
        self.assertEqual(scorer._collision_time_idcs[0], 5)


if __name__ == '__main__':
    unittest.main()
