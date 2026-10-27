import unittest
import numpy as np
import torch
from navsim.agents.worlddrive.future_registers import FutureRegisterReadout
from navsim.agents.worlddrive.worlddrive_refiner import FutureRegisterFAR
from navsim.planning.simulation.planner.pdm_planner.scoring.temporal_labels import aggregate_window_scores, aggregate_progress_scores


class TemporalLabelsTest(unittest.TestCase):
    def test_boundaries_local_not_cumulative_and_soft_collision(self):
        nc, dac = np.ones((2, 41)), np.ones((2, 41))
        nc[0, 0] = 0  # t=0 is deliberately excluded
        nc[0, 5] = .5  # .5 belongs to first interval
        nc[0, 25] = 0  # 2.5 belongs to second interval
        dac[0, 26] = 0  # 2.6 belongs to third interval
        nc[1, 40] = 0  # endpoint 4.0 included
        result = aggregate_window_scores(nc, dac, .1)
        np.testing.assert_array_equal(result[0], [[.5, 1], [0, 1], [1, 0]])
        np.testing.assert_array_equal(result[1], [[1, 1], [1, 1], [0, 1]])

    def test_reject_truncated_horizon(self):
        with self.assertRaises(ValueError):
            aggregate_window_scores(np.ones((1, 20)), np.ones((1, 20)), .1)

    def test_progress_soft_labels_are_windowed(self):
        progress = np.arange(41, dtype=np.float64)[None, :] * 0.1
        result = aggregate_progress_scores(progress, 4.0, .1)
        np.testing.assert_allclose(result, [[.125, .5, .375]])


class ReadoutTest(unittest.TestCase):
    def test_gradients_shapes_and_candidate_isolation(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        model = FutureRegisterReadout(layers=1)
        x = torch.randn(2, 3, 16, 3, 4)
        motion = torch.randn(2, 5, 512, requires_grad=True)
        result = model(x, motion)
        self.assertEqual(result['tokens'].shape, (2, 36, 512))
        self.assertEqual(result['window_scores'].shape, (2, 3, 2))
        self.assertEqual(result['global_scores'].shape, (2, 5))
        # Candidate processing must not depend on other candidates or chunk size.
        torch.testing.assert_close(result['tokens'][:1], model(x[:1], motion[:1])['tokens'], atol=2e-6, rtol=2e-5)
        losses = model.loss(result, {'window_scores': torch.rand(2, 3, 2),
                                    'global_scores': torch.rand(2, 5),
                                    'global_ep': torch.rand(2)})
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertTrue((model.registers.grad.abs().sum((1, 2)) > 0).all())
        self.assertIsNone(x.grad)
        self.assertIsNone(motion.grad)
        self.assertFalse(hasattr(model, 'motion_projection'))

    def test_motion_changes_only_corresponding_candidate_scores(self):
        torch.manual_seed(19)
        model = FutureRegisterReadout(layers=1).eval()
        x = torch.randn(2, 3, 16, 2, 3)
        motion = torch.randn(2, 5, 512)
        changed = motion.clone()
        changed[0] += 30
        with torch.no_grad():
            a, b = model(x, motion), model(x, changed)
        for key in ['window_logits', 'global_logits']:
            torch.testing.assert_close(a[key][1], b[key][1])
            self.assertGreater((a[key][0] - b[key][0]).abs().max().item(), 1e-8)

    def test_each_group_reads_only_its_latent_before_temporal_fusion(self):
        model = FutureRegisterReadout(layers=1).eval()
        model.temporal = torch.nn.Identity()
        x = torch.randn(1, 3, 16, 2, 3)
        y = x.clone()
        y[:, 1] += 3
        motion = torch.randn(1, 5, 512)
        with torch.no_grad():
            a, b = model(x, motion)['tokens'], model(y, motion)['tokens']
        torch.testing.assert_close(a[:, :12], b[:, :12])
        torch.testing.assert_close(a[:, 24:], b[:, 24:])
        self.assertGreater((a[:, 12:24]-b[:, 12:24]).abs().max().item(), .001)

    def test_reject_history_included(self):
        with self.assertRaises(ValueError):
            FutureRegisterReadout(layers=1)(torch.randn(1, 5, 16, 2, 3), torch.randn(1, 5, 512))

    def test_reject_misaligned_motion(self):
        model = FutureRegisterReadout(layers=1)
        x = torch.randn(2, 3, 16, 2, 3)
        for shape in [(1, 5, 512), (2, 5, 256), (2, 0, 512), (2, 512)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(x, torch.randn(*shape))

    def test_student_far_returns_36_512_registers(self):
        torch.manual_seed(3)
        model = FutureRegisterFAR(layers=1)
        visual = torch.randn(2, 10, 512)
        motion = torch.randn(2, 5, 512)
        result = model(visual, motion)
        self.assertEqual(result.shape, (2, 36, 512))
        result.square().mean().backward()
        self.assertIsNotNone(model.query.grad)
        self.assertIsNotNone(model.time_embedding.grad)


if __name__ == '__main__':
    unittest.main()
