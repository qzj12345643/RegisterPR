"""Pure distillation invariants, independent of full DiT allocation."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from navsim.agents.worlddrive.future_register_agent import RegisterDistillAgent
from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder, TrajWorldPlanner
from navsim.agents.worlddrive.future_registers import FutureRegisterReadout
from navsim.agents.worlddrive.worlddrive_refiner import FutureRegisterFAR


class DistillTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)

    def shell(self):
        agent = RegisterDistillAgent.__new__(RegisterDistillAgent)
        nn.Module.__init__(agent)
        agent.register_candidates = 16
        agent.validation_seed = 1234
        agent.student_far = nn.Linear(2, 2)
        return agent

    def test_sampling_rank_coverage_and_validation_repeatability(self):
        agent = self.shell()
        scores = torch.rand(4, 256)
        pool = scores.topk(16).indices
        seen = set()
        for _ in range(30):
            selected = agent._sample_candidate_indices(scores)
            torch.testing.assert_close(selected[:, :3], pool[:, :3])
            for row, allowed in zip(selected, pool):
                self.assertEqual(len(row.unique()), 6)
                self.assertTrue(set(row.tolist()) <= set(allowed.tolist()))
            seen.update(selected[0, 3:].tolist())
        self.assertEqual(seen, set(pool[0, 3:].tolist()))
        agent.eval()
        torch.testing.assert_close(agent._sample_candidate_indices(scores), agent._sample_candidate_indices(scores))

    def test_student_uses_generating_anchor_not_retrieval(self):
        agent = self.shell()
        agent.trajencoder_wm = TrajEncoder(traj_vocab=torch.randn(256, 8, 3)).eval()
        indices = torch.tensor([[12, 4, 9], [32, 17, 5]])
        offsets = torch.randn(2, 3, 24)
        encoder = agent.trajencoder_wm
        with patch.object(encoder, 'forward', side_effect=AssertionError('Must not retrieve anchors')):
            result = agent._student_motion(indices, offsets)
        expected = encoder.traj_vocab_encoder(encoder.norm_odo(encoder.traj_vocab[indices]).reshape(2, 3, 24))
        expected = expected + encoder.traj_offset_encoder(offsets)
        torch.testing.assert_close(result, expected.reshape(6, 1, 512))

    def test_train_mode_keeps_frozen_modules_in_eval(self):
        agent = self.shell()
        agent.teacher_registers = nn.Sequential(nn.Dropout(.5), nn.Linear(2, 2))
        agent.trajplanner = nn.Sequential(nn.Dropout(.5))
        agent.requires_grad_(False)
        agent.student_far.requires_grad_(True)
        agent.train()
        self.assertTrue(agent.training and agent.student_far.training)
        self.assertFalse(agent.teacher_registers.training or agent.trajplanner.training)
        self.assertTrue(all(name.startswith('student_far.') for name, p in agent.named_parameters() if p.requires_grad))
        agent.eval()
        self.assertFalse(any(module.training for module in agent.modules()))

    def test_token_only_teacher_is_identical_and_skips_heads(self):
        model = FutureRegisterReadout(layers=1).eval()
        latent, motion = torch.randn(2, 3, 16, 2, 3), torch.randn(2, 5, 512)
        with torch.no_grad():
            expected = model(latent, motion)['tokens']
            with patch.object(model.window_heads[0], 'forward', side_effect=AssertionError('Scoring called')):
                actual = model(latent, motion, return_tokens_only=True)
        torch.testing.assert_close(actual['tokens'], expected)
        self.assertEqual(set(actual), {'tokens', 'scorer_tokens'})

    def test_existing_student_weights_accept_one_motion_token(self):
        old = FutureRegisterFAR(layers=1)
        new = FutureRegisterFAR(layers=1)
        new.load_state_dict(old.state_dict(), strict=True)
        result = new(torch.randn(2, 9, 512), torch.randn(2, 1, 512), torch.randn(2, 8))
        self.assertEqual(result.shape, (2, 36, 512))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in new.parameters()))

    def test_batched_candidates_match_original_inference_without_targets(self):
        planner = TrajWorldPlanner(num_dec_layer=1).eval()
        visual, status = torch.randn(2, 2, 16, 4, 6), torch.randn(2, 8)
        vocab, embed = torch.randn(256, 8, 3), torch.randn(2, 256, 512)
        with torch.no_grad(), patch.object(planner, '_get_traj_gt_v3', side_effect=AssertionError('GT loss called')):
            scores, offsets, _ = planner(visual, status, embed, vocab,
                                        eval_mode=True, return_candidates_only=True)
            for i in range(2):
                _, old_scores, old_offsets, _ = planner(visual[i:i+1], status[i:i+1], embed[i:i+1], vocab, eval_mode=True)
                torch.testing.assert_close(scores[i:i+1], old_scores, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(offsets[i:i+1], old_offsets, atol=2e-5, rtol=2e-5)

    def test_first_decoder_is_action_conditioned_and_candidate_isolated(self):
        model = FutureRegisterFAR(layers=1).eval()
        visual = torch.randn(1, 9, 512).expand(2, -1, -1)
        motion = torch.randn(1, 1, 512).expand(2, -1, -1).clone().requires_grad_()
        local_outputs = []
        handle = model.decoder.register_forward_hook(
            lambda module, inputs, output: local_outputs.append(output))
        try:
            baseline = model(visual, motion)
            # Initially both candidates have identical visual and motion inputs.
            torch.testing.assert_close(local_outputs[0][0], local_outputs[0][1])
            changed_motion = motion.detach().clone()
            changed_motion[0] = torch.randn(1, 512) * 3
            changed = model(visual, changed_motion)
            # The action must affect the FIRST decoder, not only late fusion.
            self.assertGreater((local_outputs[1][0] - local_outputs[0][0]).abs().max().item(), 1e-5)
            torch.testing.assert_close(local_outputs[1][1], local_outputs[0][1])
            torch.testing.assert_close(changed[1], baseline[1])
            baseline.square().mean().backward()
            self.assertIsNone(motion.grad)  # frozen motion encoder boundary
            self.assertIsNotNone(model.decoder.layers[0].multihead_attn.in_proj_weight.grad)
        finally:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
