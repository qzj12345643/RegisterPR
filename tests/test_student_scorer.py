import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from navsim.agents.worlddrive.student_register_scorer import StudentRegisterScorer
from navsim.agents.worlddrive.student_scorer_agent import StudentRegisterScorerAgent
from navsim.agents.worlddrive.training_metric_cache import student_temporal_targets, score_student_candidate
from navsim.agents.worlddrive.future_registers import FutureRegisterReadout
from navsim.agents.worlddrive.worlddrive_refiner import FutureRegisterFAR


class StudentScorerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)

    def test_aggregation_and_gradients(self):
        windows = torch.tensor([[[.9,.95],[.8,.9],[.85,.92]]],requires_grad=True)
        global_scores = torch.tensor([[.7,.8,.1,.2,.3]],requires_grad=True)
        full,pdm = StudentRegisterScorer.aggregate(windows, global_scores)
        torch.testing.assert_close(full,torch.tensor([[.8,.9,.6,.7,.8]]))
        torch.testing.assert_close(pdm,torch.tensor([.8*.9*(5*.6+5*.7+2*.8)/12]))
        pdm.sum().backward()
        self.assertGreater(windows.grad.abs().sum().item(),0)
        self.assertTrue((global_scores.grad != 0).all())

    def test_heads_warmstart_and_full_loss_backprop(self):
        teacher=FutureRegisterReadout(layers=1)
        scorer=StudentRegisterScorer()
        scorer.load_state_dict({k:v for k,v in teacher.state_dict().items()
                               if k.startswith(('window_heads.','global_heads.'))},strict=True)
        tokens=torch.randn(2,37,512,requires_grad=True)
        out=scorer(tokens)
        loss=scorer.loss(out,dict(window_scores=torch.rand(2,3,2),ep_scores=torch.rand(2,3),
                                  full_scores=torch.rand(2,5),pdm_score=torch.rand(2)))
        expected=sum(loss[k] for k in ('scorer_segment_loss','scorer_global_loss','scorer_aggregate_loss','scorer_pdm_loss'))
        torch.testing.assert_close(loss['loss'],expected)
        loss['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scorer.parameters()))
        self.assertGreater(tokens.grad[:,36].abs().sum().item(),0)

    def test_37_token_interface_preserves_distillation_output(self):
        far=FutureRegisterFAR(layers=1).eval()
        visual,motion,status=torch.randn(2,5,512),torch.randn(2,1,512),torch.randn(2,8)
        a=far(visual,motion,status)
        b=far(visual,motion,status,return_scorer_tokens=True)
        self.assertEqual(b.shape,(2,37,512))
        torch.testing.assert_close(a,b[:,:36])

    def test_labels_handle_initial_safety_standstill_and_ep_saturation(self):
        for nc,dac,ep,progress in [(1,1,1,np.zeros(41)),(.5,1,.6,np.arange(41)),
                                  (1,0,0,np.arange(41)),(1,1,1,np.arange(41)*10)]:
            result=SimpleNamespace(no_at_fault_collisions=nc,drivable_area_compliance=dac,
                                   ego_progress=ep,time_to_collision_within_bound=.9,comfort=1,
                                   score=nc*dac*(5*ep+5*.9+2)/12)
            labels=student_temporal_targets(result,np.ones((3,2)),progress,[nc,dac])
            self.assertAlmostEqual(labels['ep_scores'].sum(),ep)
            np.testing.assert_allclose(labels['window_scores'].min(0),[nc,dac])
            full,pdm=StudentRegisterScorer.aggregate(torch.tensor(labels['window_scores'])[None],
                         torch.tensor([[.9,1,*labels['ep_scores']]]))
            np.testing.assert_allclose(full.numpy()[0],labels['full_scores'],rtol=1e-6)
            self.assertAlmostEqual(pdm.item(),result.score,places=6)

    def test_one_simulator_result_reused(self):
        r=SimpleNamespace(no_at_fault_collisions=1,drivable_area_compliance=1,ego_progress=.5,
                          time_to_collision_within_bound=1,comfort=1,score=9.5/12)
        scorer=SimpleNamespace(_nc_step_scores=np.ones((1,41)),_dac_step_scores=np.ones((1,41)),
                               _progress_steps=np.arange(41)[None])
        simulator=SimpleNamespace(proposal_sampling=SimpleNamespace(interval_length=.1))
        with patch('navsim.agents.worlddrive.training_metric_cache.score_training_candidate',
                   return_value=(r,np.ones((3,2)),None)) as score:
            labels=score_student_candidate(None,None,simulator,scorer)
        score.assert_called_once()
        self.assertEqual(set(labels),{'window_scores','ep_scores','full_scores','pdm_score'})

    def test_phase_schedule_and_frozen_base(self):
        agent=StudentRegisterScorerAgent.__new__(StudentRegisterScorerAgent)
        nn.Module.__init__(agent)
        agent.student_far=nn.Linear(2,2)
        agent.student_scorer=nn.Linear(2,2)
        agent.trajplanner=nn.Sequential(nn.Dropout(.5),nn.Linear(2,2)).requires_grad_(False)
        agent.scorer_warmup_epochs=1;agent.training_epochs=10;agent.far_lr_scale=.1;agent._lr=5e-5
        agent.set_training_epoch(0);agent.train()
        self.assertFalse(agent.student_far.training or agent.trajplanner.training)
        opt=agent.get_coslr_optimizers();optimizer=opt['optimizer'];scheduler=opt['lr_scheduler']
        self.assertEqual(optimizer.param_groups[1]['lr'],0)
        optimizer.step();scheduler.step();agent.set_training_epoch(1);agent.train()
        self.assertTrue(agent.student_far.training)
        self.assertFalse(agent.trajplanner.training)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr']/optimizer.param_groups[0]['lr'],.1)
        state=scheduler.state_dict()
        restored=agent.get_coslr_optimizers()['lr_scheduler'];restored.load_state_dict(state)
        self.assertEqual(restored.last_epoch,1)


if __name__=='__main__':unittest.main()
