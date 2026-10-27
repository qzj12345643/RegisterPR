"""Temporal scoring heads and differentiable NAVSIM-v1 aggregation."""
import torch
from torch import nn
from torch.nn import functional as F


class StudentRegisterScorer(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, segment_loss_weight=1.0,
                 global_loss_weight=1.0, aggregate_loss_weight=1.0, pdm_loss_weight=1.0):
        super().__init__()
        self.loss_weights = dict(segment=segment_loss_weight, global_metrics=global_loss_weight,
                                 aggregate=aggregate_loss_weight, pdm=pdm_loss_weight)
        if any(v < 0 for v in self.loss_weights.values()) or not any(self.loss_weights.values()):
            raise ValueError('Loss weights must be nonnegative with at least one active objective.')
        def head():
            return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        # Names and shapes intentionally match the trained teacher's score heads.
        self.window_heads = nn.ModuleList([head() for _ in range(2)])
        self.global_heads = nn.ModuleList([head() for _ in range(5)])

    @staticmethod
    def aggregate(window_scores, global_scores):
        # Full metrics ordered NC, DAC, EP, TTC, Comfort.
        nc, dac = window_scores.float().amin(dim=1).unbind(-1)
        ttc, comfort = global_scores.float()[:, :2].unbind(-1)
        ep = global_scores.float()[:, 2:].sum(-1).clamp(0, 1)
        full = torch.stack((nc, dac, ep, ttc, comfort), -1)
        pdm = nc * dac * (5 * ep + 5 * ttc + 2 * comfort) / 12
        return full, pdm

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (37, 512):
            raise ValueError('Expected 36 scene registers and one fused ego token: [N,37,512].')
        grouped = tokens[:, :36].reshape(-1, 3, 12, 512).mean(2)
        window_logits = torch.cat([head(grouped) for head in self.window_heads], -1)
        pooled = tokens.mean(1)
        global_logits = torch.cat([head(pooled) for head in self.global_heads], -1)
        window_scores, global_scores = window_logits.sigmoid(), global_logits.sigmoid()
        full_scores, pdm_score = self.aggregate(window_scores, global_scores)
        return dict(window_logits=window_logits, global_logits=global_logits,
                    window_scores=window_scores, global_scores=global_scores,
                    full_scores=full_scores, pdm_score=pdm_score)

    def loss(self, prediction, targets):
        # Compute objectives in FP32 under mixed precision. Soft NC/EP labels
        # remain soft; aggregate MSE is stable through min/product/clamp.
        wl, gl = prediction['window_logits'].float(), prediction['global_logits'].float()
        windows = targets['window_scores'].to(wl)
        ep = targets['ep_scores'].to(gl)
        full = targets['full_scores'].to(gl)
        nc_loss = F.binary_cross_entropy_with_logits(wl[..., 0], windows[..., 0])
        dac_loss = F.binary_cross_entropy_with_logits(wl[..., 1], windows[..., 1])
        ep_loss = F.binary_cross_entropy_with_logits(gl[:, 2:], ep)
        ttc_loss = F.binary_cross_entropy_with_logits(gl[:, 0], full[:, 3])
        comfort_loss = F.binary_cross_entropy_with_logits(gl[:, 1], full[:, 4])
        segment = (nc_loss + dac_loss + ep_loss) / 3
        global_loss = (ttc_loss + comfort_loss) / 2
        aggregate = F.mse_loss(prediction['full_scores'][:, :3], full[:, :3])
        pdm = F.mse_loss(prediction['pdm_score'], targets['pdm_score'].to(gl))
        loss = (self.loss_weights['segment'] * segment + self.loss_weights['global_metrics'] * global_loss
                + self.loss_weights['aggregate'] * aggregate + self.loss_weights['pdm'] * pdm)
        return dict(loss=loss, scorer_segment_loss=segment, scorer_global_loss=global_loss,
                    scorer_aggregate_loss=aggregate, scorer_pdm_loss=pdm,
                    scorer_nc_loss=nc_loss, scorer_dac_loss=dac_loss, scorer_ep_loss=ep_loss,
                    scorer_ttc_loss=ttc_loss, scorer_comfort_loss=comfort_loss)
