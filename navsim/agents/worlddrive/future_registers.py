"""Planning readout of frozen DiT futures; no latent reconstruction objective."""
import torch
from torch import nn
from torch.nn import functional as F


class FutureRegisterReadout(nn.Module):
    """Inputs: futures [N,3,16,H,W], motion [N,L,512]; output 36 tokens.

    Each group reads only its corresponding latent before temporal self-attention.
    The three heads refer to (0,.5], (.5,2.5], (2.5,4] seconds, not
    independent VAE receptive fields. Global output order is TTC, comfort,
    EP-0.5s, EP-2.5s, EP-4.0s.
    """

    def __init__(self, dim=512, registers_per_frame=12, layers=2, heads=8,
                 score_hidden_dim=1024, status_dim=8, loss_weights=None):
        super().__init__()
        self.dim = dim
        defaults = {
            'nc': 1.0, 'dac': 1.0, 'ttc': 1.0, 'comfort': 1.0,
            'ep': 1.0, 'ep_global': 0.25,
        }
        self.loss_weights = {**defaults, **(loss_weights or {})}
        if any(float(value) < 0 for value in self.loss_weights.values()):
            raise ValueError('Future-register loss weights must be non-negative.')
        self.registers_per_frame = registers_per_frame
        self.registers = nn.Parameter(torch.randn(3, registers_per_frame, dim) * .02)
        self.time_embedding = nn.Parameter(torch.randn(3, 1, dim) * .02)
        self.ego_token = nn.Parameter(torch.randn(1, dim) * .02)
        self.status_projection = nn.Linear(status_dim, dim)
        self.input_projection = nn.Linear(16, dim)
        self.spatial_projection = nn.Linear(4, dim, bias=False)
        self.readout = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True),
            num_layers=layers, norm=nn.LayerNorm(dim))
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True),
            num_layers=layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        # Match DrivoR's independent per-metric nonlinear readouts while
        # retaining our temporal-window NC/DAC and global TTC/C/EP layout.
        def score_head():
            return nn.Sequential(nn.Linear(dim, score_hidden_dim), nn.ReLU(), nn.Linear(score_hidden_dim, 1))
        self.window_heads = nn.ModuleList([score_head() for _ in range(2)])
        self.global_heads = nn.ModuleList([score_head() for _ in range(5)])

    def forward(self, latents, motion, ego_status=None, return_tokens_only=False):
        if latents.ndim != 5 or latents.shape[1:3] != (3, 16):
            raise ValueError('Expected [N,3,16,H,W] future latents (history removed).')
        n, _, _, h, w = latents.shape
        if motion.ndim != 3 or motion.shape[0] != n or motion.shape[1] < 1 or motion.shape[2] != 512:
            raise ValueError('Expected [N,L,512] motion tokens aligned with future candidates, L >= 1.')
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=latents.device, dtype=latents.dtype),
            torch.linspace(-1, 1, w, device=latents.device, dtype=latents.dtype), indexing='ij')
        position = torch.stack((x.sin(), x.cos(), y.sin(), y.cos()), -1).reshape(h*w, 4)
        memory = self.input_projection(latents.permute(0, 1, 3, 4, 2).reshape(n*3, h*w, 16))
        memory = memory + self.spatial_projection(position)[None]
        query = (self.registers + self.time_embedding)[None].expand(n, -1, -1, -1)
        local = self.readout(query.reshape(n*3, self.registers_per_frame, self.dim), memory)
        register_count = 3 * self.registers_per_frame
        scene_tokens = local.reshape(n, register_count, self.dim)
        # Motion is already 512-dimensional, so preserve its coordinates directly.
        motion_tokens = motion.detach().to(scene_tokens)
        if ego_status is None:
            ego_tokens = self.ego_token[None].expand(n, -1, -1).to(scene_tokens)
        else:
            if ego_status.ndim != 2 or ego_status.shape != (n, self.status_projection.in_features):
                raise ValueError(f'Expected ego status [{n},{self.status_projection.in_features}].')
            ego_tokens = self.ego_token[None] + self.status_projection(ego_status).unsqueeze(1)
            ego_tokens = ego_tokens.to(scene_tokens)
        fused = self.temporal(torch.cat((scene_tokens, ego_tokens, motion_tokens), dim=1))
        # The 37-token sequence is used by the scorer; only scene registers are
        # returned as the distillation target.
        scorer_tokens = fused[:, :register_count + 1]
        tokens = scorer_tokens[:, :register_count]
        if return_tokens_only:
            return {'tokens': tokens, 'scorer_tokens': scorer_tokens}
        grouped = tokens.reshape(n, 3, self.registers_per_frame, self.dim).mean(2)
        window_logits = torch.cat([head(grouped).squeeze(-1)[..., None]
                                   for head in self.window_heads], dim=-1)
        global_pooled = scorer_tokens.mean(1)
        global_logits = torch.cat([head(global_pooled).squeeze(-1)[..., None]
                                   for head in self.global_heads], dim=-1)
        return {'tokens': tokens, 'scorer_tokens': scorer_tokens,
                'window_logits': window_logits,
                'global_logits': global_logits, 'window_scores': window_logits.sigmoid(),
                'global_scores': global_logits.sigmoid()}

    def loss(self, predictions, targets):
        # NC can be 0.5 for collisions with non-agent objects: retain soft labels.
        window = targets['window_scores'].to(predictions['window_logits'])
        aggregate = targets['global_scores'].to(predictions['global_logits'])
        losses = {
            'register_nc_loss': self.loss_weights['nc'] * F.binary_cross_entropy_with_logits(predictions['window_logits'][..., 0], window[..., 0]),
            'register_dac_loss': self.loss_weights['dac'] * F.binary_cross_entropy_with_logits(predictions['window_logits'][..., 1], window[..., 1]),
            'register_ttc_loss': self.loss_weights['ttc'] * F.binary_cross_entropy_with_logits(predictions['global_logits'][..., 0], aggregate[..., 0]),
            'register_comfort_loss': self.loss_weights['comfort'] * F.binary_cross_entropy_with_logits(predictions['global_logits'][..., 1], aggregate[..., 1]),
            'register_ep_loss': self.loss_weights['ep'] * F.binary_cross_entropy_with_logits(
                predictions['global_logits'][..., 2:], aggregate[..., 2:]),
        }
        if 'global_ep' in targets:
            losses['register_ep_global_loss'] = self.loss_weights['ep_global'] * F.mse_loss(
                predictions['global_scores'][..., 2:].sum(-1).clamp(max=1.0),
                targets['global_ep'].to(predictions['global_logits']))
        losses['register_scorer_loss'] = sum(losses.values())
        losses['loss'] = losses['register_scorer_loss']
        return losses
