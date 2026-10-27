import tempfile
import unittest
from pathlib import Path

import torch
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.worlddrive.blocks.cogvideox import CogVideoXPatchEmbed


class TeacherRuntimeCompatibilityTest(unittest.TestCase):
    def test_named_metric_cache_and_legacy_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'train_metric_cache_ego_area_v2'
            (root / 'metadata').mkdir(parents=True)
            index = root / 'metadata/index.csv'
            actual = root / 'log/unknown/token/metric_cache.pkl'
            index.write_text(f'file_name\n{actual}\n')
            self.assertEqual(MetricCacheLoader(root).metric_cache_paths['token'], str(actual))
            index.write_text('file_name\n/old/metric_cache/log/unknown/token/metric_cache.pkl\n')
            self.assertEqual(MetricCacheLoader(root).metric_cache_paths['token'], str(actual))

    def test_torch_position_embedding_constructor_and_forward(self):
        module = CogVideoXPatchEmbed(patch_size=1, in_channels=4, embed_dim=32,
                                    cond_embed_dim=8, sample_width=4, sample_height=2,
                                    sample_frames=5, max_text_seq_length=3,
                                    use_learned_positional_embeddings=False)
        result = module(torch.zeros(1, 3, 8), torch.zeros(1, 2, 4, 2, 4))
        self.assertEqual(result.shape, (1, 19, 32))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
