import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.cache_register_features import atomic_cache
from navsim.planning.training.dataset import load_feature_target_from_pickle


class CacheAtomicWriteTest(unittest.TestCase):
    def test_rename_committed_but_reply_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'feature.gz'
            value = {'test': torch.arange(4)}
            original = os.replace

            def lost_reply(source, target):
                original(source, target)
                raise FileNotFoundError('simulated lost rename reply')

            with patch('scripts.cache_register_features.os.replace', side_effect=lost_reply) as rename:
                atomic_cache(destination, value)
            self.assertEqual(rename.call_count, 1)
            self.assertTrue(torch.equal(load_feature_target_from_pickle(destination)['test'], value['test']))

    def test_missing_temporary_file_retries_and_rejects_stale_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'feature.gz'
            atomic_cache(destination, {'test': torch.zeros(4)})
            value = {'test': torch.arange(4)}
            original = os.replace
            calls = []

            def transient_missing(source, target):
                calls.append(source)
                if len(calls) == 1:
                    Path(source).unlink()
                    raise FileNotFoundError('simulated missing temporary file')
                original(source, target)

            with patch('scripts.cache_register_features.os.replace', side_effect=transient_missing), patch('scripts.cache_register_features.time.sleep'):
                atomic_cache(destination, value)
            self.assertEqual(len(calls), 2)
            self.assertTrue(torch.equal(load_feature_target_from_pickle(destination)['test'], value['test']))


if __name__ == '__main__':
    unittest.main()
