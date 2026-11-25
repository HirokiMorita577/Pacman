import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import colab_ghost4_retrain as retrain
import colab_ghost4_recovery as storage


class RetrainStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.local, self.drive = root / 'local', root / 'drive'
        self.run = self.local / 'runs' / 'job1'
        self.run.mkdir(parents=True)
        self.drive.mkdir()
        self.cp = self.run / 'training.pt.gz'
        self.cp.write_bytes(b'first checkpoint')
        for obj, name, value in [(storage, 'LOCAL', self.local), (storage, 'DRIVE', self.drive),
                                 (retrain, 'storage', storage)]:
            p = patch.object(obj, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)

    def test_first_and_final_checkpoint_saved_in_bounded_slots(self):
        with patch.object(storage, 'drive_free', return_value=10**10):
            self.assertTrue(retrain.sync_checkpoint(self.cp, 5, 100))
            self.assertFalse(retrain.sync_checkpoint(self.cp, 6, 100))
            self.cp.write_bytes(b'final checkpoint')
            self.assertTrue(retrain.sync_checkpoint(self.cp, 100, 100))
        slots = list(self.drive.rglob('recovery_checkpoint_*.pt.gz'))
        self.assertEqual(len(slots), 2)
        record = json.loads((self.run / 'recovery_sync.json').read_text())
        self.assertEqual(record['steps'], 100)
        self.assertEqual(record['slot'], 'B')

    def test_full_drive_preserves_last_verified_slot(self):
        with patch.object(storage, 'drive_free', return_value=10**10):
            retrain.sync_checkpoint(self.cp, 5, 100)
        old = (self.run / 'recovery_sync.json').read_bytes()
        self.cp.write_bytes(b'new checkpoint')
        with patch.object(storage, 'drive_free', return_value=1):
            self.assertFalse(retrain.sync_checkpoint(self.cp, 100, 100))
        self.assertEqual((self.run / 'recovery_sync.json').read_bytes(), old)
        self.assertEqual(next(self.drive.rglob('*.pt.gz')).read_bytes(), b'first checkpoint')

    def test_completed_result_restored_only_after_model_checksum_matches(self):
        remote = self.drive / 'runs' / 'job1'
        remote.mkdir(parents=True)
        manifest = {'job': 'fixed'}
        (remote / 'result.json').write_text(json.dumps({'manifest': manifest}))
        (remote / 'model.pt').write_bytes(b'model')
        (remote / 'saved_result.json').write_text(json.dumps({'sha256': {
            n: storage.digest(remote / n) for n in ['result.json', 'model.pt']}}))
        study = SimpleNamespace(job_id=lambda j: 'job1', experiment_manifest=lambda j: manifest)
        # No checkpoint to restore; test completion receipt independently.
        self.cp.unlink()
        (remote / 'model.pt').write_bytes(b'corrupt')
        retrain.restore_job({}, study)
        self.assertFalse((self.run / 'result.json').exists())
        (remote / 'model.pt').write_bytes(b'model')
        retrain.restore_job({}, study)
        self.assertEqual((self.run / 'model.pt').read_bytes(), b'model')


if __name__ == '__main__':
    unittest.main()
