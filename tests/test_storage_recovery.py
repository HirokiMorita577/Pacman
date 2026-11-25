"""Regression checks for quota exhaustion and bounded checkpoint uploads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('storage_recovery', Path(__file__).parents[1] / 'colab_ghost4_recovery.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class StorageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.local = root / 'local'
        self.drive = root / 'drive'
        self.run = self.local / 'runs' / 'seed101'
        self.run.mkdir(parents=True)
        self.drive.mkdir()
        self.cp = self.run / 'training.pt.gz'
        self.cp.write_bytes(b'checkpoint1')
        for name, value in [('LOCAL', self.local), ('DRIVE', self.drive)]:
            p = patch.object(recovery, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_full_drive_does_not_truncate_checkpoint(self):
        target = self.drive / 'saved.pt.gz'
        target.write_bytes(b'old verified checkpoint')
        with patch.object(recovery, 'drive_free', return_value=40 * 1024 * 1024):
            self.assertFalse(recovery.upload(self.cp, target))
        self.assertEqual(target.read_bytes(), b'old verified checkpoint')
        self.assertEqual(self.cp.read_bytes(), b'checkpoint1')

    def test_repeated_updates_use_only_two_slots_without_renames(self):
        with patch.object(recovery, 'drive_free', return_value=10**10), patch.object(recovery.os, 'replace', side_effect=AssertionError('Drive rename')):
            for steps in [30000, 60000, 90000, 120000]:
                self.cp.write_bytes(str(steps).encode())
                self.assertTrue(recovery.sync_checkpoint(self.cp, steps))
        slots = list(self.drive.rglob('recovery_checkpoint_*.pt.gz'))
        self.assertEqual(len(slots), 2)
        state = json.loads((self.run / 'recovery_sync.json').read_text())
        self.assertEqual(state['steps'], 120000)
        self.assertEqual(recovery.digest(self.drive / 'runs' / 'seed101' / state['file']), state['sha256'])

    def test_failed_next_upload_preserves_previous_verified_slot(self):
        with patch.object(recovery, 'drive_free', return_value=10**10):
            self.assertTrue(recovery.sync_checkpoint(self.cp, 30000))
        state = (self.run / 'recovery_sync.json').read_bytes()
        previous = self.drive / 'runs' / 'seed101' / 'recovery_checkpoint_A.pt.gz'
        self.cp.write_bytes(b'new checkpoint')
        with patch.object(recovery, 'upload', return_value=False):
            self.assertFalse(recovery.sync_checkpoint(self.cp, 60000))
        self.assertEqual((self.run / 'recovery_sync.json').read_bytes(), state)
        self.assertEqual(previous.read_bytes(), b'checkpoint1')


if __name__ == '__main__':
    unittest.main()
