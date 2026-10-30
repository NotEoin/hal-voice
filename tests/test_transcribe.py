"""Tests for hal-transcribe that need no model."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TranscribeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HAL_DATA"] = self.tmp.name
        from haltranscribe import tools

        self.t = tools

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_and_no_recording(self):
        with self.assertRaises(ValueError):
            self.t.transcribe("/no/such.mp3")
        with self.assertRaises(RuntimeError):
            self.t.record_stop()
        self.assertEqual(self.t.recording(), {"recording": False})


if __name__ == "__main__":
    unittest.main()
