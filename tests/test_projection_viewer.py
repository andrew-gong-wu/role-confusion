import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_projection_viewer.py"
SPEC = importlib.util.spec_from_file_location("projection_viewer", SCRIPT)
VIEWER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VIEWER)


class ProjectionViewerTests(unittest.TestCase):
    def test_byte_decoder_is_bijective(self):
        decoder = VIEWER.byte_decoder()
        self.assertEqual(len(decoder), 256)
        self.assertEqual(set(decoder.values()), set(range(256)))

    def test_token_index_for_byte(self):
        offsets = [0, 3, 4, 9]
        self.assertEqual(VIEWER.token_index_for_byte(offsets, 0), 0)
        self.assertEqual(VIEWER.token_index_for_byte(offsets, 3), 1)
        self.assertEqual(VIEWER.token_index_for_byte(offsets, 8), 2)


if __name__ == "__main__":
    unittest.main()
