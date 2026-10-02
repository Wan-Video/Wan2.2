import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch


path = Path(__file__).resolve().parents[1] / 'wan' / 'utils' / 'qwen_vl_utils.py'
spec = importlib.util.spec_from_file_location('wan_qwen_vl_video_utils', path)
video_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_utils)


class TestVideoReaderBackend(unittest.TestCase):
    def setUp(self):
        video_utils.get_video_reader_backend.cache_clear()

    def tearDown(self):
        video_utils.get_video_reader_backend.cache_clear()

    def test_backend_choice_logs_without_invalid_logging_kwargs(self):
        for forced, available, expected in ((None, False, 'torchvision'), (None, True, 'decord'),
                                            ('torchvision', True, 'torchvision')):
            with self.subTest(forced=forced, available=available):
                video_utils.get_video_reader_backend.cache_clear()
                with mock.patch.object(video_utils, 'FORCE_QWENVL_VIDEO_READER', forced), \
                     mock.patch.object(video_utils, 'is_decord_available', return_value=available), \
                     self.assertLogs(video_utils.logger, level='INFO') as logged:
                    self.assertEqual(video_utils.get_video_reader_backend(), expected)
                self.assertIn(expected, logged.output[0])

    def test_fetch_video_uses_selected_reader_and_resizes_native_tensor(self):
        frames = torch.arange(4 * 3 * 56 * 56, dtype=torch.uint8).reshape(4, 3, 56, 56)
        reader = mock.Mock(return_value=frames)
        with mock.patch.object(video_utils, 'FORCE_QWENVL_VIDEO_READER', 'torchvision'), \
             mock.patch.dict(video_utils.VIDEO_READER_BACKENDS, {'torchvision': reader}):
            result = video_utils.fetch_video({'video': 'example.mp4', 'resized_height': 56, 'resized_width': 56})
        self.assertEqual(result.shape, frames.shape)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, frames.float())
        reader.assert_called_once()


if __name__ == '__main__':
    unittest.main()
