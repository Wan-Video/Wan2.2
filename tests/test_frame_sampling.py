import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "wan" / "utils" / "frame_sampling.py"
SPEC = importlib.util.spec_from_file_location("frame_sampling", MODULE_PATH)
FRAME_SAMPLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRAME_SAMPLING)
sample_video_frame_indices = FRAME_SAMPLING.sample_video_frame_indices


FPS_VALUES = [15, 16, 24, 29.97, 30, 60]


@pytest.mark.parametrize("original_fps", FPS_VALUES)
def test_from_start_matches_audio_sampling(original_fps):
    target_fps = 16
    num_frames = 80
    total_frames = round(original_fps * 10)

    indices = sample_video_frame_indices(
        original_fps, total_frames, target_fps, num_frames, from_start=True
    )

    # Keep this reference equivalent to audio_encoder.get_sample_indices.
    target_times = np.linspace(0, num_frames / target_fps, num_frames, endpoint=False)
    audio_indices = np.round(target_times * original_fps).astype(int)
    audio_indices = np.clip(audio_indices, 0, total_frames - 1)

    np.testing.assert_array_equal(indices, audio_indices)
    assert len(indices) == num_frames


@pytest.mark.parametrize("original_fps", FPS_VALUES)
def test_five_second_clip_yields_eighty_frames(original_fps):
    indices = sample_video_frame_indices(
        original_fps,
        round(original_fps * 5),
        target_fps=16,
        num_frames=80,
        from_start=True,
    )

    assert len(indices) == 80
    assert np.all(indices[1:] >= indices[:-1])


def test_upsampling_repeats_source_frames():
    indices = sample_video_frame_indices(
        original_fps=15, total_frames=75, target_fps=16, num_frames=80, from_start=True
    )

    assert len(indices) == 80
    assert len(np.unique(indices)) == 75


def test_short_clip_omits_out_of_duration_timestamps():
    indices = sample_video_frame_indices(
        original_fps=30, total_frames=60, target_fps=16, num_frames=80, from_start=True
    )

    assert len(indices) == 32
    assert indices[-1] == 58


@pytest.mark.parametrize("original_fps", FPS_VALUES)
def test_from_end_preserves_final_frame(original_fps):
    total_frames = round(original_fps * 10)
    indices = sample_video_frame_indices(
        original_fps, total_frames, target_fps=16, num_frames=80, from_start=False
    )

    assert len(indices) == 80
    assert indices[-1] == total_frames - 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"original_fps": 0}, "original_fps"),
        ({"original_fps": np.nan}, "original_fps"),
        ({"target_fps": 0}, "target_fps"),
        ({"total_frames": -1}, "total_frames"),
        ({"num_frames": -1}, "num_frames"),
    ],
)
def test_invalid_arguments(kwargs, message):
    arguments = {
        "original_fps": 30,
        "total_frames": 300,
        "target_fps": 16,
        "num_frames": 80,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        sample_video_frame_indices(**arguments)


def test_empty_source_or_request_returns_empty_indices():
    assert sample_video_frame_indices(30, 0, 16, 80).size == 0
    assert sample_video_frame_indices(30, 300, 16, 0).size == 0
