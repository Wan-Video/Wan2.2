# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np


def sample_video_frame_indices(
    original_fps, total_frames, target_fps, num_frames, from_start=False
):
    """Return source-frame indices sampled at ``target_fps``.

    Sampling is defined in time rather than by a rounded integer stride. When
    the source frame rate is below ``target_fps``, repeated indices preserve
    the requested output rate. Target timestamps beyond the source duration
    are omitted instead of being clamped to the final frame.

    If ``from_start`` is false and the source is long enough, the sampling
    window is shifted so that its final index is the final source frame.
    """
    if not np.isfinite(original_fps) or original_fps <= 0:
        raise ValueError("original_fps must be a positive finite value")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a positive finite value")
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if total_frames == 0 or num_frames == 0:
        return np.empty(0, dtype=np.int64)

    sample_times = np.linspace(0, num_frames / target_fps, num_frames, endpoint=False)
    if not from_start:
        last_frame_time = (total_frames - 1) / original_fps
        start_time = max(0.0, last_frame_time - sample_times[-1])
        sample_times += start_time

    # A frame count describes the half-open source interval [0, duration).
    # Filtering timestamps first avoids extending a short pose clip by
    # repeatedly clamping every missing target frame to its final image.
    source_duration = total_frames / original_fps
    sample_times = sample_times[sample_times < source_duration]
    frame_indices = np.rint(sample_times * original_fps).astype(np.int64)
    return np.clip(frame_indices, 0, total_frames - 1)
