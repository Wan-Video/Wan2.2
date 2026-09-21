import time, uuid, shutil, os, threading
from pathlib import Path
from app.config import settings


class MockWorker:
    """Simulates video generation for local development without GPU"""

    def __init__(self):
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self, prompt: str, image_path: str | None = None,
        task: str = "ti2v-5B", size: str = "704*1280",
        steps: int = 30, guidance: float = 6.0,
        seed: int | None = None, output_path: str | None = None,
        progress_callback=None,
        timeout: int = 600,
    ) -> str:
        """Simulate generation — create a test video with FFmpeg"""

        if output_path is None:
            output_path = str(self.output_dir / f"mock_{uuid.uuid4().hex[:8]}.mp4")

        total_steps = steps
        for i in range(total_steps):
            time.sleep(0.1)  # Simulate compute
            if progress_callback:
                progress_callback((i + 1) / total_steps)

        # Generate a test video using FFmpeg (colored bars + text)
        test_cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=blue:s={size.replace('*', 'x')}:d=5:r=24",
            "-vf", f"drawtext=text='{prompt[:50]}':fontsize=24:fontcolor=white:x=10:y=10",
            "-c:v", "libx264", "-preset", "ultrafast",
            output_path,
        ]
        try:
            subprocess.run(test_cmd, capture_output=True, timeout=30)
        except Exception:
            # If ffmpeg not available, create a minimal valid MP4
            _create_dummy_mp4(output_path)

        return output_path

    @property
    def is_available(self) -> bool:
        return True


def _create_dummy_mp4(path: str):
    """Create a minimal valid MP4 file for testing"""
    # Minimal MP4 header (works in most players)
    data = bytes([
        0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70,
        0x6D, 0x70, 0x34, 0x32, 0x00, 0x00, 0x00, 0x00,
        0x6D, 0x70, 0x34, 0x32, 0x69, 0x73, 0x6F, 0x6D,
    ])
    with open(path, "wb") as f:
        f.write(data)


import subprocess
