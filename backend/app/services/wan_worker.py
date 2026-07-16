import subprocess, time, os, uuid
from pathlib import Path


class Wan2_2Worker:
    """Wrapper for Wan2.2 generate.py — runs on GPU machines"""

    def __init__(self, wan_path: str, ckpt_dir: str):
        self.wan_path = Path(wan_path)
        self.ckpt_dir = ckpt_dir
        self.generate_script = self.wan_path / "generate.py"

    def generate(
        self, prompt: str, image_path: str | None = None,
        task: str = "ti2v-5B", size: str = "704*1280",
        steps: int = 30, guidance: float = 6.0,
        seed: int | None = None, output_path: str | None = None,
        timeout: int = 600,
    ) -> str:
        cmd = [
            "python", str(self.generate_script),
            "--task", task, "--size", size,
            "--ckpt_dir", self.ckpt_dir,
            "--offload_model", "True", "--convert_model_dtype", "--t5_cpu",
            "--prompt", prompt,
            "--sample_steps", str(steps),
            "--sample_guide_scale", str(guidance),
        ]
        if image_path:
            cmd.extend(["--image", image_path])
        if seed is not None:
            cmd.extend(["--base_seed", str(seed)])

        result = subprocess.run(
            cmd, cwd=str(self.wan_path),
            capture_output=True, text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Wan2.2 failed:\n{result.stderr[-3000:]}")

        for mp4 in sorted(self.wan_path.glob("*.mp4")):
            if output_path:
                mp4.rename(output_path)
                return output_path
            return str(mp4)
        for mp4 in sorted((self.wan_path / "outputs").glob("*.mp4")):
            if output_path:
                mp4.rename(output_path)
                return output_path
            return str(mp4)

        raise RuntimeError("No MP4 output found")

    @property
    def is_available(self) -> bool:
        return self.generate_script.exists() and (self.wan_path / self.ckpt_dir).exists()
