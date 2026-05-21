"""Unit tests for serve module — no running server or Redis required.

Tests config parsing, job_build logic, worker_main routing, and
store signal serialization using mocks.

Usage:
  pytest tests/test_serve.py -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure serve package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@contextmanager
def set_env(**kwargs):
    """Temporarily set environment variables and restore on exit."""
    old = {}
    for k, v in kwargs.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    yield
    for k in kwargs:
        if old[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old[k]


# ============================================================
# serve.config
# ============================================================


class TestSettings:
    def test_defaults(self):
        with set_env(WAN_SERVE_API_KEYS="sk-test"):
            from serve.config import Settings
            s = Settings.from_env()
            assert s.redis_url == "redis://127.0.0.1:6379/0"
            assert s.nnodes == 1
            assert s.nproc_per_node == 1
            assert s.node_rank == 0
            assert s.node_role == "master"
            assert s.signal_key == "wan:signal"
            assert s.master_port == 29500
            assert "sk-test" in s.api_keys

    def test_multi_node_config(self):
        with set_env(
            WAN_SERVE_API_KEYS="sk-test",
            WAN_NNODES="2",
            WAN_NPROC_PER_NODE="4",
            WAN_NODE_RANK="1",
            WAN_NODE_ROLE="worker",
            WAN_MASTER_ADDR="10.0.0.1",
            WAN_MASTER_PORT="29600",
        ):
            from serve.config import Settings
            s = Settings.from_env()
            assert s.nnodes == 2
            assert s.nproc_per_node == 4
            assert s.node_rank == 1
            assert s.node_role == "worker"
            assert s.master_addr == "10.0.0.1"
            assert s.master_port == 29600

    def test_multiple_api_keys(self):
        with set_env(WAN_SERVE_API_KEYS="sk-one,sk-two,sk-three"):
            from serve.config import Settings
            s = Settings.from_env()
            assert s.api_keys == frozenset({"sk-one", "sk-two", "sk-three"})

    def test_empty_api_keys(self):
        # Clear any existing key env vars
        for var in ("WAN_SERVE_API_KEYS", "WAN_SERVE_API_KEY"):
            os.environ.pop(var, None)
        from serve.config import Settings
        s = Settings.from_env()
        assert s.api_keys == frozenset()

    def test_frozen(self):
        with set_env(WAN_SERVE_API_KEYS="sk-test"):
            from serve.config import Settings
            s = Settings.from_env()
            try:
                s.redis_url = "x"
                assert False, "Should be frozen"
            except AttributeError:
                pass


# ============================================================
# serve.job_build
# ============================================================


class TestJobBuild:
    def test_basic_t2v(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest

        env = {
            "WAN_SERVE_API_KEYS": "sk-test",
            "WAN_CKPT_DIR": "/ckpt",
            "WAN_OUTPUT_DIR": "/out",
        }
        for k, v in env.items():
            os.environ[k] = v

        try:
            s = Settings.from_env()
            req = VideoGenerationRequest(
                model="wan2.2-t2v-a14b",
                input={"prompt": "A cat"},
                parameters={"size": "832*480", "frame_num": 81},
            )
            job = request_to_job(req, task_id="wan-abc123", settings=s)
            assert job["model"] == "wan2.2-t2v-a14b"
            assert job["prompt"] == "A cat"
            assert job["size"] == "832*480"
            assert job["frame_num"] == 81
            assert job["ckpt_dir"] == "/ckpt/Wan2.2-T2V-A14B"
            assert job["save_file"] == "/out/wan-abc123.mp4"
        finally:
            for k in env:
                os.environ.pop(k, None)

    def test_parameters_ckpt_dir_overrides_global(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest

        env = {
            "WAN_SERVE_API_KEYS": "sk-test",
            "WAN_CKPT_DIR": "/global-ckpt",
            "WAN_OUTPUT_DIR": "/out",
        }
        for k, v in env.items():
            os.environ[k] = v

        try:
            s = Settings.from_env()
            req = VideoGenerationRequest(
                model="wan2.2-t2v-a14b",
                input={"prompt": "A cat"},
                parameters={"ckpt_dir": "/custom-ckpt"},
            )
            job = request_to_job(req, task_id="wan-xyz", settings=s)
            # per-request ckpt_dir takes precedence
            assert job["ckpt_dir"] == "/custom-ckpt"
        finally:
            for k in env:
                os.environ.pop(k, None)

    def test_none_params_excluded(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest

        env = {"WAN_SERVE_API_KEYS": "sk-test", "WAN_OUTPUT_DIR": "/out"}
        for k, v in env.items():
            os.environ[k] = v

        try:
            s = Settings.from_env()
            req = VideoGenerationRequest(model="wan2.2-t2v-a14b")
            job = request_to_job(req, task_id="wan-min", settings=s)
            # Parameters that were None should not appear
            assert "frame_num" not in job
            assert "base_seed" not in job
            # size gets a default value when not provided
            assert job["size"] == "1280*720"
            assert "model" in job
            assert "save_file" in job
        finally:
            for k in env:
                os.environ.pop(k, None)


# ============================================================
# serve.worker_main — routing logic
# ============================================================


class TestWorkerRouting:
    def test_master_role_calls_main_master(self):
        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NODE_ROLE"] = "master"

        try:
            with patch("serve.worker_main.main_master") as mock_master, \
                 patch("serve.worker_main.main_worker") as mock_worker:
                from serve.worker_main import main
                with patch("serve.worker_main.Settings") as MockSettings, \
                     patch("serve.worker_main.TaskStore"):
                    MockSettings.from_env.return_value = MagicMock(
                        job_dir="/tmp/jobs",
                        output_dir="/tmp/out",
                        node_role="master",
                    )
                    main()
                    mock_master.assert_called_once()
                    mock_worker.assert_not_called()
        finally:
            os.environ.pop("WAN_SERVE_API_KEYS", None)
            os.environ.pop("WAN_NODE_ROLE", None)

    def test_worker_role_calls_main_worker(self):
        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NODE_ROLE"] = "worker"

        try:
            with patch("serve.worker_main.main_master") as mock_master, \
                 patch("serve.worker_main.main_worker") as mock_worker:
                from serve.worker_main import main
                with patch("serve.worker_main.Settings") as MockSettings, \
                     patch("serve.worker_main.TaskStore"):
                    MockSettings.from_env.return_value = MagicMock(
                        job_dir="/tmp/jobs",
                        output_dir="/tmp/out",
                        node_role="worker",
                    )
                    main()
                    mock_worker.assert_called_once()
                    mock_master.assert_not_called()
        finally:
            os.environ.pop("WAN_SERVE_API_KEYS", None)
            os.environ.pop("WAN_NODE_ROLE", None)


# ============================================================
# serve.store — signal publish
# ============================================================


class TestStoreSignal:
    def test_publish_signal(self):
        from serve.config import Settings
        from serve.store import TaskStore

        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        try:
            s = Settings.from_env()
            store = TaskStore(s)
            mock_redis = MagicMock()
            store._r = mock_redis

            payload = json.dumps({"task_id": "wan-test", "rdzv_id": "wan-abc"})
            store.publish_signal(payload)

            mock_redis.publish.assert_called_once_with(s.signal_key, payload)
        finally:
            os.environ.pop("WAN_SERVE_API_KEYS", None)


# ============================================================
# serve.launcher — torchrun command construction
# ============================================================


class TestLauncher:
    def test_torchrun_cmd_master(self):
        from serve.config import Settings
        from serve.launcher import _torchrun_cmd

        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NNODES"] = "2"
        os.environ["WAN_NPROC_PER_NODE"] = "4"
        os.environ["WAN_NODE_RANK"] = "0"
        os.environ["WAN_MASTER_ADDR"] = "10.0.0.1"
        os.environ["WAN_MASTER_PORT"] = "29500"

        try:
            s = Settings.from_env()
            cmd = _torchrun_cmd(s, Path("/data/jobs/wan-test.json"), "wan-test-rdzv")
            assert cmd[0] == "torchrun"
            assert "--nnodes=2" in cmd
            assert "--nproc_per_node=4" in cmd
            assert "--node_rank=0" in cmd
            assert "--rdzv_backend=c10d" in cmd
            assert "--rdzv_endpoint=10.0.0.1:29500" in cmd
            assert "--rdzv_id=wan-test-rdzv" in cmd
            assert "--job_json" in cmd
        finally:
            for k in ("WAN_SERVE_API_KEYS", "WAN_NNODES", "WAN_NPROC_PER_NODE",
                       "WAN_NODE_RANK", "WAN_MASTER_ADDR", "WAN_MASTER_PORT"):
                os.environ.pop(k, None)

    def test_torchrun_cmd_worker(self):
        from serve.config import Settings
        from serve.launcher import _torchrun_cmd

        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NNODES"] = "2"
        os.environ["WAN_NPROC_PER_NODE"] = "4"
        os.environ["WAN_NODE_RANK"] = "1"
        os.environ["WAN_MASTER_ADDR"] = "10.0.0.1"

        try:
            s = Settings.from_env()
            cmd = _torchrun_cmd(s, Path("/data/jobs/wan-test.json"), "wan-test-rdzv")
            assert "--node_rank=1" in cmd
        finally:
            for k in ("WAN_SERVE_API_KEYS", "WAN_NNODES", "WAN_NPROC_PER_NODE",
                       "WAN_NODE_RANK", "WAN_MASTER_ADDR"):
                os.environ.pop(k, None)


# ============================================================
# serve.schemas — validation
# ============================================================


class TestSchemas:
    def test_video_generation_request_model_required(self):
        from serve.schemas import VideoGenerationRequest
        try:
            VideoGenerationRequest()
            assert False, "model is required"
        except Exception:
            pass

    def test_video_generation_request_with_model(self):
        from serve.schemas import VideoGenerationRequest
        req = VideoGenerationRequest(model="wan2.2-t2v-a14b")
        assert req.model == "wan2.2-t2v-a14b"
        assert req.input.prompt is None
        assert req.parameters.size is None

    def test_video_generation_request_full(self):
        from serve.schemas import VideoGenerationRequest
        req = VideoGenerationRequest(
            model="wan2.2-i2v-a14b",
            input={"prompt": "A cat", "image": "https://example.com/cat.jpg"},
            parameters={"size": "832*480", "frame_num": 81, "base_seed": 42},
        )
        assert req.input.prompt == "A cat"
        assert req.input.image == "https://example.com/cat.jpg"
        assert req.parameters.size == "832*480"
        assert req.parameters.frame_num == 81
        assert req.parameters.base_seed == 42

    def test_task_status_body(self):
        from serve.schemas import TaskStatusBody
        body = TaskStatusBody(task_id="wan-test", task_status="SUCCEEDED")
        assert body.task_id == "wan-test"
        assert body.task_status == "SUCCEEDED"
        assert body.message == ""
        assert body.output == {}

    def test_health_response(self):
        from serve.schemas import HealthResponse
        resp = HealthResponse()
        assert resp.status == "ok"


if __name__ == "__main__":
    unittest.main()


# ============================================================
# serve.schemas — ModelEnum
# ============================================================


class TestModelEnum:
    def test_all_models_defined(self):
        from serve.schemas import ModelEnum
        assert len(ModelEnum) == 5
        assert ModelEnum.t2v_a14b.value == "wan2.2-t2v-a14b"
        assert ModelEnum.i2v_a14b.value == "wan2.2-i2v-a14b"
        assert ModelEnum.ti2v_5b.value == "wan2.2-ti2v-5b"
        assert ModelEnum.s2v_14b.value == "wan2.2-s2v-14b"
        assert ModelEnum.animate_14b.value == "wan2.2-animate-14b"

    def test_invalid_model_rejected(self):
        from serve.schemas import VideoGenerationRequest
        try:
            VideoGenerationRequest(model="invalid-model")
            assert False, "Should reject invalid model"
        except Exception:
            pass


# ============================================================
# serve.api — per-model validation
# ============================================================


class TestModelValidation:
    """Test _validate_model_input for each model's required fields."""

    def _make_body(self, model, **input_kwargs):
        from serve.schemas import VideoGenerationRequest
        return VideoGenerationRequest(model=model, input=input_kwargs)

    def test_t2v_requires_prompt(self):
        from serve.api import _validate_model_input
        from fastapi import HTTPException
        body = self._make_body("wan2.2-t2v-a14b")
        try:
            _validate_model_input(body)
            assert False, "Should require prompt"
        except HTTPException as e:
            assert e.status_code == 400

    def test_t2v_with_prompt_passes(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        body = VideoGenerationRequest(
            model="wan2.2-t2v-a14b",
            input={"prompt": "A cat"},
        )
        _validate_model_input(body)  # should not raise

    def test_i2v_requires_image(self):
        from serve.api import _validate_model_input
        from fastapi import HTTPException
        body = self._make_body("wan2.2-i2v-a14b", prompt="A cat")
        try:
            _validate_model_input(body)
            assert False, "Should require image"
        except HTTPException as e:
            assert e.status_code == 400
            assert "image" in e.detail

    def test_i2v_with_image_passes(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        body = VideoGenerationRequest(
            model="wan2.2-i2v-a14b",
            input={"prompt": "A cat", "image": "/path/to/cat.jpg"},
        )
        _validate_model_input(body)

    def test_animate_requires_video(self):
        from serve.api import _validate_model_input
        from fastapi import HTTPException
        body = self._make_body("wan2.2-animate-14b", prompt="pose")
        try:
            _validate_model_input(body)
            assert False, "Should require video"
        except HTTPException as e:
            assert e.status_code == 400
            assert "video" in e.detail

    def test_animate_with_video_passes(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        body = VideoGenerationRequest(
            model="wan2.2-animate-14b",
            input={"prompt": "pose", "video": "/path/to/ref.mp4"},
        )
        _validate_model_input(body)

    def test_s2v_requires_image(self):
        from serve.api import _validate_model_input
        from fastapi import HTTPException
        body = self._make_body("wan2.2-s2v-14b", prompt="talk", audio="/path/to.wav")
        try:
            _validate_model_input(body)
            assert False, "Should require image"
        except HTTPException as e:
            assert e.status_code == 400
            assert "image" in e.detail

    def test_s2v_requires_audio_or_tts(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        from fastapi import HTTPException
        body = VideoGenerationRequest(
            model="wan2.2-s2v-14b",
            input={"prompt": "talk", "image": "/path/to/img.jpg"},
        )
        try:
            _validate_model_input(body)
            assert False, "Should require audio or enable_tts"
        except HTTPException as e:
            assert e.status_code == 400
            assert "audio" in e.detail

    def test_s2v_with_audio_passes(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        body = VideoGenerationRequest(
            model="wan2.2-s2v-14b",
            input={"prompt": "talk", "image": "/path/to/img.jpg", "audio": "/path/to.wav"},
        )
        _validate_model_input(body)

    def test_s2v_with_tts_passes(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        body = VideoGenerationRequest(
            model="wan2.2-s2v-14b",
            input={"prompt": "talk", "image": "/path/to/img.jpg"},
            parameters={"enable_tts": True},
        )
        _validate_model_input(body)

    def test_ti2v_with_prompt_passes(self):
        from serve.api import _validate_model_input
        from serve.schemas import VideoGenerationRequest
        # ti2v only requires prompt, image is optional
        body = VideoGenerationRequest(
            model="wan2.2-ti2v-5b",
            input={"prompt": "A cat"},
        )
        _validate_model_input(body)


# ============================================================
# serve.job_build — all models
# ============================================================


class TestJobBuildAllModels:
    def _make_settings(self):
        from serve.config import Settings
        env = {"WAN_SERVE_API_KEYS": "sk-test", "WAN_CKPT_DIR": "/ckpt", "WAN_OUTPUT_DIR": "/out"}
        for k, v in env.items():
            os.environ[k] = v
        s = Settings.from_env()
        for k in env:
            os.environ.pop(k, None)
        return s

    def test_t2v_default_size(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-t2v-a14b",
            input={"prompt": "A cat"},
        )
        job = request_to_job(req, task_id="wan-t2v", settings=s)
        assert job["size"] == "1280*720"

    def test_i2v_default_size(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-i2v-a14b",
            input={"prompt": "A cat", "image": "/img.jpg"},
        )
        job = request_to_job(req, task_id="wan-i2v", settings=s)
        assert job["size"] == "832*480"
        assert job["image"] == "/img.jpg"

    def test_ti2v_default_size(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-ti2v-5b",
            input={"prompt": "A cat"},
        )
        job = request_to_job(req, task_id="wan-ti2v", settings=s)
        assert job["size"] == "1280*704"

    def test_s2v_default_size(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-s2v-14b",
            input={"prompt": "talk", "image": "/img.jpg", "audio": "/talk.wav"},
        )
        job = request_to_job(req, task_id="wan-s2v", settings=s)
        assert job["size"] == "832*480"
        assert job["audio"] == "/talk.wav"

    def test_animate_default_size(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-animate-14b",
            input={"prompt": "pose", "video": "/ref.mp4"},
        )
        job = request_to_job(req, task_id="wan-ani", settings=s)
        assert job["size"] == "720*1280"
        assert job["src_root_path"] == "/ref.mp4"
        assert "video" not in job

    def test_explicit_size_overrides_default(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-t2v-a14b",
            input={"prompt": "A cat"},
            parameters={"size": "480*832"},
        )
        job = request_to_job(req, task_id="wan-exp", settings=s)
        assert job["size"] == "480*832"


# ============================================================
# serve.job_build — ckpt_dir auto-mapping
# ============================================================


class TestCkptDirMapping:
    def _make_settings(self, ckpt_dir="/ckpt"):
        from serve.config import Settings
        env = {"WAN_SERVE_API_KEYS": "sk-test", "WAN_CKPT_DIR": ckpt_dir, "WAN_OUTPUT_DIR": "/out"}
        for k, v in env.items():
            os.environ[k] = v
        s = Settings.from_env()
        for k in env:
            os.environ.pop(k, None)
        return s

    def test_t2v_auto_ckpt_dir(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings("/ckpt")
        req = VideoGenerationRequest(model="wan2.2-t2v-a14b", input={"prompt": "A cat"})
        job = request_to_job(req, task_id="wan-t2v", settings=s)
        assert job["ckpt_dir"] == "/ckpt/Wan2.2-T2V-A14B"

    def test_i2v_auto_ckpt_dir(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings("/ckpt")
        req = VideoGenerationRequest(model="wan2.2-i2v-a14b", input={"prompt": "A cat", "image": "/img.jpg"})
        job = request_to_job(req, task_id="wan-i2v", settings=s)
        assert job["ckpt_dir"] == "/ckpt/Wan2.2-I2V-A14B"

    def test_s2v_auto_ckpt_dir(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings("/ckpt")
        req = VideoGenerationRequest(model="wan2.2-s2v-14b", input={"prompt": "talk", "image": "/img.jpg", "audio": "/a.wav"})
        job = request_to_job(req, task_id="wan-s2v", settings=s)
        assert job["ckpt_dir"] == "/ckpt/Wan2.2-S2V-14B"

    def test_animate_auto_ckpt_dir(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings("/ckpt")
        req = VideoGenerationRequest(model="wan2.2-animate-14b", input={"prompt": "pose", "video": "/ref.mp4"})
        job = request_to_job(req, task_id="wan-ani", settings=s)
        assert job["ckpt_dir"] == "/ckpt/Wan2.2-Animate-14B"

    def test_ti2v_auto_ckpt_dir(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings("/ckpt")
        req = VideoGenerationRequest(model="wan2.2-ti2v-5b", input={"prompt": "A cat"})
        job = request_to_job(req, task_id="wan-ti2v", settings=s)
        assert job["ckpt_dir"] == "/ckpt/Wan2.2-TI2V-5B"

    def test_parameters_ckpt_dir_overrides_auto(self):
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings("/ckpt")
        req = VideoGenerationRequest(
            model="wan2.2-t2v-a14b",
            input={"prompt": "A cat"},
            parameters={"ckpt_dir": "/custom/path"},
        )
        job = request_to_job(req, task_id="wan-custom", settings=s)
        assert job["ckpt_dir"] == "/custom/path"

    def test_no_global_ckpt_dir_no_parameters_ckpt_dir(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        env = {"WAN_SERVE_API_KEYS": "sk-test", "WAN_OUTPUT_DIR": "/out"}
        for k, v in env.items():
            os.environ[k] = v
        # Ensure WAN_CKPT_DIR is not set
        os.environ.pop("WAN_CKPT_DIR", None)
        try:
            s = Settings.from_env()
            assert s.ckpt_dir == ""
            req = VideoGenerationRequest(model="wan2.2-t2v-a14b", input={"prompt": "A cat"})
            job = request_to_job(req, task_id="wan-nockpt", settings=s)
            # No ckpt_dir set anywhere — it should not appear in job
            assert "ckpt_dir" not in job
        finally:
            for k in env:
                os.environ.pop(k, None)


# ============================================================
# serve.job_build — video → src_root_path mapping
# ============================================================


class TestVideoMapping:
    def _make_settings(self):
        from serve.config import Settings
        env = {"WAN_SERVE_API_KEYS": "sk-test", "WAN_CKPT_DIR": "/ckpt", "WAN_OUTPUT_DIR": "/out"}
        for k, v in env.items():
            os.environ[k] = v
        s = Settings.from_env()
        for k in env:
            os.environ.pop(k, None)
        return s

    def test_video_maps_to_src_root_path(self):
        """VideoInput.video should be mapped to src_root_path in the job dict."""
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-animate-14b",
            input={"prompt": "pose", "video": "/ckpt/animate_input"},
        )
        job = request_to_job(req, task_id="wan-ani", settings=s)
        assert "src_root_path" in job
        assert job["src_root_path"] == "/ckpt/animate_input"
        assert "video" not in job

    def test_explicit_src_root_path_not_overridden(self):
        """If both video and src_root_path are provided, src_root_path wins."""
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest
        s = self._make_settings()
        req = VideoGenerationRequest(
            model="wan2.2-animate-14b",
            input={"prompt": "pose", "video": "/ckpt/video_path"},
            parameters={"src_root_path": "/ckpt/custom_path"},
        )
        job = request_to_job(req, task_id="wan-ani-exp", settings=s)
        assert job["src_root_path"] == "/ckpt/custom_path"
        assert "video" not in job