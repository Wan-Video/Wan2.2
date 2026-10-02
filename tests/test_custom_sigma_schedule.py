import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


def load_module(filename):
    path = Path(__file__).resolve().parents[1] / 'wan' / 'utils' / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dpm = load_module('fm_solvers.py')
unipc = load_module('fm_solvers_unipc.py')
CLASSES = (dpm.FlowDPMSolverMultistepScheduler, unipc.FlowUniPCMultistepScheduler)


class TestCustomSigmaSchedule(unittest.TestCase):
    def test_python_list_and_array_schedules_match_static_shift(self):
        values = [0.9, 0.5, 0.1]
        expected = np.asarray(values) * 3 / (1 + 2 * np.asarray(values))
        for cls in CLASSES:
            with self.subTest(scheduler=cls.__name__):
                listed, array = cls(), cls()
                listed.set_timesteps(sigmas=values, shift=3.0)
                array.set_timesteps(sigmas=np.asarray(values), shift=3.0)
                torch.testing.assert_close(listed.sigmas, array.sigmas)
                torch.testing.assert_close(listed.timesteps, array.timesteps)
                np.testing.assert_allclose(listed.sigmas[:-1].numpy(), expected, rtol=1e-6)
                self.assertEqual(listed.sigmas[-1].item(), 0)
                self.assertEqual(listed.num_inference_steps, len(values))

    def test_python_list_schedule_supports_dynamic_shift(self):
        values = [0.8, 0.4, 0.05]
        mu = 0.7
        expected = np.exp(mu) / (np.exp(mu) + 1 / np.asarray(values) - 1)
        for cls in CLASSES:
            with self.subTest(scheduler=cls.__name__):
                scheduler = cls(use_dynamic_shifting=True)
                scheduler.set_timesteps(sigmas=values, mu=mu)
                np.testing.assert_allclose(scheduler.sigmas[:-1].numpy(), expected, rtol=1e-6)

    def test_retrieve_timesteps_accepts_documented_list_input(self):
        scheduler = dpm.FlowDPMSolverMultistepScheduler()
        timesteps, count = dpm.retrieve_timesteps(scheduler, sigmas=[0.95, 0.6, 0.2], shift=2.0)
        self.assertEqual(count, 3)
        self.assertEqual(len(timesteps), 3)
        self.assertEqual(len(scheduler.sigmas), 4)
        self.assertEqual(timesteps.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
