import importlib.util
from pathlib import Path
import unittest

import torch


def load_scheduler(filename, classname):
    path = Path(__file__).resolve().parents[1] / 'wan' / 'utils' / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, classname)


DPM = load_scheduler('fm_solvers.py', 'FlowDPMSolverMultistepScheduler')
UNIPC = load_scheduler('fm_solvers_unipc.py', 'FlowUniPCMultistepScheduler')


class TestFlowNoiseConversion(unittest.TestCase):
    def test_noise_conversion_recovers_independent_endpoint(self):
        x0 = torch.tensor([[-2.0, 3.0], [0.1, -0.3]])
        noise = torch.tensor([[1.0, -4.0], [2.0, 0.6]])
        velocity = noise - x0
        schedulers = [
            DPM(algorithm_type='dpmsolver', final_sigmas_type='sigma_min'),
            DPM(algorithm_type='sde-dpmsolver', final_sigmas_type='sigma_min'),
            UNIPC(predict_x0=False),
        ]
        for scheduler in schedulers:
            for sigma in (0.0, 0.05, 0.5, 0.95, 1.0):
                with self.subTest(scheduler=type(scheduler).__name__, sigma=sigma):
                    scheduler.sigmas = torch.tensor([sigma])
                    scheduler._step_index = 0
                    sample = (1 - sigma) * x0 + sigma * noise
                    torch.testing.assert_close(scheduler.convert_model_output(velocity, sample=sample), noise)

    def test_default_data_prediction_recovers_original_endpoint(self):
        x0 = torch.tensor([-3.0, 2.0])
        noise = torch.tensor([1.0, -0.5])
        for scheduler in (DPM(), UNIPC()):
            for sigma in (0.0, 0.25, 0.75, 1.0):
                scheduler.sigmas = torch.tensor([sigma])
                scheduler._step_index = 0
                sample = (1 - sigma) * x0 + sigma * noise
                torch.testing.assert_close(scheduler.convert_model_output(noise - x0, sample=sample), x0)

    def test_conversion_preserves_flow_endpoint_gradients(self):
        for cls, options in ((DPM, {'algorithm_type': 'dpmsolver', 'final_sigmas_type': 'sigma_min'}),
                             (UNIPC, {'predict_x0': False})):
            scheduler = cls(**options)
            scheduler.sigmas = torch.tensor([0.3])
            scheduler._step_index = 0
            sample = torch.tensor([1.0, -2.0], requires_grad=True)
            velocity = torch.tensor([0.4, 0.1], requires_grad=True)
            converted = scheduler.convert_model_output(velocity, sample=sample)
            sample_grad, velocity_grad = torch.autograd.grad(converted.sum(), (sample, velocity))
            torch.testing.assert_close(sample_grad, torch.ones_like(sample))
            torch.testing.assert_close(velocity_grad, torch.full_like(velocity, 0.7))


if __name__ == '__main__':
    unittest.main()
