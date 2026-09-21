# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import importlib

from . import configs, distributed, modules

# Task pipelines are resolved on first attribute access (PEP 562) so that
# importing `wan` does not require the dependencies of tasks the user is not
# running. `WanS2V` needs decord and librosa, and `WanAnimate` needs peft and
# decord, all of which live in requirements_s2v.txt / requirements_animate.txt
# rather than requirements.txt. Access is unchanged: `wan.WanT2V(...)` works
# exactly as before.
_TASK_MODULES = {
    'WanI2V': '.image2video',
    'WanS2V': '.speech2video',
    'WanT2V': '.text2video',
    'WanTI2V': '.textimage2video',
    'WanAnimate': '.animate',
}

__all__ = ['configs', 'distributed', 'modules'] + list(_TASK_MODULES)


def __getattr__(name):
    if name in _TASK_MODULES:
        module = importlib.import_module(_TASK_MODULES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache, so this runs at most once per name
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(__all__)
