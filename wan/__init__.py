# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import importlib

from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V

# WanS2V and WanAnimate pull in heavy optional dependencies (decord, peft,
# einops, opencv, ...) that the t2v / i2v / ti2v paths never touch. Importing
# them eagerly made `import wan` fail outright unless every optional stack was
# installed, so they are resolved on first attribute access instead.
_LAZY_PIPELINES = {
    'WanS2V': ('.speech2video', 'requirements_s2v.txt'),
    'WanAnimate': ('.animate', 'requirements_animate.txt'),
}

__all__ = ['WanI2V', 'WanT2V', 'WanTI2V', 'WanS2V', 'WanAnimate']


def __getattr__(name):
    if name in _LAZY_PIPELINES:
        module_name, requirements = _LAZY_PIPELINES[name]
        try:
            module = importlib.import_module(module_name, __name__)
        except ImportError as e:
            raise ImportError(
                '{} needs the optional dependencies listed in {}. Install them '
                'with: pip install -r {}\nOriginal error: {}'.format(
                    name, requirements, requirements, e)) from e
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError('module {!r} has no attribute {!r}'.format(
        __name__, name))


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY_PIPELINES)))
