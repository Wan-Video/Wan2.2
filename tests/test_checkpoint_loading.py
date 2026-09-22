# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Guards against the torch.load default flipping out from under us.

PyTorch 2.6 changed `torch.load`'s `weights_only` default from False to True.
The Wan .pth checkpoints (the VAEs, T5, the CLIP tower, the animate LoRA)
are stored in the oldest tar-based serialization format, which torch.load
routes through `legacy_load` and which cannot be read under
weights_only=True at all -- so on any fresh install picking up torch>=2.6,
loading died with:

    RuntimeError: Cannot use ``weights_only=True`` with files saved in the
    legacy .tar format.

requirements.txt allows torch>=2.4.0 with no upper bound, so this hit every
new install. Each call site now passes weights_only explicitly; this test keeps
it that way.

No model weights needed.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _torch_load_calls():
    """Yield (path, lineno, node) for every `torch.load(...)` in the package."""
    for path in sorted(REPO_ROOT.glob('wan/**/*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (isinstance(fn, ast.Attribute) and fn.attr == 'load' and
                    isinstance(fn.value, ast.Name) and fn.value.id == 'torch'):
                yield path.relative_to(REPO_ROOT), node.lineno, node


def test_repo_has_torch_load_calls():
    """Sanity check that the scanner actually finds anything."""
    assert list(_torch_load_calls()), 'scanner found no torch.load calls'


def test_every_torch_load_sets_weights_only():
    offenders = [
        '{}:{}'.format(path, lineno)
        for path, lineno, node in _torch_load_calls()
        if not any(kw.arg == 'weights_only' for kw in node.keywords)
    ]
    assert not offenders, (
        'torch.load must pass weights_only explicitly -- the default flipped '
        'in torch 2.6 and these checkpoints are legacy .tar files: ' +
        ', '.join(offenders))
