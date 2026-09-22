# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for the T5 encode path: fp32 compute, prompt batching and the embedding
cache.

T5 dominates generation on this setup -- ~153 s of a ~330 s run -- because
bfloat16 matmul is emulated on CPUs without AVX512-BF16/AMX. These tests pin
the mechanics; the speed and quality numbers live in benchmarks/.

None of this needs the 10.6 GB checkpoint.
"""
import torch
import torch.nn as nn

from wan.modules.t5 import (
    T5EncoderModel,
    _Fp32ComputeLinear,
    enable_fp32_compute_,
)


# --------------------------------------------------------------- fp32 compute
def test_fp32_linear_matches_a_float32_reference():
    """
    The wrapper must equal an fp32 Linear holding the same (bf16-rounded)
    weights: bf16 values are exactly representable in fp32, so casting is
    lossless and only the accumulation dtype changes.
    """
    torch.manual_seed(0)
    lin = nn.Linear(64, 32).to(torch.bfloat16)
    x = torch.randn(8, 64, dtype=torch.bfloat16)

    ref = nn.Linear(64, 32)
    ref.weight.data = lin.weight.data.float()
    ref.bias.data = lin.bias.data.float()

    got = _Fp32ComputeLinear(lin)(x)
    expected = ref(x.float()).to(torch.bfloat16)

    assert got.dtype == torch.bfloat16, 'must return the input dtype'
    assert torch.equal(got, expected)


def test_fp32_linear_handles_no_bias_and_extra_dims():
    lin = nn.Linear(16, 8, bias=False).to(torch.bfloat16)
    out = _Fp32ComputeLinear(lin)(torch.randn(2, 5, 16, dtype=torch.bfloat16))
    assert out.shape == (2, 5, 8)
    assert out.dtype == torch.bfloat16


def test_enable_fp32_compute_replaces_every_linear():
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.Sequential(nn.Linear(8, 8), nn.ReLU()),
        nn.LayerNorm(8),
    ).to(torch.bfloat16)

    n = enable_fp32_compute_(model)

    assert n == 2
    assert not any(isinstance(m, nn.Linear) for m in model.modules())
    # non-Linear modules are untouched
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_enable_fp32_compute_preserves_output():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    model = model.to(torch.bfloat16).eval()
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        before = model(x)
        enable_fp32_compute_(model)
        after = model(x)

    assert after.shape == before.shape
    assert after.dtype == before.dtype
    # fp32 accumulation is *more* accurate, so this is close but not identical
    rel = (after.float() - before.float()).abs().mean() / \
        before.float().abs().mean().clamp(min=1e-9)
    assert rel < 0.05, 'fp32 compute drifted too far: {:.4f}'.format(rel)


def test_fp32_compute_does_not_change_weight_storage_dtype():
    """The whole point is keeping the 10.6 GB footprint, not doubling it."""
    model = nn.Sequential(nn.Linear(64, 64)).to(torch.bfloat16)
    before = sum(p.numel() * p.element_size() for p in model.parameters())
    before += sum(b.numel() * b.element_size() for b in model.buffers())

    enable_fp32_compute_(model)

    after = sum(p.numel() * p.element_size() for p in model.parameters())
    after += sum(b.numel() * b.element_size() for b in model.buffers())
    assert after == before


# --------------------------------------------------------------------- cache
class _FakeEncoder(T5EncoderModel):
    """T5EncoderModel with the 11B forward replaced by a counter."""

    def __init__(self, cache_size=0):
        self.text_len = 512
        self.dtype = torch.bfloat16
        self.device = torch.device('cpu')
        self.checkpoint_path = '/fake/ckpt.pth'
        self.tokenizer_path = 'google/umt5-xxl'
        self.fp32_compute = False
        self.cache_size = cache_size
        from collections import OrderedDict
        self._cache = OrderedDict()
        self.encoded = []

    def _encode(self, texts, device):
        self.encoded.append(list(texts))
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
            out.append(torch.randn(4, 8, generator=g).to(device))
        return out


def test_cache_disabled_always_encodes():
    enc = _FakeEncoder(cache_size=0)
    enc(['a', 'b'], torch.device('cpu'))
    enc(['a', 'b'], torch.device('cpu'))
    assert enc.encoded == [['a', 'b'], ['a', 'b']]


def test_cache_hit_skips_the_forward():
    enc = _FakeEncoder(cache_size=2)
    first = enc(['pos', 'neg'], torch.device('cpu'))
    second = enc(['pos', 'neg'], torch.device('cpu'))

    assert enc.encoded == [['pos', 'neg']], 'second call re-encoded'
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_changed_prompt_only_encodes_the_new_one():
    """The negative prompt is the one that repeats; it must stay cached."""
    enc = _FakeEncoder(cache_size=2)
    enc(['pos1', 'neg'], torch.device('cpu'))
    enc(['pos2', 'neg'], torch.device('cpu'))

    assert enc.encoded == [['pos1', 'neg'], ['pos2']]


def test_changed_negative_prompt_is_a_miss():
    enc = _FakeEncoder(cache_size=4)
    enc(['pos', 'neg1'], torch.device('cpu'))
    enc(['pos', 'neg2'], torch.device('cpu'))
    assert enc.encoded == [['pos', 'neg1'], ['neg2']]


def test_cache_preserves_order():
    enc = _FakeEncoder(cache_size=4)
    ref = enc(['a', 'b', 'c'], torch.device('cpu'))
    enc(['c', 'a'], torch.device('cpu'))
    mixed = enc(['c', 'b', 'a'], torch.device('cpu'))

    assert torch.equal(mixed[0], ref[2])
    assert torch.equal(mixed[1], ref[1])
    assert torch.equal(mixed[2], ref[0])


def test_cache_is_bounded_lru():
    enc = _FakeEncoder(cache_size=2)
    enc(['a'], torch.device('cpu'))
    enc(['b'], torch.device('cpu'))
    enc(['c'], torch.device('cpu'))
    assert len(enc._cache) == 2
    enc(['a'], torch.device('cpu'))
    assert enc.encoded[-1] == ['a'], 'a should have been evicted'


def test_cache_key_separates_device_and_compute_mode():
    enc = _FakeEncoder(cache_size=4)
    k_cpu = enc._cache_key('x', torch.device('cpu'))
    k_cuda = enc._cache_key('x', torch.device('cuda'))
    assert k_cpu != k_cuda

    enc.fp32_compute = True
    assert enc._cache_key('x', torch.device('cpu')) != k_cpu

    enc.fp32_compute = False
    enc.checkpoint_path = '/other/ckpt.pth'
    assert enc._cache_key('x', torch.device('cpu')) != k_cpu


def test_clear_cache_forces_re_encode():
    enc = _FakeEncoder(cache_size=2)
    enc(['a'], torch.device('cpu'))
    enc.clear_cache()
    enc(['a'], torch.device('cpu'))
    assert enc.encoded == [['a'], ['a']]


def test_cached_tensor_is_not_aliased_to_the_caller():
    """A caller mutating a returned embedding must not corrupt the cache."""
    enc = _FakeEncoder(cache_size=2)
    first = enc(['a'], torch.device('cpu'))[0]
    original = first.clone()
    first.add_(99.0)

    again = enc(['a'], torch.device('cpu'))[0]
    assert torch.equal(again, original)


def test_str_input_is_accepted():
    enc = _FakeEncoder(cache_size=1)
    out = enc('a single prompt', torch.device('cpu'))
    assert len(out) == 1


# ------------------------------------------------------- pipeline integration
def _stub_pipeline(batch, cache_size=0):
    """WanTI2V with only the attributes _encode_prompts touches."""
    from wan.textimage2video import WanTI2V
    pipe = WanTI2V.__new__(WanTI2V)
    pipe.device = torch.device('cpu')
    pipe.t5_cpu = True
    pipe.t5_batch = batch
    pipe.text_encoder = _FakeEncoder(cache_size=cache_size)
    return pipe


def test_encode_prompts_batched_matches_unbatched_structure():
    """
    Batching must not reorder or reshape anything. _FakeEncoder is
    deterministic per text, so the two paths must agree exactly here; the real
    encoder differs only by GEMM reduction order.
    """
    seq = _stub_pipeline(batch=False)
    bat = _stub_pipeline(batch=True)

    c_seq, n_seq = seq._encode_prompts('positive', 'negative', False)
    c_bat, n_bat = bat._encode_prompts('positive', 'negative', False)

    assert len(c_seq) == len(c_bat) == 1
    assert torch.equal(c_seq[0], c_bat[0]), 'positive prompt got swapped'
    assert torch.equal(n_seq[0], n_bat[0]), 'negative prompt got swapped'


def test_encode_prompts_batched_uses_one_forward():
    bat = _stub_pipeline(batch=True)
    bat._encode_prompts('positive', 'negative', False)
    assert bat.text_encoder.encoded == [['positive', 'negative']]

    seq = _stub_pipeline(batch=False)
    seq._encode_prompts('positive', 'negative', False)
    assert seq.text_encoder.encoded == [['positive'], ['negative']]


def test_encode_prompts_cache_skips_second_call_entirely():
    """Repeat generation with the same prompts must not touch T5 at all."""
    pipe = _stub_pipeline(batch=True, cache_size=4)
    pipe._encode_prompts('positive', 'negative', False)
    pipe._encode_prompts('positive', 'negative', False)
    assert pipe.text_encoder.encoded == [['positive', 'negative']]


def test_encode_prompts_cache_reencodes_only_the_changed_prompt():
    pipe = _stub_pipeline(batch=True, cache_size=4)
    pipe._encode_prompts('prompt A', 'negative', False)
    pipe._encode_prompts('prompt B', 'negative', False)
    assert pipe.text_encoder.encoded == [['prompt A', 'negative'], ['prompt B']]
