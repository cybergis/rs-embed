import numpy as np
import pytest
import torch

from rs_embed.embedders import onthefly_terrafm as tf


def test_s2_scaling_uses_terrafm_l2a_stats():
    stats = tf._load_terrafm_module().STATS["s2_l2a"]
    mean, std = np.array(stats["mean"]), np.array(stats["std"])
    x = np.full((12, 4, 4), 1500.0, dtype=np.float32)
    out = tf._terrafm_scale_chw(x, modality="s2")
    expected = (1500.0 - (mean - 2 * std)) / (4 * std)
    np.testing.assert_allclose(out[:, 0, 0], expected, rtol=1e-5)


def test_s1_scaling_converts_linear_to_db():
    stats = tf._load_terrafm_module().STATS["s1"]
    mean, std = np.array(stats["mean"]), np.array(stats["std"])
    x = np.full((2, 4, 4), 0.05, dtype=np.float32)  # linear power
    out = tf._terrafm_scale_chw(x, modality="s1")
    db = 10 * np.log10(0.05)
    np.testing.assert_allclose(out[:, 0, 0], (db - (mean - 2 * std)) / (4 * std), rtol=1e-5)


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [None]
        self.calls = []

    def forward(self, x, is_l2a=False):
        self.calls.append(is_l2a)
        return torch.zeros(x.shape[0], 8)

    def extract_feature(self, x, return_h_w=True, out_indices=(0,), is_l2a=False):
        self.calls.append(is_l2a)
        return [torch.zeros(x.shape[0], 8, 14, 14)]


@pytest.mark.parametrize("channels,expected", [(12, True), (2, False)])
def test_forward_selects_l2a_stem_for_s2(channels, expected):
    model = _RecordingModel()
    x = np.zeros((1, channels, 224, 224), dtype=np.float32)
    tf._terrafm_pooled_and_grid_batch(model, x, device="cpu", want_grid=True)
    tf._terrafm_pooled_and_grid(model, x, device="cpu", want_grid=True)
    assert model.calls == [expected] * 4
