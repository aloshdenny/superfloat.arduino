import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sfedge.qat import (  # noqa: E402
    SFConv2d,
    SFGlobalAvgPool,
    SFLinear,
    act_quantize,
    choose_gain_exp,
    clamp_to_grid_,
    image_to_input,
    quantize_model,
    sf_quantize,
)


def test_sf_quantize_grid_and_bounded_ste():
    x = (torch.randn(1000) * 2).requires_grad_(True)
    q = sf_quantize(x, 8)
    assert torch.equal(q * 128, torch.round(q * 128))
    assert q.abs().max() <= 127 / 128
    q.sum().backward()
    assert torch.equal(x.grad, (x.detach().abs() <= 127 / 128).float())


def test_act_quantize_rounds_half_up():
    x = torch.tensor([0.5, 1.5, 2.5]) / 128
    assert torch.equal(act_quantize(x, "relu1") * 128, torch.tensor([1.0, 2.0, 3.0]))


def test_act_quantize_relu1_bounds():
    y = act_quantize(torch.tensor([-1.0, 0.3, 5.0]), "relu1")
    assert y[0] == 0 and y[2] == 127 / 128


@pytest.mark.parametrize("m, want", [(0.5, 0), (0.99, 0), (1.5, 1), (3.0, 2), (1000.0, 7)])
def test_choose_gain_exp(m, want):
    assert choose_gain_exp(torch.tensor([m]), 8) == want


def test_quantized_conv_weights_fit_after_gain():
    torch.manual_seed(0)
    layer = SFConv2d(3, 8)
    with torch.no_grad():
        layer.bn.weight.fill_(20.0)  # folded weights well outside [-1, 1)
    layer.eval()
    quantize_model(layer)
    w, _ = layer.folded()
    e = int(layer.gain_exp)
    assert e > 0 and (w / 2**e).abs().max() <= 1.0
    codes, bias, _ = layer.integer_params()
    assert codes.abs().max() <= 127 and bias.dtype == torch.int32


def test_sf4_codes_stay_in_range():
    layer = SFConv2d(4, 4, wbits=4).eval()
    quantize_model(layer)
    codes, _, _ = layer.integer_params()
    assert codes.abs().max() <= 7


def test_global_pool_quantized_matches_integer_rule():
    pool = SFGlobalAvgPool()
    pool.quantized = True
    x = torch.tensor([1.0, 2.0, -1.0, -2.0]).reshape(1, 2, 1, 2) / 128
    out = pool(x) * 128
    assert out.tolist() == [[2.0, -1.0]]


def test_clamp_to_grid_bounds_folded_weights():
    layer = SFConv2d(3, 4).eval()
    quantize_model(layer)
    with torch.no_grad():
        layer.conv.weight.mul_(100)
    clamp_to_grid_(layer)
    w, _ = layer.folded()
    limit = 127 / 128 * 2 ** int(layer.gain_exp)
    assert w.abs().max() <= limit * (1 + 1e-5)


def test_gradients_flow_in_quantized_phase():
    torch.manual_seed(0)
    model = nn.Sequential(SFConv2d(3, 4), SFGlobalAvgPool(), SFLinear(4, 2))
    model.eval()
    quantize_model(model)
    x = image_to_input(torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8))
    model(x).sum().backward()
    assert model[0].conv.weight.grad is not None
    assert model[0].bn.weight.grad is not None
    assert model[2].fc.weight.grad is not None


def test_image_to_input_layouts():
    img = torch.randint(0, 256, (1, 6, 5, 3), dtype=torch.uint8)
    a = image_to_input(img)
    b = image_to_input(img.permute(0, 3, 1, 2))
    assert a.shape == (1, 3, 6, 5) and torch.equal(a, b)
    assert math.isclose(float(image_to_input(torch.tensor([[[[0]]]], dtype=torch.uint8))), -1.0)
