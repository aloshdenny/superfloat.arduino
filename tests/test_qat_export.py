"""Trained == deployed: the quantised torch forward must equal the integer
runtimes bit for bit, so accuracy measured in training is what ships."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sfedge import reference as ref  # noqa: E402
from sfedge import sfm  # noqa: E402
from sfedge.export import logits_scale, to_c_header, to_graph  # noqa: E402
from sfedge.models import embernet, smokenet  # noqa: E402
from sfedge.qat import image_to_input, quantize_model  # noqa: E402


def calibrated(model, make_input, steps=8):
    """Give BatchNorm non-trivial running statistics, then quantise."""
    torch.manual_seed(0)
    model.train()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.uniform_(0.5, 3.0)
                m.bias.uniform_(-0.2, 0.2)
        for _ in range(steps):
            model(make_input(16))
    model.eval()
    return quantize_model(model)


def rgb_batch(n, size=32):
    return torch.randint(0, 256, (n, size, size, 3), dtype=torch.uint8)


@pytest.mark.parametrize("wbits", [8, 4])
def test_smokenet_torch_equals_reference(wbits):
    model = calibrated(smokenet(width=0.5, wbits=wbits, tile=32),
                       lambda n: image_to_input(rgb_batch(n)))
    graph = to_graph(model)
    imgs = rgb_batch(6)
    with torch.no_grad():
        logits = model(image_to_input(imgs)).numpy()
    raw = logits / logits_scale(graph)
    assert np.array_equal(raw, np.round(raw)), "torch logits left the accumulator grid"
    for img, want in zip(imgs.numpy(), raw.astype(np.int64)):
        np.testing.assert_array_equal(ref.run_rgb8(graph, img), want)


def test_embernet_torch_equals_reference():
    def thermal(n):
        codes = torch.randint(-127, 128, (n, 2, 24, 32)).float()
        return codes / 128

    model = calibrated(embernet(), thermal)
    graph = to_graph(model)
    x = thermal(4)
    with torch.no_grad():
        raw = (model(x) / logits_scale(graph)).numpy().astype(np.int64)
    for xi, want in zip(x, raw):
        codes = (xi.permute(1, 2, 0) * 128).numpy().astype(np.int8)
        np.testing.assert_array_equal(ref.run(graph, codes), want)


def test_export_requires_quantized_phase():
    with pytest.raises(ValueError, match="float phase"):
        to_graph(embernet())


def test_full_size_budgets():
    """Keep the documented compute and storage budgets honest."""
    g = to_graph(quantize_model(smokenet().eval()))
    assert 25e6 < g.macs() < 32e6
    assert 250e3 < g.weight_bytes() < 300e3
    e = to_graph(quantize_model(embernet().eval()))
    assert e.macs() < 1e6 and e.weight_bytes() < 20e3
    e4 = to_graph(quantize_model(embernet(wbits=4).eval()))
    assert e4.weight_bytes() <= e.weight_bytes() // 2 + 8


def test_c_header_contains_exact_bytes():
    data = sfm.dumps(to_graph(quantize_model(embernet().eval())))
    text = to_c_header(data, "ember_sfm")
    assert f"ember_sfm_len = {len(data)};" in text
    body = text[text.index("{") + 1:text.rindex("}")]
    vals = bytes(int(v, 16) for v in body.replace("\n", "").split(",") if v.strip())
    assert vals == data
