import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sfedge.models import smokenet  # noqa: E402
from sfedge.qat import image_to_input, quantize_model  # noqa: E402
from sfedge.twin import export_onnx, float_twin  # noqa: E402
from tests.test_qat_export import calibrated, rgb_batch  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return calibrated(smokenet(width=0.5, tile=32), lambda n: image_to_input(rgb_batch(n)))


def test_twin_tracks_the_integer_model(model):
    twin = float_twin(model)
    imgs = rgb_batch(32)
    with torch.no_grad():
        sf = model(image_to_input(imgs))
        fl = twin(imgs.permute(0, 3, 1, 2).float())
    # the twin only skips activation rounding (half an SF8 step per layer)
    agree = (sf.argmax(1) == fl.argmax(1)).float().mean().item()
    assert agree >= 0.9
    assert torch.allclose(sf, fl, atol=0.25 * sf.abs().max().item())


def test_twin_requires_quantized_model():
    with pytest.raises(ValueError):
        float_twin(smokenet(width=0.5, tile=32))


def test_onnx_export(model, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / "twin.onnx"
    export_onnx(model, path)
    sess = ort.InferenceSession(str(path))
    x = rgb_batch(4).permute(0, 3, 1, 2).float().numpy()
    want = float_twin(model)(torch.from_numpy(x)).detach().numpy()
    got = sess.run(None, {"image": x})[0]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
