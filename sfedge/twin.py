"""Float twin of a quantised SuperFloat model, for Edge Impulse and ONNX tools.

Edge Impulse ingests ordinary float models (ONNX, SavedModel, TFLite). The
twin is a plain torch module with the same folded, grid-snapped weights as
the SF model and clipped activations, but no activation rounding, so it can
be profiled, deployed as an .eim baseline, or quantised by Edge Impulse's own
int8 pipeline for a like-for-like comparison against the SuperFloat runtime.

Input is a float image in 0..255 (Edge Impulse's "0..255" scaling) with the
(p - 128) / 128 normalisation baked into the graph, so no preprocessing
needs to be configured on the Edge Impulse side.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .qat import SF8, SFConv2d, SFGlobalAvgPool, SFLinear


class _Normalize(nn.Module):
    def forward(self, x):
        return (x - 128.0) / 128.0


def _act(act: str) -> nn.Module:
    if act == "relu1":
        return nn.Hardtanh(0.0, SF8.vmax)
    if act == "linear":
        return nn.Hardtanh(-SF8.vmax, SF8.vmax)
    return nn.Identity()


@torch.no_grad()
def float_twin(model: nn.Sequential) -> nn.Sequential:
    layers: list[nn.Module] = [_Normalize()]
    for m in model.children():
        if isinstance(m, SFConv2d):
            if not bool(m.quantized):
                raise ValueError("quantise the model before building its twin")
            w, b = m.grid_params()
            c = m.conv
            conv = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding,
                             groups=c.groups, bias=True)
            conv.weight.copy_(w)
            conv.bias.copy_(b)
            layers += [conv, _act(m.act)]
        elif isinstance(m, SFGlobalAvgPool):
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        elif isinstance(m, SFLinear):
            w, b = m.grid_params()
            fc = nn.Linear(w.shape[1], w.shape[0])
            fc.weight.copy_(w)
            fc.bias.copy_(b)
            layers += [fc, _act(m.act)]
        else:
            raise TypeError(f"cannot twin {type(m).__name__}")
    return nn.Sequential(*layers).eval()


def export_onnx(model, path, opset: int = 17) -> None:
    """Write the twin as ONNX with an NCHW float input named "image"."""
    h, w, c = model.input_shape
    twin = float_twin(model)
    dummy = torch.full((1, c, h, w), 128.0)
    # The TorchScript exporter is deprecated but needs no onnxscript, and a
    # plain Sequential of convs is exactly the graph it handles well.
    torch.onnx.export(twin, (dummy,), str(path), input_names=["image"], output_names=["logits"],
                      opset_version=opset, dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
