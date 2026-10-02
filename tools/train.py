#!/usr/bin/env python3
"""Two-phase SuperFloat training: float first, then QAT on the integer grid.

    # SmokeNet on prepared tiles (expects train/ and val/ under --data)
    python tools/train.py smokenet --data data/tiles --out runs/smoke-sf8

    # EmberNet bootstrap model from synthetic thermal frames
    python tools/train.py embernet --synthetic 20000 --out runs/ember-synth

Writes <out>/model.pt (quantised state dict), <out>/<name>.sfm, a C header
next to it, and <out>/metrics.json. The exported file is checked against the
torch model on validation data before the run is reported as done.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from sfedge import reference, thermal  # noqa: E402
from sfedge.data import TileFolder, class_weights  # noqa: E402
from sfedge.export import export  # noqa: E402
from sfedge.metrics import confusion, format_report, report  # noqa: E402
from sfedge.models import EMBER_LABELS, SMOKE_LABELS, embernet, smokenet  # noqa: E402
from sfedge.qat import clamp_to_grid_, image_to_input, quantize_model  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(args):
    if args.model == "smokenet":
        return smokenet(width=args.width, wbits=args.wbits, tile=args.tile), SMOKE_LABELS
    return embernet(wbits=args.wbits), EMBER_LABELS


def loaders(args, labels):
    if args.synthetic:
        x, y = thermal.synthetic_dataset(args.synthetic, seed=args.seed)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        n_val = max(1, len(x) // 10)
        train = TensorDataset(x[n_val:], y[n_val:])
        val = TensorDataset(x[:n_val], y[:n_val])
        counts = np.bincount(y[n_val:].numpy(), minlength=len(labels))
    else:
        train = TileFolder(Path(args.data) / "train", labels, augment=True, seed=args.seed)
        val = TileFolder(Path(args.data) / "val", labels)
        counts = train.counts()
    kw = dict(batch_size=args.batch, num_workers=args.workers)
    return DataLoader(train, shuffle=True, drop_last=True, **kw), DataLoader(val, **kw), counts


def to_input(x: torch.Tensor) -> torch.Tensor:
    """uint8 RGB tiles or int8 thermal codes (NHWC) -> real NCHW inputs."""
    if x.dtype == torch.uint8:
        return image_to_input(x)
    return x.permute(0, 3, 1, 2).float() / 128.0


def run_epoch(model, loader, device, loss_fn, opt=None, quantized=False):
    train = opt is not None
    model.train(train)
    total, n, preds, trues = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = to_input(x).to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if quantized:
                    clamp_to_grid_(model)
            total += loss.item() * len(y)
            n += len(y)
            preds.append(logits.argmax(1).cpu())
            trues.append(y.cpu())
    return total / max(n, 1), torch.cat(trues).numpy(), torch.cat(preds).numpy()


def phase(name, model, train_dl, val_dl, device, loss_fn, epochs, lr, labels, quantized):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        t0 = time.time()
        tl, _, _ = run_epoch(model, train_dl, device, loss_fn, opt, quantized)
        vl, yt, yp = run_epoch(model, val_dl, device, loss_fn)
        sched.step()
        rep = report(confusion(yt, yp, len(labels)), labels)
        score = float(np.mean([m["recall"] for m in rep["per_class"].values() if m["support"]]))
        history.append({"phase": name, "epoch": ep, "train_loss": tl, "val_loss": vl,
                        "balanced_acc": score, **{k: rep[k] for k in ("hazard_recall", "false_alarm_rate")}})
        print(f"[{name} {ep + 1}/{epochs}] loss {tl:.4f}/{vl:.4f}  bal-acc {score:.4f}  "
              f"recall {rep['hazard_recall']:.4f}  false-alarm {rep['false_alarm_rate']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def check_export(model, graph, val_dl, limit=64) -> int:
    """Run the integer reference on validation inputs; count argmax disagreements."""
    model.eval().cpu()
    bad = seen = 0
    for x, _ in val_dl:
        want = model(to_input(x)).argmax(1)
        for xi, wi in zip(x, want):
            if xi.dtype == torch.uint8:
                got = reference.run_rgb8(graph, xi.numpy())
            else:
                got = reference.run(graph, xi.numpy().astype(np.int8))
            bad += int(np.argmax(got) != int(wi))
            seen += 1
            if seen >= limit:
                return bad
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=["smokenet", "embernet"])
    p.add_argument("--data", help="tile root with train/ and val/ (smokenet)")
    p.add_argument("--synthetic", type=int, default=0, help="synthetic thermal samples (embernet)")
    p.add_argument("--wbits", type=int, default=8, choices=[4, 8])
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--qat-epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    # SF8 needs a lower learning rate than FP32 because its grid is coarse
    # (aloshdenny/superfloat, SUPERFLOAT_RESULTS.md section 8); QAT lower still.
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--qat-lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.model == "smokenet" and not args.data:
        p.error("smokenet needs --data")
    if args.model == "embernet" and not (args.data or args.synthetic):
        p.error("embernet needs --data or --synthetic")

    torch.manual_seed(args.seed)
    # TF32 would round SF products inside the accumulation and break the
    # exact torch == runtime correspondence the QAT phase relies on.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    model, labels = build(args)
    train_dl, val_dl, counts = loaders(args, labels)
    weights = torch.tensor(class_weights(counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"{model.model_name} on {device}; train counts {dict(zip(labels, counts.tolist()))}")

    model.to(device)
    history = phase("float", model, train_dl, val_dl, device, loss_fn, args.epochs, args.lr, labels, False)
    model.eval()
    quantize_model(model)
    history += phase("qat", model, train_dl, val_dl, device, loss_fn, args.qat_epochs, args.qat_lr, labels, True)

    model.cpu().eval()
    torch.save(model.state_dict(), args.out / "model.pt")
    symbol = model.model_name.replace("-", "_").replace(".", "_") + "_sfm"
    graph = export(model, args.out / f"{model.model_name}.sfm", header_symbol=symbol)
    mismatches = check_export(model, graph, val_dl)
    _, yt, yp = run_epoch(model, val_dl, torch.device("cpu"), loss_fn)
    final = report(confusion(yt, yp, len(labels)), labels)
    print(format_report(final))
    print(f"exported {graph.name}: {graph.macs() / 1e6:.2f} M MACs, {graph.weight_bytes() / 1024:.1f} KiB "
          f"weights, {mismatches} export mismatches on validation")
    (args.out / "metrics.json").write_text(json.dumps({
        "model": graph.name, "args": {k: str(v) for k, v in vars(args).items()},
        "final": final, "history": history, "export_mismatches": mismatches,
        "macs": graph.macs(), "weight_bytes": graph.weight_bytes(),
    }, indent=2))
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
