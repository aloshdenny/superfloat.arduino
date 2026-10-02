"""Classification metrics framed the way an alerting system is judged."""

from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows are true labels, columns predictions."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m


def report(cm: np.ndarray, labels) -> dict:
    """Per-class precision/recall plus the two numbers that matter here.

    hazard_recall: fraction of tiles with smoke or flame (any non-zero class)
    flagged as any hazard. A smoke tile called flame is still a caught fire.
    false_alarm_rate: fraction of clear tiles flagged as a hazard.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    out = {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "support": int(s)}
            for name, p, r, s in zip(labels, precision, recall, cm.sum(1))
        },
    }
    hazard = cm[1:].sum()
    clear = cm[0].sum()
    out["hazard_recall"] = float(cm[1:, 1:].sum() / hazard) if hazard else float("nan")
    out["false_alarm_rate"] = float(cm[0, 1:].sum() / clear) if clear else float("nan")
    return out


def format_report(rep: dict) -> str:
    lines = [f"accuracy {rep['accuracy']:.4f}  hazard recall {rep['hazard_recall']:.4f}  "
             f"false alarms {rep['false_alarm_rate']:.4f}"]
    for name, m in rep["per_class"].items():
        lines.append(f"  {name:>8}  P {m['precision']:.3f}  R {m['recall']:.3f}  n={m['support']}")
    return "\n".join(lines)
