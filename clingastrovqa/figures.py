"""Per-stage figures, written as individual PNGs into the run folder.

Matplotlib is used head-first with the Agg backend — no display, thread-safe as
long as one figure is built at a time, which the pipeline lock already ensures.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

GROUP_COLOUR = {"Procedure & overview": "#4C72B0", "Abnormality": "#C44E52",
                "Polyps": "#55A868", "Instruments": "#DD8452",
                "Anatomical landmarks": "#8172B2", "Image quality": "#937860"}


def _save(fig, path: Path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def overlay(rgb, mask, colour=(255, 0, 0), alpha=0.45):
    ov = rgb.copy()
    m = mask.astype(bool)
    ov[m] = ((1 - alpha) * ov[m] + alpha * np.array(colour)).astype(np.uint8)
    return ov


def class_bar(ax, probs, highlight=None, fontsize=9):
    """Names are drawn inside long bars and beside short ones; there are no
    y-tick labels, so a long class name cannot spill into a neighbouring axes."""
    order = np.argsort(probs)[::-1]
    names = [C.CLASSES[i] for i in order]
    vals = [100.0 * probs[i] for i in order]
    y = np.arange(len(order))
    cols = ["#2CA02C" if n == highlight else "#4C72B0" for n in names]
    ax.barh(y, vals, color=cols, edgecolor="white", height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    ax.invert_yaxis(); ax.set_xlim(0, 124); ax.set_xticks([0, 20, 40, 60, 80, 100])
    for k, (n, v) in enumerate(zip(names, vals)):
        if v >= 34:
            ax.text(2.5, k, n, va="center", fontsize=fontsize, color="white", fontweight="bold")
        else:
            ax.text(v + 2.5, k, n, va="center", fontsize=fontsize, color="#333")
        ax.text(123, k, f"{v:.1f}%", va="center", ha="right", fontsize=fontsize - 0.5, color="#555")
    ax.set_xlabel("probability (%)")
    ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def render_all(rgb, c1, c2, c3, c4, f, outdir: Path) -> dict:
    """Write every panel as its own PNG plus a combined dashboard image."""
    outdir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, str] = {}

    cv2.imwrite(str(outdir / "input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    figs["input"] = "input.png"

    if c1 is not None:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        class_bar(ax, c1["probs"], highlight=c1["pred"])
        ax.set_title(f"Stage 1 — {c1['pred']} ({100 * c1['conf']:.1f}%)",
                     fontweight="bold", fontsize=11)
        fig.tight_layout()
        _save(fig, outdir / "class_probabilities.png")
        figs["class_probabilities"] = "class_probabilities.png"

        hm = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * c1["cam"]), cv2.COLORMAP_JET),
                          cv2.COLOR_BGR2RGB)
        cam = cv2.addWeighted(rgb, 0.55, hm, 0.45, 0)
        cv2.imwrite(str(outdir / "gradcam.png"), cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
        figs["gradcam"] = "gradcam.png"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb); ax.axis("off")
    n = c3["n"] if c3 else 0
    if n:
        for b, cf in zip(c3["boxes"], c3["confs"]):
            x0, y0, x1, y1 = np.asarray(b).astype(int)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="#2ca02c", lw=2.5))
            ax.text(x0, max(y0 - 6, 10), f"{cf:.2f}", color="#2ca02c",
                    fontweight="bold", fontsize=9)
    ax.set_title(f"Stage 3 — detection ({n} box{'es' if n != 1 else ''})",
                 fontweight="bold", fontsize=10)
    _save(fig, outdir / "detection.png")
    figs["detection"] = "detection.png"

    if c4 is not None:
        cv2.imwrite(str(outdir / "segmentation.png"),
                    cv2.cvtColor(overlay(rgb, c4["mask"]), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outdir / "mask.png"), (c4["mask"] * 255).astype(np.uint8))
        figs["segmentation"] = "segmentation.png"
        figs["mask"] = "mask.png"

    if f.vqa:
        from .findings import vqa_rows
        rows = vqa_rows(f)
        fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.4))
        y = np.arange(len(rows))
        ax.barh(y, [100 * r["confidence"] for r in rows],
                color=[GROUP_COLOUR.get(r["group"], "#4C72B0") for r in rows],
                edgecolor="white", height=0.66)
        ax.set_yticks(y); ax.set_yticklabels([r["key"] for r in rows], fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlim(0, 124)
        for i, r in enumerate(rows):
            v = 100 * r["confidence"]
            txt = str(r["answer"])[:32] + (" *" if r["restriction_changed_answer"] else "")
            if v >= 36:
                ax.text(2, i, txt, va="center", fontsize=8, color="white", fontweight="bold")
            else:
                ax.text(v + 2, i, txt, va="center", fontsize=8, color="#333")
            ax.text(123, i, f"{v:.0f}%", va="center", ha="right", fontsize=7.5, color="#666")
            if r["trained_on_this_class"] is False:
                ax.text(-1, i, "!", va="center", ha="right", fontsize=9,
                        color="#C44E52", fontweight="bold")
        ax.set_xlabel("answer confidence (%)")
        ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"Stage 2 — VQA · {len(rows)} questions", fontweight="bold", fontsize=11)
        fig.tight_layout()
        _save(fig, outdir / "vqa_confidence.png")
        figs["vqa_confidence"] = "vqa_confidence.png"

    return figs
