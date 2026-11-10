"""Model loading and the four inference stages.

Models are loaded once at process start and held in memory; a request only runs
forward passes. In demo mode nothing is loaded and the stages return synthetic
but structurally identical output, so the API and UI can be developed without a
GPU or checkpoints.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as C

_LOCK = threading.Lock()          # forward passes are serialised: one GPU, many requests
STATE: dict = {"s1": None, "s2": None, "s2_tok": None, "s2_idx2ans": None,
               "yolo": None, "s4": None, "cond": None, "device": "cpu", "loaded": False}
_ALLOWED_CACHE: dict = {}


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── startup ─────────────────────────────────────────────────────────────────
def load_models() -> dict:
    """Called once from the FastAPI lifespan hook.

    Every stage is optional — a missing checkpoint disables that stage. A
    *present but mismatched* checkpoint is a hard error, reported in
    /api/health, because a silently half-loaded model is far worse than an
    absent one: it answers every question with confident nonsense.
    """
    if C.DEMO_MODE:
        STATE.update(loaded=True, device="demo")
        return {"mode": "demo", "stages": [], "errors": {}}

    import torch
    import segmentation_models_pytorch as smp
    from transformers import AutoTokenizer

    from .models import (CrossModalVQA, FiLMUNet, GastroClassifier,
                         load_strict, read_answer_vocab)

    dev = C.DEVICE if (C.DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    STATE["device"] = dev
    loaded, errors, info = [], {}, {}

    if C.CKPT["s1"].exists():
        try:
            m = GastroClassifier(C.SWIN, len(C.CLASSES))
            load_strict(m, C.CKPT["s1"], "Stage 1 classifier")
            STATE["s1"] = m.to(dev).eval()
            loaded.append("classifier")
        except Exception as exc:
            errors["classifier"] = str(exc)

    if C.CKPT["s2"].exists():
        try:
            idx2ans = read_answer_vocab(C.CKPT["s2"], C.CKPT["s2_vocab"])
            m = CrossModalVQA(C.SWIN, C.TEXT_ENC, len(idx2ans))
            load_strict(m, C.CKPT["s2"], "Stage 2 VQA")
            STATE["s2"] = m.to(dev).eval()
            STATE["s2_tok"] = AutoTokenizer.from_pretrained(C.TEXT_ENC)
            STATE["s2_idx2ans"] = idx2ans
            info["answer_vocab_size"] = len(idx2ans)
            loaded.append("vqa")
        except Exception as exc:
            errors["vqa"] = str(exc)

    yolo_path = C.CKPT["yolo"] if C.CKPT["yolo"].exists() else None
    if yolo_path is None:                      # also accept the notebook's run layout
        hits = sorted(C.CKPT_DIR.glob("**/weights/best.pt"))
        yolo_path = hits[-1] if hits else None
    if yolo_path is not None:
        try:
            from ultralytics import YOLO
            STATE["yolo"] = YOLO(str(yolo_path))
            loaded.append("detector")
        except Exception as exc:
            errors["detector"] = str(exc)

    if C.CKPT["s4"].exists():
        try:
            m = smp.UnetPlusPlus(encoder_name=C.SEG_ENC, encoder_weights=None,
                                 in_channels=3, classes=1)
            load_strict(m, C.CKPT["s4"], "Stage 4 segmenter")
            STATE["s4"] = m.to(dev).eval()
            loaded.append("segmenter")
        except Exception as exc:
            errors["segmenter"] = str(exc)

    if C.USE_CONDITIONED and C.CKPT["cond"].exists():
        try:
            m = FiLMUNet(C.COND_ENC, len(C.LOCATION_TAGS))
            load_strict(m, C.CKPT["cond"], "Stage 4b conditioned segmenter")
            STATE["cond"] = m.to(dev).eval()
            loaded.append("conditioned-segmenter")
        except Exception as exc:
            errors["conditioned-segmenter"] = str(exc)

    STATE["loaded"] = True
    for k, v in errors.items():
        print(f"[clingastrovqa] ✗ {k}: {v}")
    return {"mode": "live", "device": dev, "stages": loaded,
            "errors": errors, "info": info}


def self_test() -> dict:
    """Sanity checks that catch a loaded-but-wrong model.

    A randomly initialised 5-class head produces near-uniform probabilities on
    any input. If the spread across classes is tiny, the checkpoint almost
    certainly did not apply, so say so at startup rather than shipping 25/21/19/
    18/16 into a findings sheet.
    """
    out = {}
    if C.DEMO_MODE or STATE["s1"] is None:
        return out
    rng = np.random.default_rng(0)
    probe = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    r = s1_classify(probe)
    spread = float(r["probs"].max() - r["probs"].min())
    out["classifier_probe_spread"] = round(spread, 4)
    out["classifier_probe_pred"] = r["pred"]
    if spread < 0.05:
        out["warning"] = ("classifier output is nearly uniform on a random probe — "
                          "the checkpoint may not have applied")
    if STATE["s2_idx2ans"]:
        out["answer_vocab_size"] = len(STATE["s2_idx2ans"])
        sample = [STATE["s2_idx2ans"][i] for i in sorted(STATE["s2_idx2ans"])[:3]]
        out["answer_vocab_sample"] = sample
        if len(STATE["s2_idx2ans"]) < 50:
            out["warning"] = (f"answer vocabulary has only {len(STATE['s2_idx2ans'])} "
                              "entries — it is being parsed at the wrong level")
    return out


def _transforms():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    cls = A.Compose([A.Resize(C.IMG_CLS, C.IMG_CLS), A.Normalize(C.MEAN, C.STD), ToTensorV2()])
    seg = A.Compose([A.Resize(C.IMG_SEG, C.IMG_SEG), A.Normalize(), ToTensorV2()])
    return cls, seg


# ── stages ──────────────────────────────────────────────────────────────────
def s1_classify(rgb):
    if STATE["s1"] is None:
        return None
    import torch
    import torch.nn.functional as F
    cls_tf, _ = _transforms()
    dev = STATE["device"]
    t = cls_tf(image=rgb)["image"].unsqueeze(0).to(dev)
    m = STATE["s1"]
    with torch.no_grad():
        probs = torch.softmax(m(t), 1).cpu().numpy().ravel()

    acts, grads = {}, {}
    h1 = m.backbone.norm.register_forward_hook(lambda md, i, o: acts.__setitem__("v", o))
    h2 = m.backbone.norm.register_full_backward_hook(lambda md, gi, go: grads.__setitem__("v", go[0]))
    m.zero_grad()
    out = m(t)
    out[0, int(out.argmax())].backward()
    h1.remove(); h2.remove()

    a, g = acts["v"].detach(), grads["v"].detach()
    if a.dim() == 4:
        cam = F.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(-1))[0]
    elif a.dim() == 3:
        c = F.relu((g.mean(dim=1, keepdim=True) * a).sum(-1))[0]
        sd = int(round(math.sqrt(c.shape[0])))
        cam = c[:sd * sd].reshape(sd, sd)
    else:
        cam = a.float().mean(0)
    cam = cam.float().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    return {"probs": probs, "pred": C.CLASSES[int(probs.argmax())],
            "conf": float(probs.max()), "cam": cam}


def allowed_idx(key: str) -> np.ndarray:
    """Vocabulary indices this question legally admits, per the dataset."""
    if key in _ALLOWED_CACHE:
        return _ALLOWED_CACHE[key]
    a2i = {v: k for k, v in (STATE["s2_idx2ans"] or {}).items()}
    idx = np.array(sorted({a2i[a] for a in C.Q_ANSWERS.get(key, []) if a in a2i}), dtype=np.int64)
    _ALLOWED_CACHE[key] = idx
    return idx


def s2_vqa(rgb, true_class: Optional[str] = None, restrict: Optional[bool] = None):
    """Answer every question in one batched pass. Each answer is restricted to
    that question's legal set, because the head shares one ~500-answer
    vocabulary and unrestricted argmax lets a count question return 'cecum'."""
    if STATE["s2"] is None:
        return None
    import torch
    restrict = C.RESTRICT_ANSWERS if restrict is None else restrict
    cls_tf, _ = _transforms()
    dev = STATE["device"]
    keys = list(C.QUESTION_KEYS)
    t = cls_tf(image=rgb)["image"].unsqueeze(0).to(dev).repeat(len(keys), 1, 1, 1)
    enc = STATE["s2_tok"]([C.QUESTION_TEXT[k] for k in keys], truncation=True,
                          padding="max_length", max_length=C.MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = STATE["s2"](t, enc["input_ids"].to(dev),
                             enc["attention_mask"].to(dev)).float().cpu().numpy()

    i2a = STATE["s2_idx2ans"]
    out = {}
    for r, k in enumerate(keys):
        z = logits[r]
        p = np.exp(z - z.max()); p /= p.sum()
        fi = int(p.argmax())
        allowed = allowed_idx(k) if restrict else np.arange(len(p))
        if len(allowed) == 0:
            allowed = np.arange(len(p))
        q = p[allowed] / max(p[allowed].sum(), 1e-12)
        ai = int(allowed[int(q.argmax())])
        out[k] = {"answer": i2a[ai], "confidence": float(q.max()),
                  "free_answer": i2a[fi], "free_confidence": float(p[fi]),
                  "restricted": bool(restrict and ai != fi),
                  "n_legal_answers": len(C.Q_ANSWERS.get(k, [])),
                  "in_coverage": C.q_in_coverage(k, true_class)}
    return out


def quadrant(cx, cy, w, h):
    return C.LOCATION_TAGS[min(2, int(3 * cy / h)) * 3 + min(2, int(3 * cx / w))]


def s3_detect(rgb, path):
    if STATE["yolo"] is None:
        return None
    r = STATE["yolo"].predict(str(path), conf=C.DET_CONF, verbose=False)[0]
    h, w = rgb.shape[:2]
    bx = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
    q = None
    if len(bx):
        b = bx[cf.argmax()]
        q = quadrant((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w, h)
    return {"boxes": bx, "confs": cf, "n": int(len(bx)),
            "top_conf": float(cf.max()) if len(cf) else 0.0, "quadrant": q}


def visible_mucosa(rgb, thr=15):
    return (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > thr).astype(np.uint8)


def size_bucket(cov):
    return "physical size not measured"


def s4_seg(rgb, locv=None):
    import torch
    _, seg_tf = _transforms()
    dev = STATE["device"]
    h, w = rgb.shape[:2]
    if STATE["cond"] is not None and locv is not None:
        t = seg_tf(image=rgb, mask=np.zeros(rgb.shape[:2], np.float32))["image"].unsqueeze(0).to(dev)
        c = torch.tensor(locv, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            pr = torch.sigmoid(STATE["cond"](t, c))[0, 0].cpu().numpy()
        used = "conditioned"
    elif STATE["s4"] is not None:
        t = seg_tf(image=rgb, mask=np.zeros(rgb.shape[:2], np.float32))["image"].unsqueeze(0).to(dev)
        with torch.no_grad():
            pr = torch.sigmoid(STATE["s4"](t))[0, 0].cpu().numpy()
        used = "unconditioned"
    else:
        return None
    pr = cv2.resize(pr, (w, h))
    mk = (pr >= 0.5).astype(np.uint8)
    cov = 100 * mk.sum() / max(visible_mucosa(rgb).sum(), 1)
    return {"prob": pr, "mask": mk, "coverage": float(cov),
            "size": size_bucket(cov), "model": used}


# ── demo stubs ──────────────────────────────────────────────────────────────
def _demo_stages(rgb, path):
    rng = np.random.default_rng(abs(hash(Path(path).stem)) % (2 ** 32))
    probs = rng.dirichlet(np.r_[1, 1, 1, 6, 1])
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.45, w * 0.55
    cam = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.22 * w) ** 2)))
    c1 = {"probs": probs, "pred": C.CLASSES[int(probs.argmax())],
          "conf": float(probs.max()), "cam": cam / cam.max()}

    demo_ans = {"procedure": "colonoscopy", "has_finding": "no", "n_findings": "1",
                "easy_detect": "yes", "abnormalities": "polyp",
                "location": "center; center-left", "color": "pink; red",
                "n_polyps": "1", "polyp_type": "paris ip", "polyp_size": "11-20mm",
                "polyps_removed": "not relevant", "instruments": "none",
                "n_instruments": "0", "instrument_location": "none",
                "landmarks": "none", "landmark_location": "none",
                "landmark_color": "none", "artefact_box": "no", "text": "yes"}
    c2 = {k: {"answer": v, "confidence": float(rng.uniform(0.45, 0.98)),
              "free_answer": v, "free_confidence": 0.5, "restricted": False,
              "n_legal_answers": len(C.Q_ANSWERS.get(k, [])) or None,
              "in_coverage": C.q_in_coverage(k, c1["pred"])}
          for k, v in demo_ans.items()}

    bw, bh = int(w * 0.26), int(h * 0.24)
    box = np.array([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]])
    c3 = {"boxes": box, "confs": np.array([0.78]), "n": 1, "top_conf": 0.78,
          "quadrant": quadrant(cx, cy, w, h)}

    mask = (cam > 0.55).astype(np.uint8)
    cov = 100 * mask.sum() / max(visible_mucosa(rgb).sum(), 1)
    c4 = {"prob": cam, "mask": mask, "coverage": float(cov),
          "size": size_bucket(cov), "model": "demo"}
    return c1, c2, c3, c4


# ── orchestration ───────────────────────────────────────────────────────────
def run(path, true_class: Optional[str] = None):
    """All four stages on one frame. Returns (rgb, c1, c2, c3, c4)."""
    rgb = load_rgb(path)
    with _LOCK:
        if C.DEMO_MODE:
            c1, c2, c3, c4 = _demo_stages(rgb, path)
            return rgb, c1, c2, c3, c4

        c1 = s1_classify(rgb)
        cls_hint = true_class or (c1["pred"] if c1 else None)
        c2 = s2_vqa(rgb, true_class=cls_hint)
        c3 = s3_detect(rgb, path)

        locv = None
        if STATE["cond"] is not None:
            locv = np.zeros(9, np.float32)
            if c3 and c3["quadrant"]:
                locv[C.LOCATION_TAGS.index(c3["quadrant"])] = 1
            elif c2:
                from .findings import tags
                for t in tags(c2.get("location", {}).get("answer", "")):
                    if t in C.LOCATION_TAGS:
                        locv[C.LOCATION_TAGS.index(t)] = 1
            if locv.sum() == 0:
                locv[4] = 1
        c4 = s4_seg(rgb, locv)
    return rgb, c1, c2, c3, c4
