"""Model architectures — copied verbatim from the training notebook.

These definitions must match the notebook exactly, parameter name for parameter
name. A checkpoint is just a dict of tensors keyed by attribute path: rename
`self.head` to `self.classifier`, or nest a Linear differently, and every key
misses. `load_state_dict(..., strict=False)` will then load *nothing*, return
quietly, and leave a randomly initialised network that happily produces
confident-looking nonsense. Loading here is always strict, so that failure mode
surfaces at startup instead of in a report.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# timm and transformers are imported inside the constructors: reading the answer
# vocabulary must not require the whole deep-learning stack to be importable.


class GastroClassifier(nn.Module):
    """Stage 1 — Swin-T backbone with a two-layer head."""

    def __init__(self, backbone: str, num_classes: int, drop: float = 0.3, pretrained: bool = False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, drop_rate=drop)
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat), nn.Dropout(drop), nn.Linear(feat, 512),
            nn.GELU(), nn.Dropout(drop / 2), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


class CrossModalVQA(nn.Module):
    """Stage 2 — Swin-T image tokens attended by a DistilBERT question vector."""

    def __init__(self, backbone: str, text_model: str, num_answers: int,
                 fusion: int = 512, drop: float = 0.3, pretrained: bool = False):
        super().__init__()
        import timm
        from transformers import AutoModel
        self.image_encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        idim = self.image_encoder.num_features
        self.text_encoder = AutoModel.from_pretrained(text_model)
        tdim = self.text_encoder.config.hidden_size
        self.img_proj = nn.Sequential(nn.LayerNorm(idim), nn.Linear(idim, fusion))
        self.txt_proj = nn.Sequential(nn.LayerNorm(tdim), nn.Linear(tdim, fusion))
        self.cross_attn = nn.MultiheadAttention(fusion, 8, dropout=drop, batch_first=True)
        self.attn_norm = nn.LayerNorm(fusion)
        self.classifier = nn.Sequential(
            nn.Linear(fusion * 2, fusion), nn.GELU(), nn.Dropout(drop),
            nn.Linear(fusion, num_answers))

    def encode_image(self, x):
        f = self.image_encoder.forward_features(x)
        if f.dim() == 4:
            b, h, w, c = f.shape
            f = f.reshape(b, h * w, c)
        return f

    def forward(self, image, ids, mask):
        it = self.img_proj(self.encode_image(image))
        tc = self.txt_proj(self.text_encoder(input_ids=ids,
                                             attention_mask=mask).last_hidden_state[:, 0])
        att, _ = self.cross_attn(tc.unsqueeze(1), it, it)
        att = self.attn_norm(att.squeeze(1))
        return self.classifier(torch.cat([tc, att], -1))


class FiLM(nn.Module):
    def __init__(self, cd: int, n: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cd, 64), nn.ReLU(True), nn.Linear(64, 2 * n))
        self.n = n

    def forward(self, f, c):
        gb = self.net(c)
        g, b = gb[:, :self.n], gb[:, self.n:]
        return (1 + g.unsqueeze(-1).unsqueeze(-1)) * f + b.unsqueeze(-1).unsqueeze(-1)


class DecoderBlock(nn.Module):
    def __init__(self, ic: int, sc: int, oc: int, cd: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic + sc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
        self.film = FiLM(cd, oc)

    def forward(self, x, skip, c):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], 1)
        return self.film(self.conv(x), c)


class FiLMUNet(nn.Module):
    """Stage 4b — location-conditioned segmenter."""

    def __init__(self, enc: str, cd: int, dec=(256, 128, 64, 32, 16), pretrained: bool = False):
        super().__init__()
        import timm
        self.encoder = timm.create_model(enc, features_only=True,
                                         pretrained=pretrained, out_indices=(0, 1, 2, 3, 4))
        ch = self.encoder.feature_info.channels()
        self.center = nn.Sequential(
            nn.Conv2d(ch[4], ch[4], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[4]), nn.ReLU(True))
        self.d4 = DecoderBlock(ch[4], ch[3], dec[0], cd)
        self.d3 = DecoderBlock(dec[0], ch[2], dec[1], cd)
        self.d2 = DecoderBlock(dec[1], ch[1], dec[2], cd)
        self.d1 = DecoderBlock(dec[2], ch[0], dec[3], cd)
        self.d0 = DecoderBlock(dec[3], 0, dec[4], cd)
        self.head = nn.Conv2d(dec[4], 1, 1)

    def forward(self, x, c):
        h, w = x.shape[-2:]
        f0, f1, f2, f3, f4 = self.encoder(x)
        x = self.center(f4)
        x = self.d4(x, f3, c); x = self.d3(x, f2, c); x = self.d2(x, f1, c)
        x = self.d1(x, f0, c); x = self.d0(x, None, c); x = self.head(x)
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


def state_dict_of(obj) -> dict:
    """Every checkpoint the notebook writes is {'model_state': ..., ...}.
    Older or hand-made files may be a bare state dict; both are accepted, and
    anything else raises rather than being guessed at."""
    if isinstance(obj, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(hasattr(v, "shape") for v in obj.values()):
            return obj
    raise ValueError("unrecognised checkpoint layout: expected a dict containing "
                     "'model_state' (as written by the notebook) or a bare state dict")


def load_strict(model: nn.Module, ckpt_path, label: str) -> nn.Module:
    """Load a checkpoint and refuse to continue on any mismatch.

    Partial loads are the failure this whole module exists to prevent: the
    service would start, serve requests, and return uniform class probabilities
    from an untrained head with no error anywhere."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state_dict_of(raw)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"{label}: checkpoint does not match the architecture — "
            f"{len(missing)} missing and {len(unexpected)} unexpected parameters. "
            f"First missing: {list(missing)[:3]}. First unexpected: {list(unexpected)[:3]}. "
            "The class definition in models.py must match the notebook exactly.")
    return model


def read_answer_vocab(ckpt_path, vocab_path) -> dict:
    """Recover {index: answer}.

    The notebook writes {'ans2idx': {answer: index}} both inside the checkpoint
    and to answer_vocab.json. Reading that file and iterating it as a list gives
    the single key 'ans2idx' — which is exactly how every answer in a report
    became the string "ans2idx" with 100% confidence.
    """
    a2i = None
    if ckpt_path and Path(ckpt_path).exists():
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and isinstance(raw.get("ans2idx"), dict):
            a2i = raw["ans2idx"]
    if a2i is None and vocab_path and Path(vocab_path).exists():
        blob = json.loads(Path(vocab_path).read_text())
        if isinstance(blob, dict) and isinstance(blob.get("ans2idx"), dict):
            a2i = blob["ans2idx"]
        elif isinstance(blob, dict) and isinstance(blob.get("idx2ans"), dict):
            return {int(k): v for k, v in blob["idx2ans"].items()}
        elif isinstance(blob, list):
            return {i: str(a) for i, a in enumerate(blob)}
    if not a2i:
        raise ValueError("could not read the answer vocabulary: expected "
                         "{'ans2idx': {answer: index}} in the checkpoint or in "
                         "answer_vocab.json")
    if len(a2i) < 50:
        raise ValueError(f"answer vocabulary has only {len(a2i)} entries; the "
                         "trained model has ~500. The file is probably being "
                         "parsed at the wrong nesting level.")
    return {int(v): str(k) for k, v in a2i.items()}
