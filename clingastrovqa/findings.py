"""Structured findings record — the single source of truth for the dashboard,
the JSON API and the printed sheet. Ported verbatim from the notebook's fusion
cell so the web output and the research output cannot diverge."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import config as C

_WORDNUM = {"none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_EMPTY = ("none", "nan", "", "n/a", "not relevant")


def tags(a: Any) -> List[str]:
    s = str(a).strip().lower()
    return [] if s in _EMPTY else [x.strip() for x in s.split(";") if x.strip()]


def yn(a: Any) -> Optional[bool]:
    """Tri-state: True, False, or None for 'not relevant' and anything else."""
    s = str(a).strip().lower()
    if s in ("yes", "true", "1", "y"):
        return True
    if s in ("no", "false", "0", "n"):
        return False
    return None


def num(a: Any) -> Optional[int]:
    s = str(a).strip().lower()
    if s in _WORDNUM:
        return _WORDNUM[s]
    try:
        return int(float(s.split()[0]))
    except Exception:
        return None


def plain(a: Any) -> Optional[str]:
    s = str(a).strip()
    return None if s.lower() in ("none", "nan", "", "n/a") else s


def fmt(v: Any, empty: str = "not reported") -> str:
    if v is None:
        return empty
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return "; ".join(map(str, v)) if len(v) else empty
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


@dataclass
class Findings:
    image_id: str = ""
    true_class: str = ""
    predicted_class: str = ""
    class_confidence: float = 0.0
    correct: Optional[bool] = None
    class_probs: Dict[str, float] = field(default_factory=dict)

    procedure: Optional[str] = None
    has_finding: Optional[bool] = None
    finding_basis: str = ""
    n_findings: Optional[int] = None
    easy_to_detect: Optional[bool] = None

    abnormalities: List[str] = field(default_factory=list)
    location: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)

    n_polyps_vqa: Optional[int] = None
    polyp_type: Optional[str] = None
    polyp_size_vqa: Optional[str] = None
    polyps_removed: Optional[bool] = None
    polyps_removed_raw: Optional[str] = None

    instruments: List[str] = field(default_factory=list)
    n_instruments: Optional[int] = None
    instrument_location: List[str] = field(default_factory=list)

    landmarks: List[str] = field(default_factory=list)
    landmark_location: List[str] = field(default_factory=list)
    landmark_colors: List[str] = field(default_factory=list)

    box_artefact: Optional[bool] = None
    text_present: Optional[bool] = None

    n_polyps_detected: int = 0
    detection_confidence: float = 0.0
    detection_quadrant: Optional[str] = None
    boxes: List[List[float]] = field(default_factory=list)

    area_pct_mucosa: Optional[float] = None
    size_category: Optional[str] = None
    seg_model: Optional[str] = None

    vqa: Dict[str, dict] = field(default_factory=dict)
    out_of_coverage: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def answer(c2: Optional[dict], key: str, default=None):
    if not c2 or key not in c2:
        return default
    v = c2[key]
    if isinstance(v, dict) and v.get("in_coverage") is False:
        return default
    return v["answer"] if isinstance(v, dict) else v[0]


def fuse(image_id: str, c1, c2, c3, c4, true_class: str = "") -> Findings:
    f = Findings(image_id=image_id, true_class=true_class)

    if c1:
        f.predicted_class = c1["pred"]
        f.class_confidence = float(c1["conf"])
        f.class_probs = {C.CLASSES[i]: float(p) for i, p in enumerate(c1["probs"])}
        if true_class:
            f.correct = c1["pred"] == true_class

    if c2:
        g = lambda k: answer(c2, k)  # noqa: E731
        f.vqa = dict(c2)
        f.out_of_coverage = [k for k, v in c2.items() if v.get("in_coverage") is False]
        f.procedure = plain(g("procedure"))
        f.abnormalities = tags(g("abnormalities"))
        f.location = tags(g("location"))
        f.colors = tags(g("color"))
        f.n_findings = num(g("n_findings"))
        f.easy_to_detect = yn(g("easy_detect"))

        # "Does this image contain any finding?" is asked only of Normal frames in
        # the training metadata, where the answer is always "no". It carries no
        # signal on any other class, so finding status is driven by the
        # abnormality answer and the question is kept only as corroboration.
        direct = yn(g("has_finding"))
        if f.abnormalities:
            f.has_finding, f.finding_basis = True, "abnormality answer"
        elif (f.n_findings or 0) > 0:
            f.has_finding, f.finding_basis = True, "finding count"
        elif direct is not None and "has_finding" not in f.out_of_coverage:
            f.has_finding, f.finding_basis = bool(direct), "finding question"
        else:
            f.has_finding, f.finding_basis = None, "no supported finding answer"

        f.n_polyps_vqa = num(g("n_polyps"))
        f.polyp_type = plain(g("polyp_type"))
        f.polyp_size_vqa = plain(g("polyp_size"))
        f.polyps_removed = yn(g("polyps_removed"))
        f.polyps_removed_raw = plain(g("polyps_removed"))
        f.instruments = tags(g("instruments"))
        f.n_instruments = num(g("n_instruments"))
        f.instrument_location = tags(g("instrument_location"))
        f.landmarks = tags(g("landmarks"))
        f.landmark_location = tags(g("landmark_location"))
        f.landmark_colors = tags(g("landmark_color"))
        f.box_artefact = yn(g("artefact_box"))
        f.text_present = yn(g("text"))

    if c3:
        f.n_polyps_detected = int(c3["n"])
        f.detection_confidence = float(c3["top_conf"])
        f.detection_quadrant = c3["quadrant"]
        f.boxes = [[float(x) for x in b] for b in c3["boxes"]]

    if c4:
        f.area_pct_mucosa = float(c4["coverage"])
        f.size_category = c4["size"]
        f.seg_model = c4["model"]

    return f


def vqa_rows(f: Findings) -> List[dict]:
    """One row per asked question, in reporting-group order."""
    rows = []
    for k in C.QUESTION_KEYS:
        if k not in f.vqa:
            continue
        v = f.vqa[k]
        rows.append({
            "group": C.QUESTION_GROUP[k],
            "key": k,
            "question": C.QUESTION_TEXT[k],
            "answer": v["answer"],
            "confidence": round(float(v["confidence"]), 4),
            "legal_answers": v.get("n_legal_answers"),
            "unrestricted_answer": v.get("free_answer"),
            "restriction_changed_answer": v.get("restricted"),
            "trained_on_this_class": v.get("in_coverage"),
        })
    return rows


def summary_rows(f: Findings) -> List[dict]:
    """Flat field/value view — what the dashboard's findings table renders."""
    pairs = [
        ("Predicted class", f.predicted_class),
        ("Class confidence", f"{100 * f.class_confidence:.1f}%"),
        ("Procedure", f.procedure),
        ("Finding present", f"{fmt(f.has_finding)} — {f.finding_basis}"),
        ("Findings counted", fmt(f.n_findings)),
        ("Easy to detect", fmt(f.easy_to_detect)),
        ("Abnormalities", fmt(f.abnormalities, "none")),
        ("Abnormality region", fmt(f.location, "n/a")),
        ("Abnormality colour", fmt(f.colors, "n/a")),
        ("Polyps (VQA count)", fmt(f.n_polyps_vqa)),
        ("Polyp type", fmt(f.polyp_type, "none")),
        ("Polyp size (VQA)", fmt(f.polyp_size_vqa)),
        ("All polyps removed", fmt(f.polyps_removed_raw)),
        ("Instruments", fmt(f.instruments, "none")),
        ("Instrument count", fmt(f.n_instruments)),
        ("Instrument region", fmt(f.instrument_location, "n/a")),
        ("Landmarks", fmt(f.landmarks, "none")),
        ("Landmark region", fmt(f.landmark_location, "n/a")),
        ("Landmark colour", fmt(f.landmark_colors, "n/a")),
        ("Box artefact", fmt(f.box_artefact)),
        ("Text overlay", fmt(f.text_present)),
        ("Detector boxes", str(f.n_polyps_detected)),
        ("Top box confidence", f"{f.detection_confidence:.2f}"),
        ("Box quadrant", fmt(f.detection_quadrant, "n/a")),
        ("Lesion burden", "n/a" if f.area_pct_mucosa is None
         else f"{f.area_pct_mucosa:.2f}% of visible mucosa"),
        ("Size bucket", fmt(f.size_category)),
        ("Segmenter", fmt(f.seg_model)),
    ]
    return [{"field": k, "value": v} for k, v in pairs]
