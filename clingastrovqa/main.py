"""FastAPI service: upload a frame, run the pipeline, return a dashboard payload
and a downloadable findings sheet.

Endpoints
    GET  /                       single-page UI
    GET  /api/health             model status
    POST /api/analyze            multipart image → full result JSON
    GET  /api/runs/{id}          re-fetch a previous result
    GET  /api/runs/{id}/figure/{name}
    GET  /api/runs/{id}/report.pdf   timestamped download
    GET  /api/runs/{id}/report.html
    GET  /api/runs/{id}/findings.json
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config as C
from . import figures, pipeline, report
from .findings import fuse, summary_rows, vqa_rows

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_STATUS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load once, at startup — never per request.
    _STATUS.update(pipeline.load_models())
    _STATUS["question_bank"] = C.QUESTION_BANK_LOADED
    _STATUS["self_test"] = pipeline.self_test()
    print(f"[clingastrovqa] stages={_STATUS.get('stages')} "
          f"device={_STATUS.get('device')} self_test={_STATUS.get('self_test')}")
    if _STATUS.get("errors"):
        print("[clingastrovqa] STAGES FAILED TO LOAD — see /api/health")
    yield


app = FastAPI(title="ClinGastroVQA", version=C.PRODUCT_VERSION, lifespan=lifespan)


@app.get("/api/health")
def health():
    return {
        "status": "ok" if C.DEMO_MODE or _ready() else "not_ready",
        "mode": _STATUS.get("mode"),
        "device": _STATUS.get("device"),
        "stages_loaded": _STATUS.get("stages", []),
        "stages_failed": _STATUS.get("errors", {}),
        "self_test": _STATUS.get("self_test", {}),
        "answer_vocab_size": (_STATUS.get("info") or {}).get("answer_vocab_size"),
        "question_bank_loaded": C.QUESTION_BANK_LOADED,
        "answer_restriction": C.RESTRICT_ANSWERS,
        "product": C.REPORTED_BY,
        "server_time": C.stamp(),
    }


def _ready() -> bool:
    required = {"classifier", "vqa", "detector", "segmenter"}
    return (required.issubset(_STATUS.get("stages", []))
            and not _STATUS.get("errors") and C.QUESTION_BANK_LOADED)


def _run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(404, "run not found")
    d = (C.RUNS_DIR / run_id).resolve()
    if d.parent != C.RUNS_DIR.resolve() or not d.is_dir():
        raise HTTPException(404, "run not found")
    return d


def _payload(run_id: str) -> dict:
    return json.loads((_run_dir(run_id) / "result.json").read_text())


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not C.DEMO_MODE and not _ready():
        raise HTTPException(503, "Required models or question bank are unavailable; check /api/health.")
    suffix = Path(file.filename or "frame.jpg").suffix.lower()
    if suffix not in C.ALLOWED_SUFFIXES:
        raise HTTPException(415, f"unsupported file type '{suffix}'")

    blob = await file.read(C.MAX_UPLOAD_BYTES + 1)
    if not blob:
        raise HTTPException(400, "empty upload")
    if len(blob) > C.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {C.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    run_id = uuid.uuid4().hex
    outdir = C.RUNS_DIR / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    src = outdir / f"upload{suffix}"
    src.write_bytes(blob)

    generated_at = C.now()
    try:
        rgb, c1, c2, c3, c4 = await run_in_threadpool(pipeline.run, src)
    except ValueError as exc:
        shutil.rmtree(outdir, ignore_errors=True)
        raise HTTPException(400, str(exc))

    f = fuse(Path(file.filename or src.name).stem, c1, c2, c3, c4)
    figs = figures.render_all(rgb, c1, c2, c3, c4, f, outdir)

    case = {
        "case_id": run_id,
        "kind": "upload",
        "title": Path(file.filename or src.name).name,
        "image_path": str(src),
        "outdir": outdir,
        "findings": f,
        "seg_metrics": None,
        "generated_at": generated_at,
        "source_label": f"Uploaded frame · {Path(file.filename or src.name).name}",
        "figures": {
            "input": outdir / figs.get("input", ""),
            "gradcam": outdir / figs.get("gradcam", ""),
            "detection": outdir / figs.get("detection", ""),
            "segmentation": outdir / figs.get("segmentation", ""),
        },
    }
    rp = report.write_report(case, outdir)

    payload = {
        "run_id": run_id,
        "filename": Path(file.filename or src.name).name,
        "generated_at": generated_at.isoformat(),
        "generated_at_display": C.stamp(generated_at),
        "reported_by": C.REPORTED_BY,
        "model_stack": C.MODEL_STACK,
        "disclaimer": C.DISCLAIMER,
        "mode": _STATUS.get("mode"),
        "classification": {
            "predicted": f.predicted_class,
            "confidence": f.class_confidence,
            "probabilities": [{"label": k, "value": v}
                              for k, v in sorted(f.class_probs.items(),
                                                 key=lambda kv: -kv[1])],
        },
        "headline": {
            "finding": "not reported" if f.has_finding is None else ("yes" if f.has_finding else "no"),
            "finding_basis": f.finding_basis,
            "boxes": f.n_polyps_detected,
            "burden_pct": f.area_pct_mucosa,
            "size_bucket": f.size_category,
        },
        "summary": summary_rows(f),
        "vqa": vqa_rows(f),
        "vqa_groups": C.QUESTION_GROUPS,
        "out_of_coverage": f.out_of_coverage,
        "findings": f.to_dict(),
        "figures": {k: f"/api/runs/{run_id}/figure/{v}" for k, v in figs.items()},
        "report": {
            "pdf": f"/api/runs/{run_id}/report.pdf" if rp["pdf"] else None,
            "html": f"/api/runs/{run_id}/report.html",
            "json": f"/api/runs/{run_id}/findings.json",
            "download_stem": rp["download_stem"],
            "engine": rp["engine"],
            "error": rp["error"],
        },
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    (outdir / "findings.json").write_text(json.dumps(f.to_dict(), indent=2, default=str))
    return JSONResponse(payload)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return JSONResponse(_payload(run_id))


@app.get("/api/runs/{run_id}/figure/{name}")
def get_figure(run_id: str, name: str):
    if not re.fullmatch(r"[a-z_]+\.png", name):
        raise HTTPException(400, "bad figure name")
    p = _run_dir(run_id) / name
    if not p.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(p, media_type="image/png")


def _sheet(run_id: str, ext: str):
    d = _run_dir(run_id)
    hits = sorted(d.glob(f"*.{ext}"))
    if not hits:
        raise HTTPException(404, f"no {ext} report for this run")
    return hits[0]


@app.get("/api/runs/{run_id}/report.pdf")
def get_pdf(run_id: str):
    p = _sheet(run_id, "pdf")
    return FileResponse(p, media_type="application/pdf", filename=p.name)


@app.get("/api/runs/{run_id}/report.html")
def get_html(run_id: str):
    p = _sheet(run_id, "html")
    return FileResponse(p, media_type="text/html", filename=p.name)


@app.get("/api/runs/{run_id}/findings.json")
def get_json(run_id: str):
    p = _run_dir(run_id) / "findings.json"
    return FileResponse(p, media_type="application/json",
                        filename=f"{C.PRODUCT_NAME}_{run_id}_findings.json")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
