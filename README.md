# ClinGastroVQA report-generation application

Research prototype for uploading an endoscopy image, running classification, visual question answering, detection and segmentation, and downloading a structured HTML, JSON or PDF report.

Authors: Moinul Islam and Sadman Hafij, Department of Computer Science and Engineering and Health Informatics Research Lab, Daffodil International University, Bangladesh.

## Publication status

This repository contains the application prepared from the supplied research folder. Publishing it to GitHub does not create a hosted website. Hosting will be selected separately. GitHub Pages cannot run this Python API.

The web/report workflow is tested with explicitly synthetic input and outputs. The trained models have **not** been run or validated in this publication environment. Live inference readiness must be established before deploying a usable prediction service. Historical research scores are not deployment acceptance tests.

## Quick start: synthetic demonstration

Use Python 3.12 in a virtual environment, from this directory:

```sh
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements-demo.txt
```

PowerShell:

```powershell
$env:CLINGASTRO_DEMO="1"
python -m uvicorn clingastrovqa.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```sh
CLINGASTRO_DEMO=1 python -m uvicorn clingastrovqa.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Demo outputs are simulated and the downloaded reports are labelled **SYNTHETIC DEMO**. They are not model predictions.

## Model files

The five supplied checkpoints and the answer vocabulary are distributed once in the repository's `model-assets-v1` release as `checkpoints.7z`. They are not committed to Git history. The release archive is the original supplied archive, not newly trained models.

Download the release asset while signed in, extract its contents to `checkpoints/`, and verify:

```sh
python verify_checkpoints.py checkpoints
```

Expected files: `best_model.pth`, `best_vqa.pth`, `best.pt`, `best_seg.pth`, `conditioned.pth`, and `answer_vocab.json`. Each file's exact size and SHA-256 are in `checkpoint_manifest.json`. If an extractor creates an extra enclosing folder, move these six files into `checkpoints/`.

## Live inference setup (requires verification)

Install `requirements.txt`, supply verified checkpoints, and unset `CLINGASTRO_DEMO` or set it to `0`. Start the same Uvicorn command. Transformer model/tokenizer assets may download on first use. The ML dependency constraints were inherited from the supplied project and are **not a tested inference lockfile**.

`GET /api/health` must show classifier, VQA, detector and segmenter loaded, a populated question bank and no loading errors. Analysis returns HTTP 503 when required models are missing or fail to load. A health response is only a loading check, not evidence of model accuracy.

The supplied corrected loader and health checks have been integrated. It uses the model definitions in `clingastrovqa/models.py`, reads the nested answer vocabulary and rejects checkpoint key mismatches. The original application copy remains unchanged outside this repository.

The baseline U-Net++ segmenter is selected by default. Set `CLINGASTRO_CONDITIONED=1` only to explicitly test the experimental conditioned model; its added benefit has not been established.

## Containers

```sh
# Synthetic workflow only
docker build --build-arg DEMO_ONLY=1 -t clingastrovqa-demo .
docker run --rm -p 8000:8000 -e CLINGASTRO_DEMO=1 clingastrovqa-demo

# Live runtime candidate: must pass actual inference testing before hosting
docker build -t clingastrovqa .
docker run --rm -p 8000:8000 -v /absolute/path/checkpoints:/app/checkpoints:ro clingastrovqa
```

The Docker recipes are provided for the next hosting step; they were not built in the publication environment. The service uses one worker to avoid duplicating model memory. `PORT` defaults to `8000`. `CLINGASTRO_DEVICE=cpu` is the container default. GPU operation needs a suitable runtime and separate testing.

## Checks

```sh
python -m unittest -v test_app
python verify_checkpoints.py /path/to/checkpoints
```

The integration test exercises image upload, all 19 question rows, generated figures, HTML/PDF/JSON downloads, rejection of missing live models and suppression of unsupported answers from summary fields. It uses synthetic data and does not load neural networks.

## Remaining limitations before hosting

- Test all supplied real checkpoints and pin a compatible inference environment.
- Review report-fusion rules and disagreements between model stages. Unsupported answers remain visible in the raw question table for inspection, with coverage flags; they are suppressed from summary fields.
- VQA scores are constrained answer-set scores, not calibrated clinical certainty. A singleton legal answer can receive a score of one.
- Segmented area is an image-derived brightness-normalized ratio, not physical lesion size. Automatic area-to-millimetre labels have been removed.
- Model coverage, calibration, patient-independent testing and clinical usefulness remain unestablished.
- Add authentication, access control, storage retention, request limits and operational monitoring before hosting. Current run download URLs are bearer-like identifiers and are not an authorization system. Files persist in `runs/`; do not upload patient-identifying information to this prototype.

## Included material

Application Python modules, the dashboard, three supplied annotation/answer CSVs, dependency specifications, container recipes, tests, and checkpoint checksums. The manuscript, previous case reports, uploaded example images, caches and duplicated archive copies are excluded.

The dataset sources are HyperKvasir (https://doi.org/10.1038/s41597-020-00622-y), Kvasir-VQA (https://arxiv.org/abs/2409.01437) and Kvasir-SEG (https://arxiv.org/abs/1911.07069). Consult their terms and the model/dependency licenses before public redistribution. No new license or clinical approval is claimed here.
