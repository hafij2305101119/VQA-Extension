FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
ARG DEMO_ONLY=0
RUN if [ "$DEMO_ONLY" = "1" ]; then pip install --no-cache-dir -r requirements-demo.txt; else pip install --no-cache-dir -r requirements.txt; fi
COPY . .
RUN useradd --create-home --uid 1000 app && mkdir -p /app/runs /app/checkpoints && chown -R app:app /app
USER app
ENV CLINGASTRO_DEVICE=cpu PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn clingastrovqa.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
