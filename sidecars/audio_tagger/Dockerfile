# Serving container for CED audio tagging + CLAP embeddings (see app.py).
# Plain python base + CUDA-bundled torch wheels: the pip wheels ship their own
# CUDA libraries, so only the nvidia runtime (driver) is needed at run time.
# torchaudio is required by CED's remote feature-extraction code.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "torch>=2.4" "torchaudio>=2.4" \
        "transformers>=4.44" \
        "soundfile>=0.12" "soxr>=0.4" "numpy" \
        "fastapi>=0.110" "uvicorn>=0.30" "python-multipart>=0.0.9"

WORKDIR /srv
COPY app.py /srv/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
