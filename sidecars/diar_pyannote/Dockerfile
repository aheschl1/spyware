# Serving container for pyannote speaker diarization (see app.py).
# Plain python base + CUDA-bundled torch wheels: the pip wheels ship their own
# CUDA libraries, so only the nvidia runtime (driver) is needed at run time,
# and pip resolves torch/torchaudio/pyannote versions together cleanly.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Exact pins: app.py depends on the pyannote 4.x output API and private
# pipeline attrs; a rebuild must not silently resolve to something else.
RUN pip install --no-cache-dir \
        "torch==2.13.0" "torchaudio==2.11.0" \
        "pyannote.audio==4.0.7" \
        "fastapi==0.141.1" "uvicorn==0.52.1" "python-multipart==0.0.32"

WORKDIR /srv
COPY app.py /srv/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
