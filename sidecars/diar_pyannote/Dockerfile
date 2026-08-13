# Serving container for DiariZen speaker diarization (see app.py).
# CUDA-bundled torch wheels: only the nvidia runtime (driver) is needed.
# DiariZen pins python 3.10 and torch 2.1.1 and vendors its own pyannote.audio
# fork in-tree — hence the git clone rather than a pip pin (README: build notes).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# DiariZen publishes no releases; a rebuild must not drift.
ARG DIARIZEN_REV=844f5555b0a98acd0931511fc641a8c5b8ba92c7

# The stock pip is too old for some current wheel metadata.
RUN pip install --no-cache-dir --upgrade pip

# torch 2.1.x bundles CUDA 12.1 in its default PyPI wheel; installed first so
# the constrained install below sees it satisfied.
RUN pip install --no-cache-dir torch==2.1.1 torchaudio==2.1.1

# requirements.txt installs whole (the package declares no runtime deps of its
# own) minus pesq/pystoi: scoring-only, and pesq needs a C toolchain.
RUN git clone https://github.com/BUTSpeechFIT/DiariZen.git /opt/diarizen \
    && cd /opt/diarizen \
    && git checkout ${DIARIZEN_REV} \
    && grep -viE '^(pesq|pystoi)\b' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt -c constraints.txt \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -e ./pyannote-audio -c constraints.txt

RUN pip install --no-cache-dir \
        "fastapi==0.141.1" "uvicorn==0.52.1" "python-multipart==0.0.32"

WORKDIR /srv
COPY app.py /srv/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
