# Serving container for DiariZen speaker diarization (see app.py).
# Plain python base + CUDA-bundled torch wheels: the pip wheels ship their own
# CUDA libraries, so only the nvidia runtime (driver) is needed at run time.
#
# DiariZen pins python 3.10 and torch 2.1.1, and vendors its own pyannote.audio
# fork in-tree — hence the git clone rather than a pip pin, and hence the older
# torch than the rest of the stack. Both are contained in this image.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pinned to a commit: DiariZen publishes no releases, and app.py depends on the
# DiariZenPipeline inference API. A rebuild must not drift.
ARG DIARIZEN_REV=844f5555b0a98acd0931511fc641a8c5b8ba92c7

# The stock pip here is too old for some current wheel metadata ("inconsistent
# Name: expected 'typing-extensions'").
RUN pip install --no-cache-dir --upgrade pip

# torch 2.1.x ships CUDA 12.1 in its default PyPI wheel, so no extra index is
# needed — and --index-url would replace PyPI outright, leaving build-time deps
# like flit_core unresolvable. Installed first so the constrained requirements
# install below sees it as already satisfied. torchvision is in DiariZen's
# constraints but nothing in the inference path imports it.
RUN pip install --no-cache-dir torch==2.1.1 torchaudio==2.1.1

# requirements.txt is otherwise installed whole: the package declares no
# runtime deps of its own, so trimming further risks a missing transitive
# import for the sake of image size. pesq and pystoi are the exception —
# speech-enhancement metrics that nothing on the inference path imports, and
# pesq is a C extension that would drag a whole toolchain into the image.
# dscore (the only submodule) is scoring-only and deliberately not fetched.
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
