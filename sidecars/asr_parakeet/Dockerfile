# Serving container for parakeet-tdt-0.6b-v3 + faster-whisper large-v3 (see app.py).
# The NeMo NGC image ships torch + nemo_toolkit; we add the HTTP layer and
# faster-whisper (CTranslate2 runtime, own CUDA wheels).
FROM nvcr.io/nvidia/nemo:25.09

# websockets: uvicorn's ws protocol backend for /v1/audio/stream.
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.30" "python-multipart>=0.0.9" "faster-whisper>=1.1" "websockets>=12"

WORKDIR /srv
COPY app.py streaming.py /srv/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
