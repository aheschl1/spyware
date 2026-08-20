"""Cache-aware streaming inference: one session per websocket connection.

The session owns the encoder cache tensors and decoder hypotheses for one
audio stream and turns incoming 16 kHz mono s16le PCM into growing partial
transcripts. Each stream step runs under the shared GPU lock; the caller is
responsible for threading (``asyncio.to_thread``).
"""

import logging
import threading

logger = logging.getLogger("asr.streaming")

SAMPLE_RATE = 16_000


def configure_streaming(model, right_context: int | None) -> None:
    """Pick the lookahead (att_context_size right frames) once at load time."""
    if right_context is not None:
        sizes = getattr(model.encoder, "att_context_size_all", None) or []
        chosen = next((s for s in sizes if s[1] == right_context), None)
        if chosen is None:
            raise ValueError(
                f"right context {right_context} not in model's att_context_size_all {sizes}"
            )
        model.encoder.set_default_att_context_size(chosen)
    model.encoder.setup_streaming_params()


class StreamingSession:
    def __init__(self, model, gpu_lock: threading.Lock) -> None:
        import numpy as np  # noqa: F401  (torch/numpy live in the sidecar image)
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        self._model = model
        self._lock = gpu_lock
        self._buffer = CacheAwareStreamingAudioBuffer(
            model=model, online_normalization=False
        )
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._stream_id = -1
        self._step = 0
        self._text = ""
        # Raw samples pending until a model-chunk-sized block accumulates:
        # the preprocessor runs per append, so tiny appends would put mel
        # boundary artifacts every 50 ms instead of once per chunk.
        size = model.encoder.streaming_cfg.chunk_size
        chunk_frames = size[1] if isinstance(size, (list, tuple)) else size
        self._block_samples = chunk_frames * 160  # 10 ms mel hop at 16 kHz
        self._raw = bytearray()

    @property
    def text(self) -> str:
        return self._text

    def feed(self, pcm: bytes) -> str | None:
        """Consume PCM; returns the updated transcript when a step ran."""
        with self._lock:
            self._raw += pcm
            block_bytes = self._block_samples * 2
            appended = False
            while len(self._raw) >= block_bytes:
                self._append(bytes(self._raw[:block_bytes]))
                del self._raw[:block_bytes]
                appended = True
            return self._decode_ready(final=False) if appended else None

    def finish(self, tail_silence_seconds: float = 0.5) -> str:
        """Flush the lookahead with trailing silence and return the final text."""
        with self._lock:
            tail = bytes(int(tail_silence_seconds * SAMPLE_RATE) * 2)
            self._append(bytes(self._raw) + tail)
            self._raw.clear()
            self._decode_ready(final=True)
        return self._text

    def _append(self, pcm: bytes) -> None:
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # NeMo's buffer returns -1 (not the new id) from the very first
        # append, and passing -1 again would open a second stream — this
        # session is always exactly stream 0.
        self._buffer.append_audio(samples, stream_id=self._stream_id)
        self._stream_id = 0

    def _next_chunk_frames(self) -> int:
        size = self._buffer.streaming_cfg.chunk_size
        if isinstance(size, (list, tuple)):
            return size[0] if self._buffer.buffer_idx == 0 else size[1]
        return size

    def _decode_ready(self, final: bool) -> str | None:
        """Run stream steps for every complete buffered chunk.

        The NeMo buffer happily yields partial chunks, and a chunk shorter
        than the pre-encoder's subsampling window crashes the encoder — so
        mid-stream we only pull once a full chunk is buffered, and the
        remainder is drained by ``finish``'s silence tail.
        """
        import torch

        updated = None
        while self._buffer.buffer is not None:
            available = self._buffer.buffer.size(-1) - self._buffer.buffer_idx
            if available <= 0 or (not final and available < self._next_chunk_frames()):
                break
            step = next(iter(self._buffer), None)
            if step is None:
                break
            chunk_audio, chunk_lengths = step
            # The first chunk carries no pre-encode cache, so nothing must be
            # dropped from its downsampled output (NeMo's own streaming
            # example does the same); dropping there loses the window's first
            # words and zeroes out small-lookahead chunks entirely.
            drop = 0 if self._step == 0 else self._model.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=final and self._buffer.is_buffer_empty(),
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            self._step += 1
            hyp = transcribed_texts[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            self._text = (text or "").strip()
            updated = self._text
        return updated
