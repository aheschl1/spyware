"""Processing-worker settings, loaded from the environment / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingSettings(BaseSettings):
    """Worker cadence and retry policy, read from PROCESSING_* variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PROCESSING_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # How long a worker waits for a NOTIFY before running another
    # discover-and-drain pass; also the retry/discovery latency ceiling.
    poll_interval_seconds: float = 5.0
    discovery_batch: int = 100

    # Stamped onto jobs at enqueue (the table default is the backstop).
    max_attempts: int = 5
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_cap_seconds: float = 300.0

    # Supervisor: SIGTERM -> SIGKILL window and child-restart backoff.
    shutdown_grace_seconds: float = 20.0
    restart_backoff_base_seconds: float = 1.0
    restart_backoff_cap_seconds: float = 30.0
    restart_reset_seconds: float = 60.0

    # Speech detection (the VAD tier) — see processing/vad.py for the span
    # assembly these tune.
    vad_backend: str = "silero"  # or "energy": deterministic RMS threshold
    # Deliberately low: spans are a coarse, high-recall activity gate that
    # feeds the diarize tier's block assembly — no longer the ASR gate. On a
    # glasses mic silero scores far (non-wearer) speakers low, so precision
    # here would silence one side of every conversation.
    vad_threshold: float = 0.15
    vad_min_speech_ms: int = 250
    # Wide enough that ordinary conversational pauses don't split a span;
    # only a real lull does. Long pauses SHOULD split: silence wastes GPU
    # time and invites ASR hallucination.
    vad_merge_gap_ms: int = 2500
    vad_pad_ms: int = 200
    # The transcription model's input window: Canary/Whisper-family models
    # are trained on <=~40s clips, so spans must stay inside that envelope.
    # Forced cuts land on the quietest frame near the cap (processing/vad.py).
    vad_max_span_ms: int = 30_000
    vad_energy_threshold: int = 500  # int16 RMS amplitude, energy backend only

    # Transcription tier. The base URL speaks the standard transcriptions API
    # ("openai" protocol: POST {base}/audio/transcriptions) or a cog wrapper
    # ("cog": POST {base}/predictions) — swapping backends is env-only.
    transcriber_base_url: str = "http://127.0.0.1:8033/v1"
    transcriber_model: str = "nvidia/canary-qwen-2.5b"
    transcriber_protocol: str = "openai"  # or "cog"
    transcriber_timeout_seconds: float = 600.0

    # Diarization tier. Speech spans re-merge into blocks (contiguous speech,
    # gap-joined and capped) because diarization label consistency needs long
    # context — see docs/processing-pipelines.md.
    diarizer_base_url: str = "http://127.0.0.1:8034/v1"
    diarizer_timeout_seconds: float = 900.0  # a 30-min block is minutes of GPU
    diarize_block_merge_gap_ms: int = 30_000
    diarize_max_block_ms: int = 1_800_000
    diarize_min_turn_ms: int = 300
    # Utterances: same-speaker turns merged into the ASR units the transcribe
    # tier consumes. The cap is the transcription model's input window, same
    # rationale as vad_max_span_ms.
    diarize_utterance_merge_gap_ms: int = 1_500
    diarize_max_utterance_ms: int = 30_000

    # Speaker clustering tier: cosine distances over per-(block, speaker)
    # embeddings. Assign attaches an embedding to its nearest cluster; merge
    # (stricter) collapses clusters whose centroids converge. Embeddings from
    # under min_talk_ms of speech are skipped as unreliable voice-prints.
    # Calibrated on real glasses sessions (pyannote 3.1 / wespeaker): the
    # same voice spreads up to ~0.6 across blocks and sessions; known
    # different voices in one conversation sat at ~0.9.
    cluster_assign_max_distance: float = 0.60
    cluster_merge_max_distance: float = 0.50
    cluster_min_talk_ms: int = 3_000


@lru_cache(maxsize=1)
def get_settings() -> ProcessingSettings:
    return ProcessingSettings()
