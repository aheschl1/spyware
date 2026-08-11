import { useEffect, useRef, useState } from "react"
import { api, type SessionRead } from "../api/client"
import { fmtDate, fmtDuration, shortId } from "../format"
import AudioPlayer from "./AudioPlayer"
import TagSummary from "./TagSummary"
import Timeline from "./Timeline"

export default function SessionView({
  sessionId,
  seekMs,
  onBack,
}: {
  sessionId: string
  seekMs?: number
  onBack: () => void
  onOpenSession: (id: string, seekMs?: number) => void
}) {
  const [session, setSession] = useState<SessionRead | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const pendingSeek = useRef<number | undefined>(seekMs)

  useEffect(() => {
    void api
      .GET("/v1/sessions/{session_id}", {
        params: { path: { session_id: sessionId } },
      })
      .then(({ data }) => {
        if (data) setSession(data)
      })
  }, [sessionId])

  const seekTo = (ms: number) => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.readyState === 0) {
      pendingSeek.current = ms
      return
    }
    audio.currentTime = ms / 1000
    void audio.play().catch(() => {})
  }

  // A seek requested before the element had metadata (deep link from search)
  // applies once the audio is ready.
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const applyPending = () => {
      if (pendingSeek.current !== undefined) {
        audio.currentTime = pendingSeek.current / 1000
        pendingSeek.current = undefined
        void audio.play().catch(() => {})
      }
    }
    audio.addEventListener("loadedmetadata", applyPending)
    return () => audio.removeEventListener("loadedmetadata", applyPending)
  }, [session])

  if (!session) return <div className="loading">Loading session…</div>

  return (
    <div className="session-view">
      <div className="session-header">
        <button className="btn ghost slim" onClick={onBack}>
          ← back
        </button>
        <div>
          <h2 className="session-title">
            {session.label ?? session.device ?? shortId(session.id)}
            {session.is_open && <span className="badge live">live</span>}
          </h2>
          <div className="session-sub">
            {fmtDate(session.started_at)} · {fmtDuration(session.started_at, session.ended_at)}
            {session.device && ` · ${session.device}`}
          </div>
        </div>
      </div>
      <AudioPlayer sessionId={sessionId} audioRef={audioRef} />
      <TagSummary sessionId={sessionId} />
      <Timeline sessionId={sessionId} onSeek={seekTo} />
    </div>
  )
}
