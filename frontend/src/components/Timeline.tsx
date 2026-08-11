import { useCallback, useEffect, useState } from "react"
import { api, type TimelineEvent } from "../api/client"
import { fmtClock } from "../format"

const PAGE = 200

function eventKey(event: TimelineEvent, index: number): string {
  const artifact = "artifact_id" in event ? event.artifact_id : "session"
  return `${artifact}-${event.type}-${event.at_ms}-${index}`
}

// Renders the session's event stream. The union is open by contract —
// unknown event types from future tiers are silently skipped (default
// branch), never an error.
export default function Timeline({
  sessionId,
  onSeek,
}: {
  sessionId: string
  onSeek: (ms: number) => void
}) {
  const [events, setEvents] = useState<TimelineEvent[] | null>(null)
  const [hasMore, setHasMore] = useState(false)
  const [busy, setBusy] = useState(false)

  const load = useCallback(
    async (offset: number) => {
      setBusy(true)
      const { data } = await api.GET("/v1/sessions/{session_id}/timeline", {
        params: {
          path: { session_id: sessionId },
          query: { limit: PAGE, offset },
        },
      })
      setBusy(false)
      if (!data) return
      setHasMore(data.has_more)
      setEvents((prev) => (offset === 0 ? data.items : [...(prev ?? []), ...data.items]))
    },
    [sessionId],
  )

  useEffect(() => {
    void load(0)
  }, [load])

  if (!events) return <div className="loading">Loading timeline…</div>

  const rendered = events.map((event, index) => {
    switch (event.type) {
      case "transcript": {
        const speaker = event.speaker_name ?? event.speaker
        return (
          <div key={eventKey(event, index)} className="event transcript">
            <button className="event-time" onClick={() => onSeek(event.at_ms)}>
              {fmtClock(event.at_ms)}
            </button>
            <div className="event-body">
              {speaker && (
                <span className={`speaker ${event.speaker_name ? "" : "unresolved"}`}>
                  {speaker}
                </span>
              )}
              <span className="transcript-text">{event.text}</span>
            </div>
          </div>
        )
      }
      case "audio-tag": {
        if (event.labels.length === 0) return null
        return (
          <div key={eventKey(event, index)} className="event tags">
            <button className="event-time" onClick={() => onSeek(event.at_ms)}>
              {fmtClock(event.at_ms)}
            </button>
            <div className="event-body chips">
              {event.labels.slice(0, 5).map((tag) => (
                <span key={tag.label} className="chip" title={`score ${tag.score.toFixed(2)}`}>
                  {tag.label}
                </span>
              ))}
            </div>
          </div>
        )
      }
      case "session-start":
        return (
          <div key={eventKey(event, index)} className="event marker">
            session started
          </div>
        )
      case "session-end":
        return (
          <div key={eventKey(event, index)} className="event marker">
            session ended · {fmtClock(event.at_ms)}
          </div>
        )
      default:
        // speech-start/speech-end and any future event types: not rendered.
        return null
    }
  })

  return (
    <div className="timeline">
      {rendered.every((node) => node === null) ? (
        <div className="empty">Nothing on the timeline yet — pipelines may still be running.</div>
      ) : (
        rendered
      )}
      {hasMore && (
        <button className="btn ghost full" onClick={() => load(events.length)} disabled={busy}>
          {busy ? "loading…" : "more"}
        </button>
      )}
    </div>
  )
}
