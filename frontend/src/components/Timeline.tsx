import { useCallback, useEffect, useRef, useState } from "react"
import { api, type TimelineEvent } from "../api/client"
import { fmtClock } from "../format"
import TimelineFilter, { loadHidden, type EventKind } from "./TimelineFilter"

const PAGE = 200
// Deep links load the timeline from a little before the target moment, so
// the interesting event is on the first page even in a long session.
const FOCUS_LEAD_MS = 15_000

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
  focusMs,
}: {
  sessionId: string
  onSeek: (ms: number) => void
  focusMs?: number
}) {
  const [events, setEvents] = useState<TimelineEvent[] | null>(null)
  const [hasMore, setHasMore] = useState(false)
  const [busy, setBusy] = useState(false)
  const [fromMs, setFromMs] = useState<number | null>(
    focusMs !== undefined ? Math.max(0, focusMs - FOCUS_LEAD_MS) : null,
  )
  const [flashKey, setFlashKey] = useState<string | null>(null)
  const flashed = useRef(false)
  const [hidden, setHidden] = useState<Set<EventKind>>(loadHidden)

  const load = useCallback(
    async (offset: number, windowStart: number | null) => {
      setBusy(true)
      const { data } = await api.GET("/v1/sessions/{session_id}/timeline", {
        params: {
          path: { session_id: sessionId },
          query: {
            limit: PAGE,
            offset,
            ...(windowStart !== null ? { from_ms: windowStart } : {}),
          },
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
    void load(0, fromMs)
  }, [load, fromMs])

  // Flash-and-scroll the first substantive event at/after the focus moment,
  // once per deep link.
  useEffect(() => {
    if (focusMs === undefined || flashed.current || !events) return
    const index = events.findIndex(
      (event) =>
        (event.type === "transcript" || event.type === "audio-tag") &&
        event.at_ms >= focusMs,
    )
    if (index >= 0) {
      flashed.current = true
      setFlashKey(eventKey(events[index]!, index))
    }
  }, [events, focusMs])

  const flashRef = useCallback((node: HTMLDivElement | null) => {
    node?.scrollIntoView({ block: "center", behavior: "smooth" })
  }, [])

  if (!events)
    return (
      <div className="list">
        {[36, 52, 36, 68, 36].map((height, i) => (
          <div key={i} className="skeleton" style={{ height }} />
        ))}
      </div>
    )

  const rendered = events.map((event, index) => {
    const key = eventKey(event, index)
    const flash = key === flashKey
    switch (event.type) {
      case "transcript": {
        if (hidden.has("transcript")) return null
        const speaker = event.speaker_name ?? event.speaker
        return (
          <div
            key={key}
            ref={flash ? flashRef : undefined}
            className={`event transcript ${flash ? "flash" : ""}`}
          >
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
        if (hidden.has("audio-tag") || event.labels.length === 0) return null
        return (
          <div
            key={key}
            ref={flash ? flashRef : undefined}
            className={`event tags ${flash ? "flash" : ""}`}
          >
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
        if (hidden.has("marker")) return null
        return (
          <div key={key} className="event marker">
            session started
          </div>
        )
      case "session-end":
        if (hidden.has("marker")) return null
        return (
          <div key={key} className="event marker">
            session ended · {fmtClock(event.at_ms)}
          </div>
        )
      default:
        // speech-start/speech-end and any future event types: not rendered.
        return null
    }
  })

  const allFilteredOut =
    rendered.every((node) => node === null) && hidden.size > 0 && events.length > 0

  return (
    <>
      <div className="timeline-toolbar">
        <TimelineFilter hidden={hidden} onChange={setHidden} />
      </div>
      <div className="timeline">
        {fromMs !== null && fromMs > 0 && (
          <button
            className="btn ghost full"
            onClick={() => {
              flashed.current = true // keep the view where the user is
              setFlashKey(null)
              setEvents(null)
              setFromMs(null)
            }}
          >
            ↑ show from the beginning (jumped to {fmtClock(fromMs + FOCUS_LEAD_MS)})
          </button>
        )}
        {rendered.every((node) => node === null) ? (
          <div className="empty">
            {allFilteredOut
              ? "Everything here is hidden by the filter."
              : "Nothing on the timeline yet — pipelines may still be running."}
          </div>
        ) : (
          rendered
        )}
        {hasMore && (
          <button
            className="btn ghost full"
            onClick={() => load(events.length, fromMs)}
            disabled={busy}
          >
            {busy ? "loading…" : "more"}
          </button>
        )}
      </div>
    </>
  )
}
