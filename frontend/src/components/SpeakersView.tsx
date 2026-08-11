import { useEffect, useState } from "react"
import { api, type SpeakerRead, type SpeakerTranscriptRead } from "../api/client"
import { fmtClock, shortId } from "../format"

export default function SpeakersView({
  onOpen,
}: {
  onOpen: (id: string, seekMs?: number) => void
}) {
  const [speakers, setSpeakers] = useState<SpeakerRead[] | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState("")
  const [transcripts, setTranscripts] = useState<Record<string, SpeakerTranscriptRead[]>>({})

  const refresh = async () => {
    const { data } = await api.GET("/v1/speakers", {
      params: { query: { limit: 100, offset: 0 } },
    })
    if (data) setSpeakers(data.items)
  }
  useEffect(() => {
    void refresh()
  }, [])

  const rename = async (speaker: SpeakerRead) => {
    const name = draft.trim()
    const { data } = await api.POST("/v1/speakers/{speaker_id}/label", {
      params: { path: { speaker_id: speaker.id } },
      // Explicit null clears the label; the key is required by the API.
      body: { name: name === "" ? null : name },
    })
    setEditing(null)
    if (data) {
      setSpeakers((prev) =>
        (prev ?? []).map((s) => (s.id === data.id ? data : s)),
      )
    }
  }

  const toggle = async (speaker: SpeakerRead) => {
    const next = expanded === speaker.id ? null : speaker.id
    setExpanded(next)
    if (next && !transcripts[speaker.id]) {
      const { data } = await api.GET("/v1/speakers/{speaker_id}/transcripts", {
        params: { path: { speaker_id: speaker.id }, query: { limit: 50, offset: 0 } },
      })
      if (data) setTranscripts((prev) => ({ ...prev, [speaker.id]: data.items }))
    }
  }

  if (!speakers) return <div className="loading">Loading speakers…</div>
  if (speakers.length === 0)
    return <div className="empty">No speaker clusters yet — the clustering tier builds them.</div>

  return (
    <div className="list">
      {speakers.map((speaker) => (
        <div key={speaker.id} className="speaker-card">
          <div className="speaker-row">
            {editing === speaker.id ? (
              <form
                className="speaker-edit"
                onSubmit={(e) => {
                  e.preventDefault()
                  void rename(speaker)
                }}
              >
                <input
                  className="input slim"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder="name (empty clears)"
                  autoFocus
                />
                <button className="btn primary slim">save</button>
                <button
                  type="button"
                  className="btn ghost slim"
                  onClick={() => setEditing(null)}
                >
                  cancel
                </button>
              </form>
            ) : (
              <>
                <button className="speaker-name" onClick={() => toggle(speaker)}>
                  {speaker.name ?? <span className="row-dim">unlabeled · {shortId(speaker.id)}</span>}
                </button>
                <span className="row-sub">
                  {speaker.sessions} session{speaker.sessions === 1 ? "" : "s"} ·{" "}
                  {speaker.embeddings} voice-print{speaker.embeddings === 1 ? "" : "s"}
                </span>
                <button
                  className="btn ghost slim"
                  onClick={() => {
                    setEditing(speaker.id)
                    setDraft(speaker.name ?? "")
                  }}
                >
                  rename
                </button>
              </>
            )}
          </div>
          {expanded === speaker.id && (
            <div className="speaker-transcripts">
              {(transcripts[speaker.id] ?? []).length === 0 ? (
                <div className="empty">No transcripts.</div>
              ) : (
                (transcripts[speaker.id] ?? []).map((t) => (
                  <button
                    key={t.artifact_id}
                    className="row"
                    onClick={() => onOpen(t.session_id, t.start_ms)}
                  >
                    <span className="event-time as-span">{fmtClock(t.start_ms)}</span>
                    <span className="transcript-text">{t.text}</span>
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
