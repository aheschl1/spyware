import { useEffect, useRef, useState } from "react"
import {
  api,
  type SimilarSpeakerRead,
  type SpeakerRead,
  type SpeakerTranscriptRead,
} from "../api/client"
import { fmtClock, shortId } from "../format"
import ClipButton, { ClipProgress } from "./ClipButton"

// Centroids of one voice spread up to ~0.6; different voices sit near ~0.9.
// At or past this, the confirm step warns that the merge looks cross-voice.
const FAR_DISTANCE = 0.85

function initials(name: string | null | undefined, id: string): string {
  if (!name) return id.slice(0, 2)
  const parts = name.trim().split(/\s+/)
  return ((parts[0]?.[0] ?? "") + (parts[1]?.[0] ?? "")).toUpperCase() || id.slice(0, 2)
}

// Deterministic accent per speaker so the same cluster always wears the
// same colour, labeled or not.
function hue(id: string): number {
  let h = 0
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 360
  return h
}

function speakerLabel(speaker: { name?: string | null; id: string }): string {
  return speaker.name ?? shortId(speaker.id)
}

// The merge popover for one card: candidates ranked by centroid distance,
// then an inline confirm (with a warning when the pair looks cross-voice).
function MergeMenu({
  speaker,
  busy,
  onMerge,
}: {
  speaker: SpeakerRead
  busy: boolean
  onMerge: (survivor: SpeakerRead) => void
}) {
  const [open, setOpen] = useState(false)
  const [candidates, setCandidates] = useState<SimilarSpeakerRead[] | null>(null)
  const [target, setTarget] = useState<SimilarSpeakerRead | null>(null)
  const root = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const onOutside = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false)
    }
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false)
    }
    document.addEventListener("mousedown", onOutside)
    document.addEventListener("keydown", onEscape)
    return () => {
      document.removeEventListener("mousedown", onOutside)
      document.removeEventListener("keydown", onEscape)
    }
  }, [open])

  const toggle = async () => {
    const next = !open
    setOpen(next)
    setTarget(null)
    if (next) {
      setCandidates(null)
      const { data } = await api.GET("/v1/speakers/{speaker_id}/similar", {
        params: { path: { speaker_id: speaker.id } },
      })
      if (data) setCandidates(data.items)
    }
  }

  return (
    <div className="filter-menu" ref={root}>
      <button className="btn ghost slim" onClick={() => void toggle()}>
        merge
      </button>
      {open && (
        <div className="filter-pop merge-pop">
          {target ? (
            <div className="merge-confirm">
              <div>
                merge {speaker.embeddings} voice-print
                {speaker.embeddings === 1 ? "" : "s"} into{" "}
                <strong>{speakerLabel(target)}</strong>?
              </div>
              {target.distance >= FAR_DISTANCE && (
                <div className="merge-warning">
                  these clusters are {target.distance.toFixed(2)} apart — that's
                  typical of two different voices; merging will pull the
                  voice-print center between them
                </div>
              )}
              <div className="merge-actions">
                <button
                  className="btn primary slim"
                  disabled={busy}
                  onClick={() => {
                    onMerge(target)
                    setOpen(false)
                  }}
                >
                  {busy ? "merging…" : "merge"}
                </button>
                <button className="btn ghost slim" onClick={() => setTarget(null)}>
                  back
                </button>
              </div>
            </div>
          ) : candidates === null ? (
            <div className="loading">Measuring distances…</div>
          ) : candidates.length === 0 ? (
            <div className="empty">No other clusters to merge into.</div>
          ) : (
            candidates.map((candidate) => (
              <button
                key={candidate.id}
                className="merge-candidate"
                onClick={() => setTarget(candidate)}
              >
                <span
                  className="avatar"
                  style={{ background: `hsl(${hue(candidate.id)} 45% 28%)` }}
                >
                  {initials(candidate.name, candidate.id)}
                </span>
                <span className="merge-candidate-main">
                  <span>
                    {candidate.name ?? (
                      <span className="row-dim">unlabeled · {shortId(candidate.id)}</span>
                    )}
                  </span>
                  <span className="row-sub">
                    {candidate.embeddings} voice-print
                    {candidate.embeddings === 1 ? "" : "s"}
                  </span>
                </span>
                <span
                  className={`distance ${candidate.distance >= FAR_DISTANCE ? "far" : ""}`}
                >
                  {candidate.distance.toFixed(2)} apart
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}

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
  const [merging, setMerging] = useState(false)
  const [namePrompt, setNamePrompt] = useState<{
    loserId: string
    survivorId: string
    name: string
  } | null>(null)

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
      setSpeakers((prev) => (prev ?? []).map((s) => (s.id === data.id ? data : s)))
      // Same name on two clusters usually means one split voice — offer to
      // heal it. The just-renamed cluster merges into the pre-existing one.
      const twin = (speakers ?? []).find(
        (s) => s.id !== data.id && s.name === data.name && s.model === data.model,
      )
      setNamePrompt(
        data.name && twin
          ? { loserId: data.id, survivorId: twin.id, name: data.name }
          : null,
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

  const merge = async (loserId: string, survivor: { id: string }) => {
    setMerging(true)
    const { data } = await api.POST("/v1/speakers/{speaker_id}/merge", {
      params: { path: { speaker_id: loserId } },
      body: { into_speaker_id: survivor.id },
    })
    setMerging(false)
    if (!data) return
    setSpeakers((prev) =>
      (prev ?? []).filter((s) => s.id !== loserId).map((s) => (s.id === data.id ? data : s)),
    )
    // Both transcript caches are stale: the survivor gained utterances.
    setTranscripts((prev) => {
      const next = { ...prev }
      delete next[loserId]
      delete next[data.id]
      return next
    })
    setExpanded((prev) => (prev === loserId ? null : prev))
    setNamePrompt((prev) => (prev && prev.loserId === loserId ? null : prev))
  }

  if (!speakers) return <div className="loading">Loading speakers…</div>
  if (speakers.length === 0)
    return <div className="empty">No speaker clusters yet — the clustering tier builds them.</div>

  return (
    <div className="list">
      {speakers.map((speaker) => (
        <div key={speaker.id} className="speaker-card">
          <div className="speaker-row">
            <span
              className="avatar"
              style={{ background: `hsl(${hue(speaker.id)} 45% 28%)` }}
            >
              {initials(speaker.name, speaker.id)}
            </span>
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
                  {speaker.name ?? (
                    <span className="row-dim">unlabeled · {shortId(speaker.id)}</span>
                  )}
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
                <MergeMenu
                  speaker={speaker}
                  busy={merging}
                  onMerge={(survivor) => void merge(speaker.id, survivor)}
                />
              </>
            )}
          </div>
          {namePrompt?.loserId === speaker.id && (
            <div className="merge-hint">
              <span>
                another cluster is also named “{namePrompt.name}” — merge them?
              </span>
              <button
                className="btn primary slim"
                disabled={merging}
                onClick={() => void merge(namePrompt.loserId, { id: namePrompt.survivorId })}
              >
                {merging ? "merging…" : "merge"}
              </button>
              <button className="btn ghost slim" onClick={() => setNamePrompt(null)}>
                dismiss
              </button>
            </div>
          )}
          {expanded === speaker.id && (
            <div className="speaker-transcripts">
              {(transcripts[speaker.id] ?? []).length === 0 ? (
                <div className="empty">No transcripts.</div>
              ) : (
                (transcripts[speaker.id] ?? []).map((t) => {
                  const clip = {
                    key: t.artifact_id,
                    sessionId: t.session_id,
                    startMs: t.start_ms,
                    endMs: t.end_ms,
                  }
                  return (
                    <div
                      key={t.artifact_id}
                      className="row clickable"
                      onClick={() => onOpen(t.session_id, t.start_ms)}
                      title="open in session"
                    >
                      <ClipButton clip={clip} />
                      <span className="event-time as-span">{fmtClock(t.start_ms)}</span>
                      <span className="transcript-text">{t.text}</span>
                      <ClipProgress clip={clip} />
                    </div>
                  )
                })
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
