import { useEffect, useRef, useState } from "react"
import { api, type SpeakerRead } from "../api/client"
import { shortId } from "../format"
import { hue, localLabelTail } from "../speakers"

// A voice, wearable anywhere: colour dot + best-known name. Clicking opens
// the labeling popover — rename the cluster in place, or declare "this is …"
// an existing named speaker (which merges this cluster into it). Voices the
// clustering tier hasn't resolved yet get an explainer instead of controls.
export default function SpeakerChip({
  clusterId,
  name,
  localLabel,
  onChanged,
}: {
  clusterId: string | null | undefined
  name: string | null | undefined
  localLabel?: string | null
  onChanged: () => void
}) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLSpanElement | null>(null)

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

  const colorKey = clusterId ?? localLabel ?? "?"
  const text =
    name ??
    (localLabel ? localLabelTail(localLabel) : clusterId ? shortId(clusterId) : "voice")
  const resolved = clusterId != null && name != null

  return (
    <span className="filter-menu speaker-chip-root" ref={root}>
      <button
        className={`speaker-chip ${resolved ? "" : "unresolved"}`}
        title={resolved ? `${text} — click to relabel` : "click to label this voice"}
        onClick={(event) => {
          event.stopPropagation()
          setOpen((o) => !o)
        }}
      >
        <span
          className="speaker-dot"
          style={{ background: `hsl(${hue(colorKey)} 62% 58%)` }}
        />
        {text}
      </button>
      {open && (
        <LabelPop
          clusterId={clusterId ?? null}
          currentName={name ?? null}
          onDone={(changed) => {
            setOpen(false)
            if (changed) onChanged()
          }}
        />
      )}
    </span>
  )
}

function LabelPop({
  clusterId,
  currentName,
  onDone,
}: {
  clusterId: string | null
  currentName: string | null
  onDone: (changed: boolean) => void
}) {
  const [draft, setDraft] = useState(currentName ?? "")
  const [candidates, setCandidates] = useState<SpeakerRead[] | null>(null)
  const [confirmTarget, setConfirmTarget] = useState<SpeakerRead | null>(null)
  const [busy, setBusy] = useState(false)

  // "This is …" candidates: named clusters in the same embedding space.
  useEffect(() => {
    if (!clusterId) return
    let cancelled = false
    void (async () => {
      const me = await api.GET("/v1/speakers/{speaker_id}", {
        params: { path: { speaker_id: clusterId } },
      })
      const all = await api.GET("/v1/speakers", { params: { query: { limit: 200 } } })
      if (cancelled) return
      const model = me.data?.model
      setCandidates(
        (all.data?.items ?? []).filter(
          (s) => s.id !== clusterId && s.name != null && (!model || s.model === model),
        ),
      )
    })()
    return () => {
      cancelled = true
    }
  }, [clusterId])

  if (!clusterId) {
    return (
      <div className="filter-pop chip-pop">
        <div className="chip-pop-hint">
          This voice hasn't been matched to a speaker cluster yet — labels attach to
          clusters, so there's nothing to name here until the clustering tier catches
          up.
        </div>
      </div>
    )
  }

  const save = async () => {
    const name = draft.trim() || null
    if (name === currentName) {
      onDone(false)
      return
    }
    setBusy(true)
    const { error } = await api.POST("/v1/speakers/{speaker_id}/label", {
      params: { path: { speaker_id: clusterId } },
      body: { name },
    })
    setBusy(false)
    onDone(!error)
  }

  const mergeInto = async (target: SpeakerRead) => {
    setBusy(true)
    const { error } = await api.POST("/v1/speakers/{speaker_id}/merge", {
      params: { path: { speaker_id: clusterId } },
      body: { into_speaker_id: target.id },
    })
    setBusy(false)
    onDone(!error)
  }

  return (
    <div className="filter-pop chip-pop" onClick={(e) => e.stopPropagation()}>
      {confirmTarget ? (
        <div className="merge-confirm">
          <div>
            Fold this voice into <strong>{confirmTarget.name}</strong>? All its
            voice-prints move over.
          </div>
          <div className="merge-actions">
            <button
              className="btn primary slim"
              disabled={busy}
              onClick={() => void mergeInto(confirmTarget)}
            >
              {busy ? "merging…" : "yes, same person"}
            </button>
            <button className="btn ghost slim" onClick={() => setConfirmTarget(null)}>
              back
            </button>
          </div>
        </div>
      ) : (
        <>
          <form
            className="chip-pop-name"
            onSubmit={(e) => {
              e.preventDefault()
              void save()
            }}
          >
            <input
              className="input slim"
              placeholder="name this voice…"
              value={draft}
              autoFocus
              onChange={(e) => setDraft(e.target.value)}
            />
            <button className="btn primary slim" disabled={busy}>
              save
            </button>
          </form>
          {candidates === null ? (
            <div className="loading">Loading speakers…</div>
          ) : candidates.length > 0 ? (
            <>
              <div className="chip-pop-hint">or, this is:</div>
              {candidates.map((candidate) => (
                <button
                  key={candidate.id}
                  className="merge-candidate"
                  disabled={busy}
                  onClick={() => setConfirmTarget(candidate)}
                >
                  <span
                    className="speaker-dot"
                    style={{ background: `hsl(${hue(candidate.id)} 62% 58%)` }}
                  />
                  <span className="merge-candidate-main">
                    <span>{candidate.name}</span>
                  </span>
                </button>
              ))}
            </>
          ) : null}
        </>
      )}
    </div>
  )
}
